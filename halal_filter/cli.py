from __future__ import annotations
import sys
from pathlib import Path
import typer
import pandas as pd
from .core import HalalFilter

app = typer.Typer(add_help_option=True)

def _load_symbols_from_file(path: str | Path, column: str | None = None) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"Symbols file not found: {p}")
    df = pd.read_csv(p)
    if column and column in df.columns:
        col = column
    else:
        for candidate in ("symbol", "ticker", "Symbol", "Ticker"):
            if candidate in df.columns:
                col = candidate
                break
        else:
            col = df.columns[0]
    return (
        df[col].astype(str).str.strip().str.upper().loc[lambda s: s != ""].tolist()
    )

@app.command()
def screen(
    symbols: str = typer.Option(None, help="Comma-separated tickers, e.g. 'AAPL,MSFT,TSLA'. Optional if --symbols-file is used."),
    symbols_file: str | None = typer.Option(None, help="Path to CSV of symbols."),
    symbols_column: str | None = typer.Option(None, help="Optional column name for --symbols-file."),
    allowlist: str | None = typer.Option(None, help="CSV (one ticker per line or column 'symbol')."),
    denylist: str | None  = typer.Option(None, help="CSV (one ticker per line or column 'symbol')."),
    whitelist: str | None = typer.Option(None, help="Output CSV path for halal whitelist (approved only)."),
    full_report: str | None = typer.Option(None, help="Output CSV path for full screening report (all tickers)."),
    out_dir: str | None = typer.Option(None, help="If set, writes whitelist to <out_dir>/halal.csv and report to <out_dir>/report.csv."),
    sleep_ms: int = typer.Option(0, help="Sleep between symbols (ms)."),
    quiet: bool = typer.Option(False, help="Suppress table print when writing files."),
) -> None:
    # Resolve symbols
    syms: list[str] = []
    if symbols:
        syms.extend([s.strip().upper() for s in symbols.split(",") if s.strip()])
    if symbols_file:
        syms.extend(_load_symbols_from_file(symbols_file, symbols_column))
    syms = sorted(set([s for s in syms if s]))
    if not syms:
        raise typer.BadParameter("No symbols provided. Use --symbols or --symbols-file.")

    # Resolve outputs
    if out_dir:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        if not whitelist:
            whitelist = str(outp / "halal.csv")
        if not full_report:
            full_report = str(outp / "report.csv")

    hf = HalalFilter(sleep_between_calls=max(0, sleep_ms) / 1000.0)
    if allowlist: hf.load_allowlist_csv(allowlist)
    if denylist:  hf.load_denylist_csv(denylist)

    df = hf.screen_symbols(syms, use_allowlist=True)

    # Write outputs
    if full_report:
        Path(full_report).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(full_report, index=False)
        if not quiet: print(f"Wrote full report: {full_report}")
    if whitelist:
        Path(whitelist).parent.mkdir(parents=True, exist_ok=True)
        df[df["is_halal"]].loc[:, ["symbol"]].drop_duplicates().to_csv(whitelist, index=False)
        if not quiet: print(f"Wrote whitelist: {whitelist}")

    # Print table if not quiet and no explicit outputs
    if not quiet and not (whitelist or full_report):
        cols = ["symbol","is_halal","reason","sector","industry","debt_to_assets","debt_to_mktcap","cashinv_to_mktcap"]
        print(df[cols].sort_values(["is_halal","symbol"], ascending=[False,True]).to_string(index=False))

    # Summary + CI-friendly exit code
    total = len(df); approved = int(df["is_halal"].sum()); blocked = total - approved
    if not quiet: print(f"\nSummary: total={total}  approved={approved}  blocked={blocked}")
    if blocked > 0: sys.exit(1)

def main():
    # single-command style: allow `halal-filter ...` (no subcommand)
    typer.run(screen)

if __name__ == "__main__":
    # keeps subcommand style available if you ever do `python -m halal_filter.cli screen ...`
    app()
