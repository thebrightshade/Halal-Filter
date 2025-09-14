from __future__ import annotations
from enum import Enum
import sys
import time
from pathlib import Path
import typer
import pandas as pd
from .core import HalalFilter


class DotPolicy(str, Enum):
    translate = "translate"
    skip = "skip"
    keep = "keep"


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
    return df[col].astype(str).str.strip().str.upper().loc[lambda s: s != ""].tolist()


@app.command()
def screen(
    symbols: str = typer.Option(
        None, help="Comma-separated tickers, e.g. 'AAPL,MSFT,TSLA'. Optional if --symbols-file is used."),
    symbols_file: str | None = typer.Option(
        None, help="Path to CSV of symbols."),
    symbols_column: str | None = typer.Option(
        None, help="Optional column name for --symbols-file."),
    allowlist: str | None = typer.Option(
        None, help="CSV (one ticker per line or column 'symbol')."),
    denylist: str | None = typer.Option(
        None, help="CSV (one ticker per line or column 'symbol')."),
    whitelist: str | None = typer.Option(
        None, help="Output CSV path for halal whitelist (approved only)."),
    full_report: str | None = typer.Option(
        None, help="Output CSV path for full screening report (all tickers)."),
    out_dir: str | None = typer.Option(
        None, help="If set, writes whitelist to <out_dir>/halal.csv and report to <out_dir>/report.csv."),
    sleep_ms: int = typer.Option(0, help="Sleep between symbols (ms)."),
    quiet: bool = typer.Option(
        False, help="Suppress table print when writing files."),
    dot_policy: DotPolicy = typer.Option(
        DotPolicy.skip, "--dot-policy", help="Handle tickers containing '.' (translate|skip|keep)"),
    # batch helper
    batch_size: int = typer.Option(
        0, help="If >0, split the list into chunks of this size."),
    max_batches: int | None = typer.Option(
        None, help="Optional cap on how many chunks to process."),
    batch_sleep_ms: int = typer.Option(0, help="Sleep between batches (ms)."),
) -> None:
    # Resolve symbols
    syms: list[str] = []
    if symbols:
        syms.extend([s.strip().upper()
                    for s in symbols.split(",") if s.strip()])
    if symbols_file:
        syms.extend(_load_symbols_from_file(symbols_file, symbols_column))
    syms = sorted(set([s for s in syms if s]))
    if not syms:
        raise typer.BadParameter(
            "No symbols provided. Use --symbols or --symbols-file.")

    # Resolve outputs
    if out_dir:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        if not whitelist:
            whitelist = str(outp / "halal.csv")
        if not full_report:
            full_report = str(outp / "report.csv")

    hf = HalalFilter(sleep_between_calls=max(
        0, sleep_ms) / 1000.0, quiet=quiet)
    if allowlist:
        hf.load_allowlist_csv(allowlist)
    if denylist:
        hf.load_denylist_csv(denylist)

    # ---- Single pass or batch mode ----
    frames: list[pd.DataFrame] = []
    if batch_size and batch_size > 0:
        total = len(syms)
        chunks = [syms[i:i + batch_size] for i in range(0, total, batch_size)]
        if max_batches is not None:
            chunks = chunks[:max_batches]
        for i, chunk in enumerate(chunks, start=1):
            if not quiet:
                print(
                    f"[Batch {i}/{len(chunks)}] Screening {len(chunk)} symbols (dot_policy={dot_policy.value})...")
            part = hf.screen_symbols(
                chunk, use_allowlist=True, dot_policy=dot_policy.value)
            frames.append(part)
            if batch_sleep_ms > 0 and i < len(chunks):
                time.sleep(batch_sleep_ms / 1000.0)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = hf.screen_symbols(syms, use_allowlist=True,
                               dot_policy=dot_policy.value)

    # Write outputs
    if full_report:
        Path(full_report).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(full_report, index=False)
        if not quiet:
            print(f"Wrote full report: {full_report}")

    if whitelist:
        Path(whitelist).parent.mkdir(parents=True, exist_ok=True)
        df[df["is_halal"]].loc[:, ["symbol"]].drop_duplicates().to_csv(
            whitelist, index=False)
        if not quiet:
            print(f"Wrote whitelist: {whitelist}")

    # Print table if not quiet and no explicit outputs
    if not quiet and not (whitelist or full_report):
        cols = ["symbol", "is_halal", "reason", "sector", "industry",
                "debt_to_assets", "debt_to_mktcap", "cashinv_to_mktcap"]
        present = [c for c in cols if c in df.columns]
        print(df[present].sort_values(["is_halal", "symbol"],
              ascending=[False, True]).to_string(index=False))

    # Summary + CI-friendly exit code
    total = len(df)
    approved = int(df["is_halal"].sum()) if "is_halal" in df.columns else 0
    blocked = max(0, total - approved)
    if not quiet:
        print(
            f"\nSummary: total={total}  approved={approved}  blocked={blocked}")
    if blocked > 0:
        sys.exit(1)


def main():
    typer.run(screen)


if __name__ == "__main__":
    main()
