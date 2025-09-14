from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Literal
import typer
import pandas as pd
from .core import HalalFilter

DotPolicy = Literal["translate", "skip", "keep"]


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
        df[col].astype(str).str.strip().str.upper(
        ).loc[lambda s: s != ""].tolist()
    )


def _apply_dot_policy(symbols: list[str], policy: DotPolicy):
    """
    Returns (fetch_list, fetch_to_original_map)

    - translate: replace '.' with '-' for fetch; keep original for output
    - skip: drop any symbol containing '.'
    - keep: use symbols as-is
    """
    fetch_to_orig: dict[str, str] = {}
    fetch_list: list[str] = []

    for sym in symbols:
        if "." in sym:
            if policy == "skip":
                continue
            elif policy == "translate":
                fetch_sym = sym.replace(".", "-")
            else:  # keep
                fetch_sym = sym
        else:
            fetch_sym = sym

        # Deduplicate if multiple originals map to same fetch (rare)
        if fetch_sym not in fetch_to_orig:
            fetch_to_orig[fetch_sym] = sym
            fetch_list.append(fetch_sym)

    return fetch_list, fetch_to_orig


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
    # Batch mode
    batch_size: int = typer.Option(
        0, help="If > 0, split symbols into batches of this size and run sequentially."),
    batch_sleep_ms: int = typer.Option(0, help="Sleep between batches (ms)."),
    max_batches: int | None = typer.Option(
        None, help="Optional cap on number of batches (for testing)."),
    # NEW: dot ticker policy
    dot_policy: DotPolicy = typer.Option(
        "translate", help="How to handle dot tickers (e.g., BRK.B): translate | skip | keep"),
) -> None:
    # Resolve symbols (originals)
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

    # Apply dot policy → build fetch list + map back to originals
    fetch_list, fetch_to_orig = _apply_dot_policy(syms, dot_policy)

    # Outputs
    if out_dir:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        if not whitelist:
            whitelist = str(outp / "halal.csv")
        if not full_report:
            full_report = str(outp / "report.csv")

    hf = HalalFilter(sleep_between_calls=max(0, sleep_ms) / 1000.0)
    if allowlist:
        hf.load_allowlist_csv(allowlist)
    if denylist:
        hf.load_denylist_csv(denylist)

    # ---- Batch or single pass on FETCH symbols ----
    all_frames: list[pd.DataFrame] = []

    if batch_size and batch_size > 0:
        total = len(fetch_list)
        chunks = [fetch_list[i:i + batch_size]
                  for i in range(0, total, batch_size)]
        if max_batches is not None:
            chunks = chunks[:max_batches]

        for i, chunk in enumerate(chunks, start=1):
            if not quiet:
                print(
                    f"[Batch {i}/{len(chunks)}] Screening {len(chunk)} symbols (dot_policy={dot_policy})...")
            df_part = hf.screen_symbols(chunk, use_allowlist=True)
            all_frames.append(df_part)
            if batch_sleep_ms > 0 and i < len(chunks):
                time.sleep(batch_sleep_ms / 1000.0)

        df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(
            columns=["symbol", "is_halal"])
    else:
        df = hf.screen_symbols(fetch_list, use_allowlist=True)

    # Map symbols in the result back to the ORIGINALS
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].map(fetch_to_orig).fillna(df["symbol"])

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
        present_cols = [c for c in cols if c in df.columns]
        print(df[present_cols].sort_values(["is_halal", "symbol"],
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
    # single-command style: allow `halal-filter ...`
    typer.run(screen)


if __name__ == "__main__":
    main()
