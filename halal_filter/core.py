"""
halal_filter.core — Personal-use halal stock screener (industry + ratios + allow/deny)

Features
--------
1) Sector/industry exclusion (movies/TV/music/streaming, gambling, alcohol, pork, tobacco,
   adult, weapons/defense, conventional finance, cannabis). We scan ONLY sector/industry
   for general prohibitions; business summary is scanned *only* for cannabis terms to catch
   misclassified pharma tickers.
2) Financial ratio screens (AAOIFI-style defaults; configurable):
     - interest-bearing debt / total assets ≤ 33%
     - interest-bearing debt / market cap ≤ 33%
     - (cash + short-term investments) / market cap ≤ 33%
3) Optional allowlist overlay (e.g., Islamic ETF constituents like SPUS/HLAL).
4) Optional denylist overlay to force-exclude edge cases.
5) Fail-closed semantics: missing/ambiguous fundamentals => excluded.

Usage (programmatic)
--------------------
from halal_filter import HalalFilter
hf = HalalFilter()
hf.load_allowlist_csv("data/allowlist.csv")  # optional
hf.load_denylist_csv("data/denylist.csv")    # optional
ok = hf.ok("AAPL")
df = hf.screen_symbols(["AAPL","MSFT","TSLA"])
"""

from __future__ import annotations

import dataclasses as _dc
from typing import Dict, List, Optional
import math
import re
import time

import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Please `pip install yfinance pandas` to use halal_filter") from e


# -----------------------------
# Retry & safe access helpers
# -----------------------------
def _with_retry(fn, tries: int = 2, wait: float = 0.3):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(wait)
    raise last


def _safe_info(ticker):
    # Prefer get_info(); .info may be flaky on some versions.
    try:
        return _with_retry(ticker.get_info) or {}
    except Exception:
        try:
            return ticker.info or {}
        except Exception:
            return {}


def _safe_balance_sheet(ticker):
    # Prefer annual; fallback to quarterly
    try:
        bs = _with_retry(lambda: ticker.balance_sheet)
    except Exception:
        bs = None
    if bs is None or bs is ... or (hasattr(bs, "empty") and bs.empty):
        try:
            bs = _with_retry(lambda: ticker.quarterly_balance_sheet)
        except Exception:
            bs = None
    return bs if (bs is not None and hasattr(bs, "empty") and not bs.empty) else None


def _safe_market_cap(ticker, info: dict):
    mc = info.get("marketCap")
    if mc is None or not mc or (isinstance(mc, float) and math.isnan(mc)):
        try:
            fi = getattr(ticker, "fast_info", None)
            if fi:
                mc = getattr(fi, "market_cap", None)
        except Exception:
            mc = None
    return mc


def _contains_kw(text: str, keywords: set[str]):
    """Return (True, kw) if any keyword matches as a whole word in text."""
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return True, kw
    return False, None


def _first_available(df: pd.DataFrame, names: List[str], col):
    for name in names:
        try:
            val = df.loc[name, col]
            return float(val)
        except Exception:
            continue
    return None


# -----------------------------
# Config (tweak to your standard)
# -----------------------------
DEBT_ROW_NAMES = [
    "Short Long Term Debt",    # yfinance naming quirk
    "Short Term Debt",
    "Long Term Debt",
    "Long Term Debt Noncurrent",
    "Current Debt",
]

ASSETS_ROW_NAMES = ["Total Assets", "Total assets"]
CASH_ROW_NAMES = [
    "Cash",
    "Cash And Cash Equivalents",
    "Cash And Short Term Investments",
]
SHORT_INV_ROW_NAMES = [
    "Short Term Investments",
    "Other Short Term Investments",
]

AAOIFI_MAX_DEBT_TO_ASSETS = 0.33
AAOIFI_MAX_CASH_INV_TO_MKT_CAP = 0.33
AAOIFI_MAX_NONCOMPLIANT_REV = 0.05  # placeholder; requires segment data (not enforced here)

# Strict sector/industry keywords (whole-word; sector/industry ONLY)
PROHIBITED_KEYWORDS = {
    # Conventional finance
    "bank", "insurance", "mortgage", "loan", "lending", "brokerage",
    # Alcohol & tobacco
    "alcohol", "brewery", "distillery", "wine", "spirits", "tobacco", "cigarette", "vape",
    # Pork
    "pork", "ham", "bacon", "swine",
    # Gambling
    "casino", "gambling", "betting", "wagering", "lottery", "sportsbook", "bookmaker",
    # Entertainment / media
    "movie", "film", "cinema", "television", "media", "entertainment",
    # Adult content
    "adult", "porn",
    # Weapons & defense
    "weapons", "defense", "aerospace", "military", "arms",
    # Cannabis / intoxicants
    "cannabis", "marijuana", "hemp", "weed", "dispensary", "psychedelic",
}

# Edge words we also exclude if they appear in sector/industry
EDGE_KEYWORDS = {"music", "streaming"}

# Summary is ignored EXCEPT for cannabis-family words (to catch misclassified pharma)
CANNABIS_KEYWORDS_SUMMARY_ONLY = {"cannabis", "marijuana", "hemp", "weed", "dispensary"}


# -----------------------------
# Result schema
# -----------------------------
@_dc.dataclass
class ScreenResult:
    symbol: str
    is_halal: bool
    reason: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    debt_to_assets: Optional[float] = None
    debt_to_mktcap: Optional[float] = None
    cashinv_to_mktcap: Optional[float] = None
    noncompliant_rev_ratio: Optional[float] = None
    used_allowlist: bool = False


# -----------------------------
# Main filter
# -----------------------------
class HalalFilter:
    """
    Personal-use halal filter with:
      - Sector/industry keyword screen
      - Interest-bearing debt ratio screens
      - Cash+investments/market cap screen
      - Optional allowlist/denylist overlays
    """

    def __init__(
        self,
        max_debt_to_assets: float = AAOIFI_MAX_DEBT_TO_ASSETS,
        max_cashinv_to_mktcap: float = AAOIFI_MAX_CASH_INV_TO_MKT_CAP,
        max_noncompliant_rev: float = AAOIFI_MAX_NONCOMPLIANT_REV,
        prohibited_keywords: Optional[set[str]] = None,
        fail_closed: bool = True,
        sleep_between_calls: float = 0.0,
    ):
        self.max_debt_to_assets = max_debt_to_assets
        self.max_cashinv_to_mktcap = max_cashinv_to_mktcap
        self.max_noncompliant_rev = max_noncompliant_rev
        self.prohibited_keywords = set(prohibited_keywords) if prohibited_keywords else set(PROHIBITED_KEYWORDS)
        self.fail_closed = fail_closed
        self.sleep_between_calls = sleep_between_calls

        self._allowlist: set[str] = set()
        self._denylist: set[str] = set()
        self._cache: Dict[str, ScreenResult] = {}

    # ---------- Public API ----------

    def load_allowlist_csv(self, path: str) -> None:
        """Load tickers from CSV/TXT (one ticker per line or a column named 'symbol')."""
        df = pd.read_csv(path)
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        symbols = df[col].astype(str).str.upper().str.strip().tolist()
        self._allowlist |= set(s for s in symbols if s)

    def load_denylist_csv(self, path: str) -> None:
        df = pd.read_csv(path)
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        symbols = df[col].astype(str).str.upper().str.strip().tolist()
        self._denylist |= set(s for s in symbols if s)

    def ok(self, symbol: str, use_allowlist: bool = True) -> bool:
        """Return True if symbol is halal according to rules (or allowlist if enabled)."""
        symbol = symbol.upper().strip()
        res = self._screen_symbol(symbol, use_allowlist=use_allowlist)
        return res.is_halal

    def screen_symbols(self, symbols: List[str], use_allowlist: bool = True) -> pd.DataFrame:
        """Screen multiple symbols. Returns a DataFrame of ScreenResult rows."""
        out: List[ScreenResult] = []
        for s in symbols:
            s = s.upper().strip()
            if not s:
                continue
            out.append(self._screen_symbol(s, use_allowlist=use_allowlist))
            if self.sleep_between_calls:
                time.sleep(self.sleep_between_calls)
        return pd.DataFrame([_dc.asdict(r) for r in out])

    # ---------- Internals ----------

    def _screen_symbol(self, symbol: str, use_allowlist: bool) -> ScreenResult:
        if symbol in self._cache:
            return self._cache[symbol]

        if symbol in self._denylist:
            res = ScreenResult(symbol=symbol, is_halal=False, reason="denylist")
            self._cache[symbol] = res
            return res

        # Allowlist short-circuit (trusted overlay—e.g., Islamic ETF list you control)
        if use_allowlist and symbol in self._allowlist:
            res = ScreenResult(symbol=symbol, is_halal=True, reason="allowlist", used_allowlist=True)
            self._cache[symbol] = res
            return res

        try:
            ticker = yf.Ticker(symbol)

            info = _safe_info(ticker) or {}
            sector = str(info.get("sector", "") or "").lower()
            industry = str(info.get("industry", "") or "").lower()

            # ---- Industry exclusion (sector/industry ONLY) ----
            sector_industry_text = " ".join([sector, industry])
            hit, kw = _contains_kw(sector_industry_text, self.prohibited_keywords)
            if hit:
                res = ScreenResult(symbol=symbol, is_halal=False, reason=f"industry keyword '{kw}'",
                                   sector=sector or None, industry=industry or None)
                self._cache[symbol] = res
                return res

            hit_edge, kw_edge = _contains_kw(sector_industry_text, EDGE_KEYWORDS)
            if hit_edge:
                res = ScreenResult(symbol=symbol, is_halal=False, reason=f"industry keyword '{kw_edge}'",
                                   sector=sector or None, industry=industry or None)
                self._cache[symbol] = res
                return res

            # ---- Cannabis-only summary scan (to catch misclassified pharma) ----
            summary = str(info.get("longBusinessSummary", "") or "").lower()
            hit_cannabis, kw_c = _contains_kw(summary, CANNABIS_KEYWORDS_SUMMARY_ONLY)
            if hit_cannabis:
                res = ScreenResult(symbol=symbol, is_halal=False, reason=f"summary keyword '{kw_c}'",
                                   sector=sector or None, industry=industry or None)
                self._cache[symbol] = res
                return res

            # ---- Ratios ----
            bs = _safe_balance_sheet(ticker)
            if bs is None:
                return self._fail(symbol, "missing balance sheet", sector, industry)

            latest_col = bs.columns[0]

            # Interest-bearing debt = sum of available debt rows (missing treated as 0)
            interest_bearing_debt = 0.0
            for row in DEBT_ROW_NAMES:
                val = _first_available(bs, [row], latest_col)
                if val is not None:
                    interest_bearing_debt += float(val)

            total_assets = _first_available(bs, ASSETS_ROW_NAMES, latest_col)
            if not (total_assets and total_assets > 0):
                return self._fail(symbol, "missing assets", sector, industry)

            cash = _first_available(bs, CASH_ROW_NAMES, latest_col) or 0.0
            short_inv = _first_available(bs, SHORT_INV_ROW_NAMES, latest_col) or 0.0

            market_cap = _safe_market_cap(ticker, info)
            if market_cap is None or market_cap <= 0:
                return self._fail(symbol, "missing market cap", sector, industry)

            # Ratios per common Shariah practice
            debt_to_assets = float(interest_bearing_debt) / float(total_assets)
            debt_to_mktcap = float(interest_bearing_debt) / float(market_cap)
            cashinv_to_mktcap = float(cash + short_inv) / float(market_cap)

            # Block if ANY debt ratio breaches threshold
            if (debt_to_assets is not None and debt_to_assets > self.max_debt_to_assets) or \
               (debt_to_mktcap is not None and debt_to_mktcap > self.max_debt_to_assets):
                return self._block_ratios(symbol,
                                          debt_to_assets=debt_to_assets,
                                          debt_to_mktcap=debt_to_mktcap,
                                          sector=sector, industry=industry)

            if cashinv_to_mktcap > self.max_cashinv_to_mktcap:
                return self._block_cash(symbol, cashinv_to_mktcap, self.max_cashinv_to_mktcap,
                                        sector, industry)

            res = ScreenResult(
                symbol=symbol, is_halal=True, reason="passed",
                sector=sector or None, industry=industry or None,
                debt_to_assets=debt_to_assets,
                debt_to_mktcap=debt_to_mktcap,
                cashinv_to_mktcap=cashinv_to_mktcap,
                noncompliant_rev_ratio=None, used_allowlist=False
            )
            self._cache[symbol] = res
            return res

        except Exception as e:
            if self.fail_closed:
                res = ScreenResult(symbol=symbol, is_halal=False, reason=f"error: {type(e).__name__}")
                self._cache[symbol] = res
                return res
            else:
                res = ScreenResult(symbol=symbol, is_halal=True, reason="error-ignored")
                self._cache[symbol] = res
                return res

    def _fail(self, symbol: str, why: str, sector: Optional[str], industry: Optional[str]) -> ScreenResult:
        res = ScreenResult(symbol=symbol, is_halal=False, reason=why,
                           sector=sector or None, industry=industry or None)
        self._cache[symbol] = res
        return res

    def _block_ratios(self, symbol: str,
                      debt_to_assets: Optional[float],
                      debt_to_mktcap: Optional[float],
                      sector: Optional[str],
                      industry: Optional[str]) -> ScreenResult:
        parts = []
        if debt_to_assets is not None:
            parts.append(f"debt/assets={debt_to_assets:.3f}")
        if debt_to_mktcap is not None:
            parts.append(f"debt/mktcap={debt_to_mktcap:.3f}")
        reason = f"ratio {' or '.join(parts)} > {self.max_debt_to_assets:.2f}"
        res = ScreenResult(symbol=symbol, is_halal=False, reason=reason,
                           sector=sector or None, industry=industry or None,
                           debt_to_assets=debt_to_assets, debt_to_mktcap=debt_to_mktcap)
        self._cache[symbol] = res
        return res

    def _block_cash(self, symbol: str, value: float, max_allowed: float,
                    sector: Optional[str], industry: Optional[str]) -> ScreenResult:
        res = ScreenResult(symbol=symbol, is_halal=False,
                           reason=f"ratio cash+inv/mktcap={value:.3f} > {max_allowed:.2f}",
                           sector=sector or None, industry=industry or None,
                           cashinv_to_mktcap=value)
        self._cache[symbol] = res
        return res
