# halal_filter/halal_filter/core.py
from __future__ import annotations

import time
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import yfinance as yf
import requests
from requests import exceptions as req_exc
import urllib.error as urlerr


# --------- simple helpers ---------
def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


# Sectors/industries that should be excluded outright
_HARAM_NAME_KWS = tuple(s.lower() for s in [
    # conventional finance & real estate
    "bank", "banks", "banking", "capital markets", "brokerage", "investment management",
    "asset management", "mortgage", "consumer finance", "thrifts", "insurance",
    "reit", "real estate",
    # sin industries
    "tobacco", "alcohol", "brewers", "distillers", "wine", "spirits", "casino", "gambling",
    "gaming", "lottery", "adult entertainment", "porn", "defense", "weapons", "aerospace & defense",
])

# Safety-net for warrants/rights/units
_SUFFIX_RE = re.compile(r"[-./](W|WS|WT|WTS|RT|R|U|UN)$", re.IGNORECASE)


@dataclass
class RowResult:
    symbol: str
    is_halal: bool
    reason: str
    sector: str
    industry: str
    debt_to_assets: float
    debt_to_mktcap: float
    cashinv_to_mktcap: float


def _map_symbol(sym: str, policy: str) -> str | None:
    if "." in sym:
        if policy == "skip":
            return None
        if policy == "translate":
            return sym.replace(".", "-")
    return sym


class HalalFilter:
    """
    Robust halal screen with quiet HTTP handling.
    - Suppresses HTTP 404 spam by default (quiet=True).
    - Continues on data gaps; fills with 0 and flags reason.
    - Basic industry exclusion + 3 ratio checks (tunable below).
    """

    def __init__(self, sleep_between_calls: float = 0.0, quiet: bool = True):
        self.sleep_between_calls = float(sleep_between_calls)
        self.quiet = bool(quiet)
        self.allow: set[str] = set()
        self.deny: set[str] = set()
        # error counters
        self._err_404 = 0
        self._err_other = 0
        # ratio thresholds
        self.max_debt_to_assets = 0.30
        self.max_debt_to_mktcap = 0.33
        self.max_cashinv_to_mktcap = 0.33

    # ---------- allow/deny lists ----------
    def _read_symbols_csv(self, path: str) -> list[str]:
        df = pd.read_csv(path)
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        return df[col].astype(str).str.upper().str.strip().tolist()

    def load_allowlist_csv(self, path: str) -> None:
        self.allow = set(self._read_symbols_csv(path))

    def load_denylist_csv(self, path: str) -> None:
        self.deny = set(self._read_symbols_csv(path))

    # ---------- fetchers (quiet on 404s) ----------
    def _fetch_profile(self, symbol: str) -> Tuple[str, str]:
        """
        Returns (sector, industry). Empty strings if unavailable.
        Uses yfinance .get_info() first, falls back to .info.
        """
        t = yf.Ticker(symbol)
        info: Dict[str, Any] = {}

        try:
            info = t.get_info()
        except req_exc.HTTPError as e:
            if getattr(e, "response", None) is not None and e.response.status_code == 404:
                self._err_404 += 1
                return "", ""
            self._err_other += 1
            if not self.quiet:
                print(f"[warn] {symbol} get_info error: {e}")
            return "", ""
        except urlerr.HTTPError as e:
            if getattr(e, "code", None) == 404:
                self._err_404 += 1
                return "", ""
            self._err_other += 1
            if not self.quiet:
                print(f"[warn] {symbol} get_info error: {e}")
            return "", ""
        except Exception as e:
            # try legacy .info
            try:
                info = t.info
            except Exception:
                self._err_other += 1
                if not self.quiet:
                    print(f"[warn] {symbol} info error: {e}")
                return "", ""

        sector = str(info.get("sector") or "").strip()
        industry = str(info.get("industry") or info.get(
            "industryKey") or "").strip()
        return sector, industry

    def _fetch_fundamentals(self, symbol: str) -> Tuple[float, float, float, float]:
        """
        Returns (total_debt, total_assets, cash_plus_st_inv, market_cap).
        Any missing fields become 0.
        """
        t = yf.Ticker(symbol)

        # balance sheet (prefer get_balance_sheet)
        bs = pd.DataFrame()
        try:
            bs = t.get_balance_sheet()
        except Exception:
            try:
                bs = t.balance_sheet
            except Exception:
                bs = pd.DataFrame()

        if isinstance(bs, pd.DataFrame) and not bs.empty:
            # Use the last reported column
            col = bs.columns[0]
            # normalize row labels
            idx = {str(i).strip().lower(): i for i in bs.index}

            def _pick(*cands):
                for k in cands:
                    key = k.lower()
                    if key in idx:
                        return _safe_float(bs.loc[idx[key], col])
                return 0.0

            total_assets = _pick("Total Assets", "TotalAssets", "Total assets")
            total_debt = _pick("Total Debt", "TotalDebt",
                               "Long Term Debt", "LongTermDebt")
            cash_eq = _pick("Cash And Cash Equivalents",
                            "CashAndCashEquivalents", "Cash")
            st_inv = _pick("Short Term Investments", "ShortTermInvestments")
            cash_plus_st = cash_eq + st_inv
        else:
            total_assets = total_debt = cash_plus_st = 0.0

        # market cap
        mktcap = 0.0
        try:
            fi = t.fast_info  # cache-friendly
            mc = getattr(fi, "market_cap", None)
            mktcap = _safe_float(
                mc if mc is not None else fi.get("market_cap", 0.0))
        except Exception:
            pass
        if not mktcap:
            try:
                hist = t.history(period="5d", interval="1d")
                last = float(hist["Close"].iloc[-1])
                shares = _safe_float(t.get_shares_full(period="1y").iloc[-1])
                mktcap = last * shares if (last and shares) else 0.0
            except Exception:
                pass

        return float(total_debt), float(total_assets), float(cash_plus_st), float(mktcap)

    # ---------- decision rules ----------
    def _is_haram_by_name(self, sector: str, industry: str, symbol: str) -> bool:
        s, i = _lower(sector), _lower(industry)
        if _SUFFIX_RE.search(symbol):
            return True
        text = f"{s} {i}"
        return any(kw in text for kw in _HARAM_NAME_KWS)

    def _screen_one(self, sym: str) -> RowResult:
        S = sym.upper()

        # explicit lists first
        if S in self.deny:
            return RowResult(S, False, "denylist", "", "", 0.0, 0.0, 0.0)
        if S in self.allow:
            # still try to fetch for reporting, but allow regardless
            sector, industry = self._fetch_profile(S)
            debt, assets, cashst, mktcap = self._fetch_fundamentals(S)
            return RowResult(S, True, "allowlist",
                             sector, industry,
                             self._ratio(debt, assets),
                             self._ratio(debt, mktcap),
                             self._ratio(cashst, mktcap))

        sector, industry = self._fetch_profile(S)

        # Industry/sector exclusion (if fetch fails, we rely on ratios)
        if sector or industry:
            if self._is_haram_by_name(sector, industry, S):
                return RowResult(S, False, "industry_excluded", sector, industry, 0.0, 0.0, 0.0)

        debt, assets, cashst, mktcap = self._fetch_fundamentals(S)

        # Ratios (0 if missing)
        r_da = self._ratio(debt, assets)
        r_dm = self._ratio(debt, mktcap)
        r_cm = self._ratio(cashst, mktcap)

        reasons = []
        halal = True
        if assets and r_da > self.max_debt_to_assets:
            halal = False
            reasons.append(f"debt/assets>{self.max_debt_to_assets:.2f}")
        if mktcap and r_dm > self.max_debt_to_mktcap:
            halal = False
            reasons.append(f"debt/mktcap>{self.max_debt_to_mktcap:.2f}")
        if mktcap and r_cm > self.max_cashinv_to_mktcap:
            halal = False
            reasons.append(f"cashinv/mktcap>{self.max_cashinv_to_mktcap:.2f}")

        if not (assets and mktcap):
            reasons.append("data_sparse")

        return RowResult(S, halal, "|".join(reasons) if reasons else "ok",
                         sector, industry, r_da, r_dm, r_cm)

    @staticmethod
    def _ratio(num: float, den: float) -> float:
        if den and den != 0.0:
            return float(num) / float(den)
        return 0.0

    # ---------- public API ----------
    def screen_symbols(self, symbols: list[str], use_allowlist: bool = True, dot_policy: str = "skip") -> pd.DataFrame:
        """
        dot_policy: "skip" | "translate" | "keep"
        - We map each symbol for fetching, but keep the original in the output.
        """
        # Build fetch list + map back to original
        fetch_to_orig: dict[str, str] = {}
        fetch_list: list[str] = []
        for s in symbols:
            orig = s.strip().upper()
            if not orig:
                continue
            mapped = _map_symbol(orig, dot_policy)
            if not mapped:
                continue
            if mapped not in fetch_to_orig:
                fetch_to_orig[mapped] = orig
                fetch_list.append(mapped)

        rows: list[RowResult] = []
        for i, fetch_sym in enumerate(fetch_list, 1):
            try:
                rr = self._screen_one(fetch_sym)
                # map symbol back to original for output
                rr.symbol = fetch_to_orig.get(fetch_sym, rr.symbol)
                rows.append(rr)
            except Exception as e:
                self._err_other += 1
                if not self.quiet:
                    print(f"[warn] {fetch_sym}: {e}")
            if self.sleep_between_calls > 0:
                time.sleep(self.sleep_between_calls)

        df = pd.DataFrame([r.__dict__ for r in rows])
        if df.empty:
            return df

        # nice ordering
        cols = ["symbol", "is_halal", "reason", "sector", "industry",
                "debt_to_assets", "debt_to_mktcap", "cashinv_to_mktcap"]
        df = df.reindex(columns=cols)

        # final summary (quietly suppressed 404s)
        suppressed = {}
        if self._err_404:
            suppressed["http_404_suppressed"] = int(self._err_404)
        if self._err_other:
            suppressed["other_fetch_errors"] = int(self._err_other)
        if suppressed and not self.quiet:
            print({"fetch_warn": suppressed})
        return df
