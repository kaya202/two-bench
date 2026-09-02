"""
data/fetch.py
=============

yfinance wrapper for pulling price, shares outstanding, beta, and historical
financial-statement data for a ticker, with:

  1. A local pickle cache (data/cache/) so the notebook doesn't re-hit the
     Yahoo Finance endpoints on every run -- those endpoints are
     unauthenticated and rate-limit aggressively, and annual statements
     only change a few times a year anyway.
  2. A manual-override path for any field, since detailed NWC components
     (and sometimes tax rate, D&A, or the statements themselves) aren't
     reliably available via yfinance for every ticker.

This module is the only place in the toolkit that talks to yfinance --
dcf.py, comps.py, etc. only ever see a populated `Company` object.

Data-source disclaimer: yfinance is an unofficial, community-maintained
wrapper around Yahoo Finance's internal (undocumented, unsupported) web
endpoints, not a licensed market-data API. Field availability, exact row
labels, and even whether a given endpoint responds at all can change
without notice -- this module's row-name candidate lists and fallback
logic (below) exist because of exactly that fragility, observed firsthand
while building this toolkit. Treat all fetched figures as
illustrative/educational, not as a data source for real investment or
credit decisions.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

# Zero-install convenience: makes `from valuation.company import ...` resolve
# even when this repo hasn't been `pip install -e .`'d (see pyproject.toml
# for the proper editable-install alternative, which pytest/an IDE will
# prefer -- this fallback exists so `python notebooks/...` or a fresh clone
# works without a packaging step first).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from valuation.company import Company, REQUIRED_FINANCIAL_COLUMNS  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / "cache"

# yfinance's row labels for the same economic line item vary slightly by
# ticker/sector/reporting standard, so each target field is tried against
# a short list of candidate row names, in priority order.
_INCOME_STMT_ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "ebit": ["EBIT", "Operating Income", "Total Operating Income As Reported"],
    "d_and_a": ["Reconciled Depreciation"],
    "tax_provision": ["Tax Provision"],
    "pretax_income": ["Pretax Income"],
    "tax_rate": ["Tax Rate For Calcs"],
}
_CASHFLOW_ROWS = {
    "capex": ["Capital Expenditure", "Purchase Of PPE"],
    "d_and_a": ["Depreciation And Amortization", "Depreciation Amortization Depletion"],
}
_BALANCE_SHEET_ROWS = {
    "nwc": ["Working Capital"],
    "total_debt": ["Total Debt"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "net_debt": ["Net Debt"],
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(ticker: str, kind: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker.upper()}_{kind}.pkl"


def _load_cache(ticker: str, kind: str) -> Any | None:
    path = _cache_path(ticker, kind)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(ticker: str, kind: str, obj: Any) -> None:
    with open(_cache_path(ticker, kind), "wb") as f:
        pickle.dump(obj, f)


def _first_available_row(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """Return the first row (by label) present in `df` from a candidate list, else None."""
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


# ---------------------------------------------------------------------------
# Raw statement pull (cached)
# ---------------------------------------------------------------------------


def _get_raw_statements(ticker: str, use_cache: bool = True, refresh: bool = False) -> dict[str, Any]:
    """
    Pull annual income statement, balance sheet, cashflow statement, and the
    `info` dict for a ticker via yfinance -- or return the cached copy.

    `refresh=True` forces a fresh API pull even if a cache file exists,
    for when you know the underlying data has changed.
    """
    if use_cache and not refresh:
        cached = _load_cache(ticker, "statements")
        if cached is not None:
            return cached

    t = yf.Ticker(ticker)
    data = {
        "income_stmt": t.income_stmt,
        "balance_sheet": t.balance_sheet,
        "cashflow": t.cashflow,
        "info": t.info,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    if use_cache:
        _save_cache(ticker, "statements", data)
    return data


def cache_age(ticker: str) -> Any:
    """
    How long ago (as a `datetime.timedelta`) this ticker's cached data was
    fetched, or None if nothing is cached yet. There is no automatic cache
    expiry (see module docstring) -- annual statements only change a few
    times a year, so a stale cache is usually harmless, but a notebook
    re-run months later could be quoting a materially old share price /
    market cap without any other indication of that. Check this (or pass
    refresh=True) before trusting cached market data in a fresh session.
    """
    cached = _load_cache(ticker, "statements")
    if cached is None or "fetched_at" not in cached:
        return None
    return datetime.now(timezone.utc) - datetime.fromisoformat(cached["fetched_at"])


# ---------------------------------------------------------------------------
# Individual field fetchers (each usable standalone, each override-able)
# ---------------------------------------------------------------------------


def fetch_price(ticker: str, use_cache: bool = True, refresh: bool = False) -> float:
    info = _get_raw_statements(ticker, use_cache, refresh)["info"]
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price is None:
        raise ValueError(f"'{ticker}': yfinance returned no current price; supply share_price manually.")
    return float(price)


def fetch_shares_outstanding(ticker: str, use_cache: bool = True, refresh: bool = False) -> float:
    info = _get_raw_statements(ticker, use_cache, refresh)["info"]
    shares = info.get("sharesOutstanding")
    if shares is None:
        raise ValueError(f"'{ticker}': yfinance returned no shares outstanding; supply manually.")
    return float(shares)


def fetch_beta(ticker: str, use_cache: bool = True, refresh: bool = False) -> float:
    """
    Levered (equity) beta vs. the market, as reported by Yahoo Finance.
    Yahoo does not publicly document its exact beta methodology (window
    length, benchmark index, sampling frequency) -- it's commonly believed
    to approximate a 5-year monthly regression against the S&P 500, but
    that is not an official specification, just an industry-common
    assumption. Used directly as the CAPM beta in dcf.py's cost-of-equity
    calc; override manually if you'd rather use a disclosed-methodology
    source, or an industry-unlevered-and-relevered beta.
    """
    info = _get_raw_statements(ticker, use_cache, refresh)["info"]
    beta = info.get("beta")
    if beta is None:
        raise ValueError(f"'{ticker}': yfinance returned no beta; supply manually.")
    return float(beta)


def fetch_sector(ticker: str, use_cache: bool = True, refresh: bool = False) -> str:
    info = _get_raw_statements(ticker, use_cache, refresh)["info"]
    sector = info.get("sector")
    if sector is None:
        raise ValueError(f"'{ticker}': yfinance returned no sector; supply manually.")
    return str(sector)


def fetch_currency(ticker: str, use_cache: bool = True, refresh: bool = False) -> tuple[str | None, str | None]:
    """
    (quote currency, financial-statement currency) for a ticker.

    These differ for some foreign issuers traded via a USD-quoted ADR while
    reporting financial statements in their home currency -- e.g. SAP
    (currency='USD', financialCurrency='EUR'), confirmed by direct
    inspection of yfinance's output while building this toolkit. See
    `Company.assert_single_currency` for why this matters: combining a
    USD share price/market cap with EUR-denominated EBITDA/net debt
    produces a number wrong by roughly the FX rate, silently.
    """
    info = _get_raw_statements(ticker, use_cache, refresh)["info"]
    return info.get("currency"), info.get("financialCurrency")


def fetch_trailing_pe(ticker: str, use_cache: bool = True, refresh: bool = False) -> float:
    """
    Trailing P/E (price / trailing twelve-month EPS), as reported by Yahoo
    Finance. Used by comps.py alongside EV/EBITDA and EV/Revenue -- unlike
    those two, P/E isn't derivable from `Company`'s financials (which don't
    carry net income/EPS), so it's pulled directly here.
    """
    info = _get_raw_statements(ticker, use_cache, refresh)["info"]
    pe = info.get("trailingPE")
    if pe is None:
        raise ValueError(f"'{ticker}': yfinance returned no trailing P/E; supply manually.")
    return float(pe)


def fetch_net_debt(ticker: str, use_cache: bool = True, refresh: bool = False) -> float:
    """
    Net debt = Total Debt - Cash & Equivalents, for the most recent
    balance-sheet date. Prefers yfinance's own precomputed 'Net Debt' row;
    falls back to Total Debt - Cash if that row isn't present for this
    ticker.
    """
    data = _get_raw_statements(ticker, use_cache, refresh)
    bs = data["balance_sheet"]
    if bs.empty:
        raise ValueError(f"'{ticker}': yfinance returned no balance sheet; supply net_debt manually.")

    net_debt_row = _first_available_row(bs, _BALANCE_SHEET_ROWS["net_debt"])
    if net_debt_row is not None and pd.notna(net_debt_row.iloc[0]):
        return float(net_debt_row.iloc[0])

    total_debt_row = _first_available_row(bs, _BALANCE_SHEET_ROWS["total_debt"])
    cash_row = _first_available_row(bs, _BALANCE_SHEET_ROWS["cash"])
    if total_debt_row is None or cash_row is None:
        raise ValueError(f"'{ticker}': could not derive net debt from balance sheet; supply manually.")
    return float(total_debt_row.iloc[0]) - float(cash_row.iloc[0])


def fetch_financials(ticker: str, years: int = 4, use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    """
    Build a historical financials table matching `REQUIRED_FINANCIAL_COLUMNS`
    (revenue, ebitda, ebit, d_and_a, capex, nwc, tax_rate), indexed by fiscal
    year, from yfinance's annual income statement / balance sheet / cashflow
    statement.

    yfinance typically exposes ~4 fiscal years of annual detail; `years`
    caps how many of the most recent columns are used. Columns with no data
    at all (yfinance sometimes returns a trailing all-NaN year) are dropped.
    Any field yfinance can't supply for a given ticker is left as NaN --
    fill it via `manual_overrides` in `build_company()`.
    """
    data = _get_raw_statements(ticker, use_cache, refresh)
    inc, bs, cf = data["income_stmt"], data["balance_sheet"], data["cashflow"]
    if inc.empty:
        raise ValueError(f"'{ticker}': yfinance returned no income statement; supply financials manually.")

    # yfinance has returned columns most-recent-first in every version
    # tested for this toolkit, but that ordering isn't a documented
    # contract -- sorting explicitly makes "most recent `years` columns"
    # correct regardless, rather than silently pulling the OLDEST years if
    # a future yfinance version ever changes column order.
    fiscal_dates = sorted(inc.columns, reverse=True)[:years]
    rows: dict[int, dict[str, float]] = {}

    for date in fiscal_dates:
        year = int(pd.Timestamp(date).year)
        revenue = _row_value(inc, _INCOME_STMT_ROWS["revenue"], date)
        if pd.isna(revenue):
            continue  # placeholder/empty fiscal-year column -- skip it

        ebitda = _row_value(inc, _INCOME_STMT_ROWS["ebitda"], date)
        ebit = _row_value(inc, _INCOME_STMT_ROWS["ebit"], date)

        # D&A: prefer the income-statement figure, fall back to cashflow's.
        d_and_a = _row_value(inc, _INCOME_STMT_ROWS["d_and_a"], date)
        if pd.isna(d_and_a):
            d_and_a = _row_value(cf, _CASHFLOW_ROWS["d_and_a"], date)

        # Capex is reported by yfinance as a cash *outflow* (negative);
        # the toolkit's convention (see company.py) is a positive spend
        # amount that gets subtracted explicitly in the UFCF build.
        capex_raw = _row_value(cf, _CASHFLOW_ROWS["capex"], date)
        capex = abs(capex_raw) if pd.notna(capex_raw) else float("nan")

        nwc = _row_value(bs, _BALANCE_SHEET_ROWS["nwc"], date)

        tax_rate = _row_value(inc, _INCOME_STMT_ROWS["tax_rate"], date)
        if pd.isna(tax_rate):
            # Fallback: effective tax rate = Tax Provision / Pretax Income.
            tax_provision = _row_value(inc, _INCOME_STMT_ROWS["tax_provision"], date)
            pretax_income = _row_value(inc, _INCOME_STMT_ROWS["pretax_income"], date)
            if pd.notna(tax_provision) and pd.notna(pretax_income) and pretax_income != 0:
                tax_rate = tax_provision / pretax_income

        rows[year] = {
            "revenue": revenue,
            "ebitda": ebitda,
            "ebit": ebit,
            "d_and_a": d_and_a,
            "capex": capex,
            "nwc": nwc,
            "tax_rate": tax_rate,
        }

    if not rows:
        raise ValueError(f"'{ticker}': no usable fiscal years found in yfinance data.")

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(columns=REQUIRED_FINANCIAL_COLUMNS).sort_index()
    return df


def _row_value(df: pd.DataFrame, candidates: list[str], date) -> float:
    row = _first_available_row(df, candidates)
    if row is None or date not in row.index:
        return float("nan")
    val = row[date]
    return float(val) if pd.notna(val) else float("nan")


# ---------------------------------------------------------------------------
# Top-level: build a fully-populated Company, with manual overrides
# ---------------------------------------------------------------------------


def build_company(
    ticker: str,
    name: str | None = None,
    years: int = 4,
    manual_overrides: dict[str, Any] | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> Company:
    """
    Fetch everything needed for a `Company` in one call, degrading
    gracefully field-by-field on fetch failure, then applying
    `manual_overrides` on top of whatever was fetched.

    `manual_overrides` shape (all keys optional):
        {
            "name": "...", "sector": "...",
            "share_price": ..., "shares_outstanding": ..., "net_debt": ...,
            "financials": {2024: {"nwc": 5000, "tax_rate": 0.24}, ...},
        }
    Per-field overrides are applied even where the fetch succeeded, so this
    is also how you'd correct a field yfinance gets subtly wrong (e.g. a
    non-standard fiscal year-end, or a one-off in reported EBIT).
    """
    overrides = manual_overrides or {}

    def _try(label: str, fn, *args):
        # Every fetch_* call here can fail for many different reasons
        # (network, rate limiting, a ticker yfinance doesn't recognize, a
        # field that ticker doesn't report) -- the design intent is to
        # degrade gracefully rather than let one missing field abort the
        # whole build, but silently swallowing the exception would also
        # hide a genuine bug (e.g. a typo'd row-name candidate) as
        # indistinguishable from "field just isn't available." Warn so the
        # failure is visible even though the pipeline continues.
        try:
            return fn(*args)
        except Exception as e:
            warnings.warn(f"'{ticker}': {label} failed ({type(e).__name__}: {e}); leaving it unset.", stacklevel=2)
            return None

    sector = _try("fetch_sector", fetch_sector, ticker, use_cache, refresh) or "Unknown"
    sector = overrides.get("sector", sector)

    currency, financial_currency = _try("fetch_currency", fetch_currency, ticker, use_cache, refresh) or (None, None)
    currency = overrides.get("currency", currency)
    financial_currency = overrides.get("financial_currency", financial_currency)
    if currency is not None and financial_currency is not None and currency != financial_currency:
        warnings.warn(
            f"'{ticker}' quotes in {currency} but reports financials in {financial_currency}. "
            f"This toolkit does not convert currencies -- functions that combine market data "
            f"with financial-statement data for this company will raise until you resolve it "
            f"via manual_overrides (see Company.assert_single_currency).",
            stacklevel=2,
        )

    financials = _try("fetch_financials", fetch_financials, ticker, years, use_cache, refresh)
    if financials is None:
        financials = pd.DataFrame(columns=REQUIRED_FINANCIAL_COLUMNS)

    for year, fields in overrides.get("financials", {}).items():
        for col, val in fields.items():
            financials.loc[year, col] = val
    if not financials.empty:
        financials = financials.sort_index()

    price = overrides.get("share_price", _try("fetch_price", fetch_price, ticker, use_cache, refresh))
    shares = overrides.get(
        "shares_outstanding", _try("fetch_shares_outstanding", fetch_shares_outstanding, ticker, use_cache, refresh)
    )
    net_debt = overrides.get("net_debt", _try("fetch_net_debt", fetch_net_debt, ticker, use_cache, refresh))

    return Company(
        ticker=ticker.upper(),
        name=overrides.get("name", name or ticker.upper()),
        sector=sector,
        financials=financials,
        share_price=price,
        shares_outstanding=shares,
        net_debt=net_debt,
        currency=currency,
        financial_currency=financial_currency,
    )
