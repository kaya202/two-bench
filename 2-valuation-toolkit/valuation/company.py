"""
company.py
==========

`Company` is the shared spine of the toolkit: it holds identifying info,
historical financials, and market data for a single company, and knows how
to turn its own historical financials into unlevered free cash flow (UFCF).

Instances of `Company` get passed into dcf.py, comps.py,
precedent_transactions.py, and credit_overlay.py -- none of those modules
touch raw financial statements directly, they all read from this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Columns a historical financials table must carry for compute_historical_ufcf()
# to work. NWC is stored as a *level* (total net working capital at each
# period-end), not a change -- the change is computed internally, since you
# need at least two periods to know a delta.
REQUIRED_FINANCIAL_COLUMNS = [
    "revenue",
    "ebitda",
    "ebit",
    "d_and_a",
    "capex",
    "nwc",
    "tax_rate",
]


@dataclass
class Company:
    # --- Identifying info ---------------------------------------------------
    ticker: str
    name: str
    sector: str

    # --- Historical financials ----------------------------------------------
    # Indexed by fiscal year (e.g. int years 2021..2024), columns per
    # REQUIRED_FINANCIAL_COLUMNS. Populated either from data/fetch.py or by
    # passing a manually-built DataFrame/dict straight into the constructor.
    financials: pd.DataFrame = field(default_factory=pd.DataFrame)

    # --- Market data ----------------------------------------------------------
    share_price: float | None = None
    shares_outstanding: float | None = None
    net_debt: float | None = None

    # --- Currency metadata ----------------------------------------------------
    # `currency` is the currency share_price/market cap are quoted in;
    # `financial_currency` is the currency the financial statements (and so
    # net_debt) are reported in. For most US-listed companies these are both
    # USD and this is invisible. They differ for some foreign issuers traded
    # via a USD-quoted ADR while reporting statements in their home currency
    # (e.g. SAP: quoted in USD, reports in EUR) -- see `currency_mismatch`.
    currency: str | None = None
    financial_currency: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.financials, pd.DataFrame):
            # Allow callers to pass a plain dict-of-dicts, e.g.
            # {2023: {"revenue": 100, ...}, 2024: {"revenue": 110, ...}}
            self.financials = pd.DataFrame.from_dict(self.financials, orient="index")

        if not self.financials.empty:
            self.financials = self.financials.sort_index()
            missing = set(REQUIRED_FINANCIAL_COLUMNS) - set(self.financials.columns)
            if missing:
                raise ValueError(
                    f"Company '{self.ticker}' financials missing required columns: {sorted(missing)}"
                )

    # -------------------------------------------------------------------
    # Market data derived values
    # -------------------------------------------------------------------

    @property
    def market_cap(self) -> float:
        """Share price x shares outstanding -- current market value of equity."""
        if self.share_price is None or self.shares_outstanding is None:
            raise ValueError(f"'{self.ticker}': share_price and shares_outstanding required for market_cap.")
        return self.share_price * self.shares_outstanding

    @property
    def currency_mismatch(self) -> bool:
        """
        True when this company's quote currency and financial-statement
        currency are both known and differ (e.g. a USD-quoted ADR that
        reports financials in EUR). Only meaningful when both fields were
        actually populated -- a manually-built Company with neither field
        set returns False here, since there's nothing to compare.
        """
        return self.currency is not None and self.financial_currency is not None and self.currency != self.financial_currency

    def assert_single_currency(self) -> None:
        """
        Guard against silently combining market data (quote currency) with
        financial-statement data (financial currency) when they differ.

        This toolkit does NOT perform FX conversion. Discovered via testing
        against SAP (currency='USD' quote via its ADR, financialCurrency=
        'EUR' statements): naively computing EV/EBITDA, WACC weights, or a
        DCF-implied share price from such a company silently blends two
        currencies and produces a number that is wrong by roughly the FX
        rate, with no error and no visible sign anything is off. Call this
        at the top of any function that combines the two (see
        enterprise_value_market, dcf.compute_wacc, comps.comps_valuation,
        precedent_transactions.precedent_valuation).
        """
        if self.currency_mismatch:
            raise ValueError(
                f"'{self.ticker}' is quoted in {self.currency} but reports financials in "
                f"{self.financial_currency}. This toolkit does not convert currencies -- "
                f"combining market data (share price, market cap) with financial-statement "
                f"data (EBITDA, revenue, net debt) here would silently produce a number "
                f"wrong by roughly the FX rate. Resolve by supplying manual_overrides that "
                f"put share_price/shares_outstanding and financials/net_debt in the same "
                f"currency, then reconstruct the Company."
            )

    @property
    def enterprise_value_market(self) -> float:
        """
        Market-implied Enterprise Value = Market Cap + Net Debt.

        This is today's market-observed EV, distinct from the *modeled* EV
        that dcf.py / comps.py / precedent_transactions.py each produce --
        useful as a sanity-check anchor when reviewing model outputs.
        """
        self.assert_single_currency()
        if self.net_debt is None:
            raise ValueError(f"'{self.ticker}': net_debt required for enterprise_value_market.")
        return self.market_cap + self.net_debt

    # -------------------------------------------------------------------
    # Latest-year convenience accessors (used as DCF projection base year)
    # -------------------------------------------------------------------

    @property
    def latest_year(self) -> int:
        if self.financials.empty:
            raise ValueError(f"'{self.ticker}' has no historical financials loaded.")
        return int(self.financials.index[-1])

    def latest(self, column: str) -> float:
        """
        Most recent historical value for a given financials column.

        Raises rather than silently returning NaN: yfinance frequently can't
        supply every field for every ticker (see data/fetch.py), and letting
        a NaN quietly flow into a DCF/comps/credit calculation produces
        confusing all-NaN output several function calls away from the actual
        cause, instead of a clear error at the source.
        """
        value = float(self.financials.loc[self.latest_year, column])
        if pd.isna(value):
            raise ValueError(
                f"'{self.ticker}': '{column}' is missing (NaN) for FY{self.latest_year}. "
                f"Supply it via manual_overrides before using this company in a valuation."
            )
        return value

    @property
    def latest_ebitda_margin(self) -> float:
        return self.latest("ebitda") / self.latest("revenue")

    # -------------------------------------------------------------------
    # Historical UFCF build
    # -------------------------------------------------------------------

    def compute_historical_ufcf(self) -> pd.DataFrame:
        """
        Build unlevered free cash flow (UFCF) for every historical year,
        showing each step explicitly rather than one black-box formula:

            1. EBIT                        (operating profit, pre-interest/tax)
            2. NOPAT   = EBIT * (1 - tax_rate)   -- taxes as if the firm had no debt,
                                                     since UFCF is capital-structure-neutral
            3. + D&A                       -- add back: non-cash, was already
                                                subtracted to compute EBIT
            4. - Capex                     -- cash actually spent to maintain/grow
                                                the asset base, not an income-statement item
            5. - Delta NWC                 -- cash tied up funding growth in
                                                working capital (AR/inventory less AP)

            UFCF = NOPAT + D&A - Capex - Delta NWC

        Delta NWC for the first historical year is left as NaN (no prior
        period to difference against).
        """
        if self.financials.empty:
            raise ValueError(f"'{self.ticker}' has no historical financials loaded.")

        df = self.financials.copy()

        df["nopat"] = df["ebit"] * (1.0 - df["tax_rate"])
        df["delta_nwc"] = df["nwc"].diff()
        df["ufcf"] = df["nopat"] + df["d_and_a"] - df["capex"] - df["delta_nwc"]

        return df[["revenue", "ebit", "tax_rate", "nopat", "d_and_a", "capex", "delta_nwc", "ufcf"]]

    def __repr__(self) -> str:
        return f"Company(ticker={self.ticker!r}, name={self.name!r}, sector={self.sector!r})"
