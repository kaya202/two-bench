"""
precedent_transactions.py
==========================

Precedent M&A transaction analysis: apply the median/quartile EV/EBITDA
multiples paid in real, sourced historical deals to the target company's
own EBITDA, the same median-multiple-to-target logic as comps.py.

Live precedent-transaction data isn't freely available via API (it lives
behind terminals like Bloomberg/Capital IQ/PitchBook), so this module reads
from a small curated CSV (data/precedent_transactions.csv) of real,
individually-sourced enterprise-software deals -- see that file for each
transaction's announcement date, disclosed multiple, and source citation.

The other half of this module is the control premium: precedent deal
multiples are paid to acquire *control* of a business (the right to direct
its strategy, cash flows, and capture synergies), which public-market
trading multiples don't reflect -- that's the standard reason precedent
transaction multiples run above trading comps multiples for the same
sector, and it's made explicit here rather than left implicit.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from . import utils
from .comps import implied_share_price_range  # noqa: F401  (re-exported for convenience)
from .company import Company

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "precedent_transactions.csv"


# ---------------------------------------------------------------------------
# 1. Load the curated deal dataset
# ---------------------------------------------------------------------------


def load_precedent_transactions(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the curated precedent-transaction dataset. Each row is one real,
    sourced deal (see data/precedent_transactions.csv's `source_url` and
    `notes` columns) -- this function does no filtering; use
    `filter_transactions` to narrow to a relevant sector/date range.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    df = pd.read_csv(path, parse_dates=["announcement_date"])
    return df


def filter_transactions(
    transactions: pd.DataFrame,
    sector: str | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
) -> pd.DataFrame:
    """
    Narrow the curated dataset to a relevant sector and/or date window --
    explicit, adjustable criteria, same principle as `comps.select_peers`.

    Important: `sector` here matches this CSV's own hand-curated deal-type
    labels (e.g. "Enterprise Infrastructure Software", "Cybersecurity
    Software" -- see the `sector` column in data/precedent_transactions.csv),
    NOT yfinance's coarser GICS `sector` field (e.g. "Technology") that
    `comps.select_peers` filters on. Passing a target's
    `data.fetch.fetch_sector(...)` result here will silently match zero
    rows, since "Technology" never equals any label in this CSV -- inspect
    `transactions["sector"].unique()` for the values this dataset actually
    uses.
    """
    df = transactions
    if sector is not None:
        df = df[df["sector"] == sector]
    if min_date is not None:
        df = df[df["announcement_date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["announcement_date"] <= pd.Timestamp(max_date)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Summary stats + application to the target company
# ---------------------------------------------------------------------------


def summarize_transaction_multiples(transactions: pd.DataFrame, metrics: tuple[str, ...] = ("ev_ebitda",)) -> pd.DataFrame:
    """
    Mean/median/quartile stats per deal multiple across the transaction set.

    Defaults to EV/EBITDA only: the curated dataset has EV/Revenue for just
    one deal (CA Technologies), too small a sample to summarize -- pass
    metrics=("ev_ebitda", "ev_revenue") once/if more revenue-multiple deals
    are added to the CSV.
    """
    stats = {}
    for metric in metrics:
        clean = transactions[metric].dropna()
        if clean.empty:
            continue
        if len(clean) < 5:
            # The curated dataset has 6 deals total (see
            # data/precedent_transactions.csv); quartiles from this few
            # data points are illustrative, not statistically robust --
            # each individual deal materially moves the quartile.
            warnings.warn(
                f"summarize_transaction_multiples: only {len(clean)} deal(s) have a usable "
                f"'{metric}' multiple -- quartile stats from this small a sample are noisy.",
                stacklevel=2,
            )
        stats[metric] = utils.multiple_stats(clean)
    return pd.DataFrame(stats)


def precedent_valuation(company: Company, transaction_stats: pd.DataFrame, metrics: tuple[str, ...] = ("ev_ebitda",)) -> pd.DataFrame:
    """
    Apply the precedent-transaction set's mean/median/quartile multiples to
    the target company's own latest financials -- structurally identical
    output to `comps.comps_valuation` (same columns), so the same
    `comps.implied_share_price_range` helper and football-field chart work
    on either one.
    """
    # Same currency-consistency requirement as comps.comps_valuation -- see
    # Company.assert_single_currency.
    company.assert_single_currency()

    metric_to_financial = {"ev_ebitda": "ebitda", "ev_revenue": "revenue"}

    rows = []
    for metric in metrics:
        if metric not in metric_to_financial:
            raise ValueError(f"precedent_valuation only supports metrics {list(metric_to_financial)}, got '{metric}'.")
        if metric not in transaction_stats.columns:
            continue

        financial_value = company.latest(metric_to_financial[metric])
        for stat_name in ["q1", "median", "q3", "mean"]:
            multiple = transaction_stats.loc[stat_name, metric]
            implied_ev = multiple * financial_value
            implied_equity_value, implied_share_price = utils.ev_to_share_price(
                implied_ev, company.net_debt, company.shares_outstanding
            )
            rows.append(
                {
                    "metric": metric,
                    "stat": stat_name,
                    "multiple": multiple,
                    "implied_ev": implied_ev,
                    "implied_equity_value": implied_equity_value,
                    "implied_share_price": implied_share_price,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Control premium
# ---------------------------------------------------------------------------


def control_premium(
    transaction_stats: pd.DataFrame,
    trading_comps_stats: pd.DataFrame,
    metric: str = "ev_ebitda",
    stat: str = "median",
) -> float:
    """
    The control premium embedded in precedent deal multiples, relative to
    where comparable public companies trade today:

        premium = precedent transaction multiple / trading comps multiple - 1

    Acquirers pay this premium for control: the ability to direct the
    target's strategy and cash flows, and to capture synergies that a
    passive public-market minority shareholder can't access. It's the
    standard, textbook reason precedent transaction multiples run above
    trading comps multiples for the same sector -- this function makes
    that gap an explicit, computed number rather than leaving it implicit
    in two separately-eyeballed tables.

    Caveat: this only comes out positive when the two multiple sets are
    reasonably matched in vintage and growth profile. The curated deal set
    in data/precedent_transactions.csv spans 2014-2022 legacy enterprise
    infrastructure software transactions; if `trading_comps_stats` comes
    from a *current* peer set that includes richly-rated, higher-growth
    names, the "premium" can come out negative -- that's the peer/vintage
    mismatch showing up, not a sign the control premium concept is wrong.
    Read a negative result as a flag to reselect a more comparable peer or
    deal set, not as "no control premium exists."
    """
    precedent_multiple = transaction_stats.loc[stat, metric]
    trading_multiple = trading_comps_stats.loc[stat, metric]
    return precedent_multiple / trading_multiple - 1.0


def apply_control_premium(value_or_multiple: float, premium: float) -> float:
    """
    Gross a trading-comps-derived multiple or value up by a control
    premium -- e.g. to sanity-check "what would a full takeout of this
    company likely cost, given where it trades today and the premium
    control buyers have historically paid in this sector."

    `premium` as a decimal (0.42 = 42%), typically the output of
    `control_premium()`.
    """
    return value_or_multiple * (1.0 + premium)
