"""
comps.py
========

Trading comparables: filter a candidate peer universe down to genuine
peers by sector + market-cap range, pull their trading multiples, and
apply the peer set's median/quartile multiples to the target company's
own financials to get an implied valuation range.

Also owns `plot_football_field`, the horizontal-bar range chart that later
overlays DCF, trading comps, and precedent transaction ranges together --
the money chart for the whole toolkit.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from . import utils
from .company import Company

# The three trading multiples this module pulls per peer. EV/EBITDA and
# EV/Revenue are capital-structure-neutral (numerator and denominator both
# sit above the debt line), which is why they're the standard multiples for
# comparing companies with different leverage -- P/E is included too since
# it's the one recruiters expect to see, despite being levered.
MULTIPLE_COLUMNS = ["ev_ebitda", "ev_revenue", "pe"]


# ---------------------------------------------------------------------------
# 1. Peer selection
# ---------------------------------------------------------------------------


def select_peers(
    candidate_tickers: Sequence[str],
    target_sector: str | None = None,
    market_cap_range: tuple[float, float] | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Filter a candidate ticker universe down to peers that actually match
    the target's sector and are in a comparable size range.

    This is deliberately explicit and adjustable rather than a hardcoded
    peer list: `candidate_tickers` is any longlist worth checking (e.g. the
    rest of an index/sector), and `target_sector` / `market_cap_range` are
    the filter criteria a real comps analysis would state up front and
    could tighten or loosen.

    Skips (rather than errors on) any candidate yfinance can't return
    sector/price/shares data for, and reports which were skipped.
    """
    from data.fetch import fetch_price, fetch_sector, fetch_shares_outstanding

    rows = []
    skipped = []
    for ticker in candidate_tickers:
        try:
            sector = fetch_sector(ticker, use_cache, refresh)
            price = fetch_price(ticker, use_cache, refresh)
            shares = fetch_shares_outstanding(ticker, use_cache, refresh)
        except Exception:
            skipped.append(ticker)
            continue
        rows.append({"ticker": ticker.upper(), "sector": sector, "market_cap": price * shares})

    if skipped:
        print(f"select_peers: skipped {len(skipped)} candidate(s) with no usable data: {skipped}")

    peers = pd.DataFrame(rows, columns=["ticker", "sector", "market_cap"])

    if target_sector is not None:
        peers = peers[peers["sector"] == target_sector]
    if market_cap_range is not None:
        lo, hi = market_cap_range
        peers = peers[(peers["market_cap"] >= lo) & (peers["market_cap"] <= hi)]

    return peers.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Peer multiples
# ---------------------------------------------------------------------------


def fetch_peer_multiples(peer_tickers: Sequence[str], use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    """
    Pull EV/EBITDA, EV/Revenue, and trailing P/E for each peer.

    EV/EBITDA and EV/Revenue are computed from each peer's own market-
    implied EV (`Company.enterprise_value_market`) against its latest
    historical EBITDA/revenue; P/E comes straight from yfinance.

    A peer missing any single multiple still contributes the multiples it
    does have (row holds NaN for the rest) -- callers should dropna per
    metric before computing summary stats, since dropping the whole peer
    would throw away otherwise-usable data.
    """
    from data.fetch import build_company, fetch_trailing_pe

    rows = []
    skipped = []
    for ticker in peer_tickers:
        try:
            company = build_company(ticker, years=1, use_cache=use_cache, refresh=refresh)
        except Exception as e:
            skipped.append((ticker, str(e)))
            continue

        try:
            # Also catches Company.assert_single_currency() inside
            # enterprise_value_market: a peer whose quote currency differs
            # from its financial-statement currency (e.g. a foreign ADR)
            # would otherwise produce an EV/EBITDA wrong by roughly the FX
            # rate -- excluded here rather than silently included.
            ev = company.enterprise_value_market
        except Exception as e:
            skipped.append((ticker, str(e)))
            ev = float("nan")

        row = {"ticker": ticker.upper()}
        try:
            row["ev_ebitda"] = ev / company.latest("ebitda")
        except Exception:
            row["ev_ebitda"] = float("nan")
        try:
            row["ev_revenue"] = ev / company.latest("revenue")
        except Exception:
            row["ev_revenue"] = float("nan")
        try:
            row["pe"] = fetch_trailing_pe(ticker, use_cache, refresh)
        except Exception:
            row["pe"] = float("nan")

        rows.append(row)

    if skipped:
        print(f"fetch_peer_multiples: {len(skipped)} peer(s) had an issue computing EV: {skipped}")

    return pd.DataFrame(rows, columns=["ticker"] + MULTIPLE_COLUMNS).set_index("ticker")


def summarize_peer_multiples(peer_multiples: pd.DataFrame) -> pd.DataFrame:
    """
    Mean/median/quartile stats per multiple, across the peer set.

    Each column is dropna'd independently before computing stats, so one
    peer's missing P/E (common for unprofitable/negative-earnings peers)
    doesn't remove that peer's still-valid EV/EBITDA and EV/Revenue from
    their respective stats.
    """
    stats = {}
    for col in peer_multiples.columns:
        clean = peer_multiples[col].dropna()
        if clean.empty:
            continue
        if len(clean) < 5:
            # Quartiles computed from fewer than ~5 points are dominated by
            # whichever single peer happens to sit at that percentile --
            # not wrong, just worth the reader knowing the range is noisy,
            # since a comps peer set realistically has 5-10 names, not 50.
            warnings.warn(
                f"summarize_peer_multiples: only {len(clean)} peer(s) have a usable '{col}' "
                f"multiple -- quartile stats from this small a sample are noisy.",
                stacklevel=2,
            )
        stats[col] = utils.multiple_stats(clean)
    return pd.DataFrame(stats)


# ---------------------------------------------------------------------------
# 3. Apply peer multiples to the target company
# ---------------------------------------------------------------------------


def comps_valuation(
    company: Company, peer_stats: pd.DataFrame, metrics: Sequence[str] = ("ev_ebitda", "ev_revenue")
) -> pd.DataFrame:
    """
    Apply the peer set's mean/median/quartile multiples to the target
    company's own latest financials, bridging each implied EV to equity
    value and implied share price.

    P/E is excluded from the default `metrics` because it needs the
    target's own net income/EPS (not part of `Company`'s financials) to
    apply directly to a share price -- pass metrics=(...,"pe") plus your
    own net income if you want it included.

    Returns a long-format DataFrame: one row per (metric, stat), with the
    peer multiple, implied EV, implied equity value, and implied share
    price -- e.g. useful directly, and this is also what feeds the
    football field chart's per-method low/high range.
    """
    # The bridge below combines the target's net_debt/shares_outstanding
    # (quote currency) with an implied EV built off the target's own
    # EBITDA/revenue (financial-statement currency) -- see
    # Company.assert_single_currency for why these must match.
    company.assert_single_currency()

    metric_to_financial = {"ev_ebitda": "ebitda", "ev_revenue": "revenue"}

    rows = []
    for metric in metrics:
        if metric not in metric_to_financial:
            raise ValueError(f"comps_valuation only supports metrics {list(metric_to_financial)}, got '{metric}'.")
        if metric not in peer_stats.columns:
            continue

        financial_value = company.latest(metric_to_financial[metric])
        for stat_name in ["q1", "median", "q3", "mean"]:
            multiple = peer_stats.loc[stat_name, metric]
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


def implied_share_price_range(
    valuation: pd.DataFrame, metric: str, low_stat: str = "q1", high_stat: str = "q3"
) -> tuple[float, float]:
    """
    Convenience accessor: pull the (low, high) implied-share-price range
    for one metric out of a `comps_valuation` output, e.g. for feeding
    `plot_football_field`. Defaults to the interquartile range (q1-q3),
    the conventional football-field bar width.
    """
    subset = valuation[valuation["metric"] == metric].set_index("stat")
    low = subset.loc[low_stat, "implied_share_price"]
    high = subset.loc[high_stat, "implied_share_price"]
    # Normally low_stat's multiple <= high_stat's multiple implies
    # low <= high automatically. The swap guards the one case where that
    # inverts: a target with NEGATIVE EBITDA/revenue, where a *larger*
    # multiple produces a *more negative* implied EV (and so a lower
    # implied share price) -- a real scenario for early-stage/unprofitable
    # targets, not a defensive no-op.
    return (low, high) if low <= high else (high, low)


# ---------------------------------------------------------------------------
# 4. Football field chart (shared across DCF / comps / precedent transactions)
# ---------------------------------------------------------------------------


def plot_football_field(
    ranges: dict[str, tuple[float, float]],
    midpoints: dict[str, float] | None = None,
    title: str = "Football Field: Implied Share Price",
    xlabel: str | None = None,
    currency_symbol: str = "$",
):
    """
    Horizontal bar range chart comparing implied valuation ranges across
    methods -- e.g. {"DCF (Gordon Growth)": (low, high),
    "Trading Comps (EV/EBITDA)": (low, high),
    "Precedent Transactions": (low, high)}.

    This is the standard "football field" used to present a valuation
    range across methodologies side by side; `midpoints`, if given, marks
    a point estimate (e.g. the peer median) inside each bar.

    `currency_symbol` defaults to "$" (USD) -- pass the target's actual
    currency symbol (e.g. "€") if it isn't USD-denominated; this
    toolkit does not track currency automatically in chart labels the way
    it enforces currency *consistency* in the underlying math (see
    Company.assert_single_currency).
    """
    if not ranges:
        raise ValueError("ranges must contain at least one method.")
    if xlabel is None:
        xlabel = f"Implied Share Price ({currency_symbol})"

    methods = list(ranges.keys())
    lows = [ranges[m][0] for m in methods]
    highs = [ranges[m][1] for m in methods]
    widths = [h - l for l, h in zip(lows, highs)]

    fig, ax = plt.subplots(figsize=(9, 0.8 * len(methods) + 1.5))
    y_pos = range(len(methods))

    ax.barh(y_pos, widths, left=lows, height=0.5, color="#4C72B0", alpha=0.85, edgecolor="black")

    # Wide bars get their value labels just inside each end (white, so they
    # sit legibly on the bar); bars too narrow for two inside labels to fit
    # without touching instead get labels just outside each end (black) --
    # outside-left never collides with the y-axis method labels because
    # it's still to the right of the plot's left spine once xlim padding
    # (added below) is applied.
    axis_span = max(highs) - min(lows) if len(methods) > 0 else 1.0
    pad = 0.015 * axis_span
    narrow_cutoff = 0.25 * axis_span
    for i, (low, high) in enumerate(zip(lows, highs)):
        if (high - low) >= narrow_cutoff:
            ax.text(low + pad, i, f"{currency_symbol}{low:,.0f}", va="center", ha="left", color="white", fontsize=9, fontweight="bold")
            ax.text(high - pad, i, f"{currency_symbol}{high:,.0f}", va="center", ha="right", color="white", fontsize=9, fontweight="bold")
        else:
            ax.text(low - pad, i, f"{currency_symbol}{low:,.0f}", va="center", ha="right", color="black", fontsize=9)
            ax.text(high + pad, i, f"{currency_symbol}{high:,.0f}", va="center", ha="left", color="black", fontsize=9)

    ax.set_xlim(min(lows) - 0.1 * axis_span, max(highs) + 0.1 * axis_span)

    if midpoints:
        for i, method in enumerate(methods):
            if method in midpoints:
                ax.plot(midpoints[method], i, marker="D", color="black", markersize=6, zorder=5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(methods)
    ax.invert_yaxis()  # first method listed appears at the top
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
