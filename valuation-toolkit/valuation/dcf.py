"""
dcf.py
======

Discounted Cash Flow engine: projects unlevered free cash flow off a set of
explicit, reusable assumptions, computes WACC, values the business under
both Gordon Growth and Exit Multiple terminal value methods, bridges
Enterprise Value to an implied share price, and runs a WACC x terminal-value
sensitivity grid.

Nothing in this module is company-specific -- every function takes a
`Company` plus a plain-dict assumption set, so the same code path runs for
any ticker `company.py` / `data/fetch.py` can populate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import utils
from .company import Company

DEFAULT_PROJECTION_YEARS = 5

# Assumption keys that vary by forecast year. Each may be passed either as a
# single float (applied flat across every year) or as a list of length
# `years` (an explicit trajectory, e.g. fading revenue growth).
_YEARLY_ASSUMPTION_KEYS = [
    "revenue_growth",
    "ebitda_margin",
    "da_pct_revenue",
    "capex_pct_revenue",
    "nwc_pct_revenue",
    "tax_rate",
]


def _broadcast(value: float | Sequence[float], years: int, key: str) -> list[float]:
    """Expand a scalar assumption to a flat trajectory, or validate an explicit one."""
    if isinstance(value, (int, float)):
        return [float(value)] * years
    values = list(value)
    if len(values) != years:
        raise ValueError(f"assumptions['{key}'] has {len(values)} entries, expected {years} (years).")
    return [float(v) for v in values]


# ---------------------------------------------------------------------------
# 1. UFCF projection
# ---------------------------------------------------------------------------


def project_ufcf(company: Company, assumptions: dict) -> pd.DataFrame:
    """
    Project unlevered free cash flow forward from the company's latest
    historical year, using the same explicit step-by-step build as
    `Company.compute_historical_ufcf`:

        Revenue (grown)
          -> EBITDA  = Revenue * EBITDA margin
          -> D&A     = Revenue * D&A % of revenue
          -> EBIT    = EBITDA - D&A
          -> NOPAT   = EBIT * (1 - tax rate)
          -> Capex   = Revenue * Capex % of revenue
          -> NWC     = Revenue * NWC % of revenue        (a level, not a delta)
          -> Delta NWC = NWC_t - NWC_(t-1)
          -> UFCF    = NOPAT + D&A - Capex - Delta NWC

    Driving everything off % of revenue (rather than projecting each line
    item independently) keeps the forecast internally consistent and is the
    standard convention for a driver-based DCF.

    `assumptions` keys (each scalar or a length-`years` list):
        years (int, default 5), revenue_growth, ebitda_margin,
        da_pct_revenue, capex_pct_revenue, nwc_pct_revenue,
        tax_rate (optional -- defaults to the company's latest historical rate)
    """
    years = int(assumptions.get("years", DEFAULT_PROJECTION_YEARS))
    if years < 1:
        raise ValueError("assumptions['years'] must be at least 1.")

    required = [k for k in _YEARLY_ASSUMPTION_KEYS if k != "tax_rate"]
    missing = [k for k in required if k not in assumptions]
    if missing:
        raise ValueError(f"assumptions missing required keys: {missing}")

    growth = _broadcast(assumptions["revenue_growth"], years, "revenue_growth")
    margins = _broadcast(assumptions["ebitda_margin"], years, "ebitda_margin")
    da_pct = _broadcast(assumptions["da_pct_revenue"], years, "da_pct_revenue")
    capex_pct = _broadcast(assumptions["capex_pct_revenue"], years, "capex_pct_revenue")
    nwc_pct = _broadcast(assumptions["nwc_pct_revenue"], years, "nwc_pct_revenue")
    tax_rate_default = assumptions.get("tax_rate", company.latest("tax_rate"))
    tax_rates = _broadcast(tax_rate_default, years, "tax_rate")

    base_revenue = company.latest("revenue")
    base_nwc = company.latest("nwc")

    revenues = utils.project_with_growth(base_revenue, growth)

    rows: dict[int, dict[str, float]] = {}
    prev_nwc = base_nwc
    for i in range(years):
        year = company.latest_year + i + 1
        revenue = revenues[i]
        ebitda = revenue * margins[i]
        d_and_a = revenue * da_pct[i]
        ebit = ebitda - d_and_a
        nopat = ebit * (1.0 - tax_rates[i])
        capex = revenue * capex_pct[i]
        nwc = revenue * nwc_pct[i]
        delta_nwc = nwc - prev_nwc
        ufcf = nopat + d_and_a - capex - delta_nwc

        rows[year] = {
            "revenue": revenue,
            "ebitda": ebitda,
            "d_and_a": d_and_a,
            "ebit": ebit,
            "tax_rate": tax_rates[i],
            "nopat": nopat,
            "capex": capex,
            "nwc": nwc,
            "delta_nwc": delta_nwc,
            "ufcf": ufcf,
        }
        prev_nwc = nwc

    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# 2. WACC
# ---------------------------------------------------------------------------


def compute_wacc(
    company: Company,
    risk_free_rate: float,
    equity_risk_premium: float,
    cost_of_debt_pre_tax: float,
    beta: float | None = None,
    tax_rate: float | None = None,
    market_value_debt: float | None = None,
) -> dict[str, float]:
    """
    WACC = E/(E+D) * Re + D/(E+D) * Kd * (1 - t), where:

      Re (cost of equity)  = CAPM = Rf + beta * ERP
      Kd (cost of debt)    = the company's actual pre-tax borrowing cost,
                              supplied here rather than derived, since a
                              reliable pre-tax yield isn't reconstructable
                              from the income-statement-level data this
                              toolkit pulls -- in practice an analyst reads
                              this off the company's bonds/credit spread.
      E                    = market value of equity = share price x shares out
      D                    = market value of debt; true market value is
                              rarely observable, so net debt (book value) is
                              used as the standard practical proxy.

    `beta`, if not supplied, is pulled from yfinance via data/fetch.py
    (manual override always takes precedence -- pass `beta` directly to
    skip the network call or to substitute an industry-unlevered-and-
    relevered beta).

    Returns every component (not just the final WACC) so the notebook can
    show the full build-up rather than a single opaque number.
    """
    company.assert_single_currency()  # WACC blends market_cap (quote currency) with net_debt (financial currency)

    if beta is None:
        from data.fetch import fetch_beta  # lazy import: avoids a hard yfinance dependency for manual-beta users

        beta = fetch_beta(company.ticker)

    if tax_rate is None:
        tax_rate = company.latest("tax_rate")

    if market_value_debt is None:
        market_value_debt = company.net_debt

    market_value_equity = company.market_cap

    cost_of_equity = utils.capm_cost_of_equity(risk_free_rate, beta, equity_risk_premium)
    cost_of_debt_after_tax = utils.after_tax_cost_of_debt(cost_of_debt_pre_tax, tax_rate)
    wacc_value = utils.wacc(cost_of_equity, cost_of_debt_pre_tax, tax_rate, market_value_equity, market_value_debt)

    total_capital = market_value_equity + market_value_debt
    return {
        "risk_free_rate": risk_free_rate,
        "beta": beta,
        "equity_risk_premium": equity_risk_premium,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt_pre_tax": cost_of_debt_pre_tax,
        "tax_rate": tax_rate,
        "cost_of_debt_after_tax": cost_of_debt_after_tax,
        "market_value_equity": market_value_equity,
        "market_value_debt": market_value_debt,
        "equity_weight": market_value_equity / total_capital,
        "debt_weight": market_value_debt / total_capital,
        "wacc": wacc_value,
    }


# ---------------------------------------------------------------------------
# 3 & 4. Terminal value + discounting -> EV -> equity value -> share price
# ---------------------------------------------------------------------------


def ev_to_share_price(enterprise_value: float, company: Company) -> tuple[float, float]:
    """Company-based convenience wrapper around `utils.ev_to_share_price`."""
    return utils.ev_to_share_price(enterprise_value, company.net_debt, company.shares_outstanding)


@dataclass
class DCFResult:
    projection: pd.DataFrame
    wacc_breakdown: dict[str, float]
    terminal_growth_rate: float
    exit_multiple: float

    pv_explicit_ufcf: float

    tv_gordon: float
    pv_tv_gordon: float
    ev_gordon: float
    equity_value_gordon: float
    implied_share_price_gordon: float

    tv_exit: float
    pv_tv_exit: float
    ev_exit: float
    equity_value_exit: float
    implied_share_price_exit: float

    def summary(self) -> pd.DataFrame:
        """Gordon Growth vs. Exit Multiple, side by side -- the headline DCF output."""
        return pd.DataFrame(
            {
                "Gordon Growth": {
                    "Terminal value (undiscounted)": self.tv_gordon,
                    "PV of terminal value": self.pv_tv_gordon,
                    "PV of explicit-period UFCF": self.pv_explicit_ufcf,
                    "Enterprise Value": self.ev_gordon,
                    "Equity Value": self.equity_value_gordon,
                    "Implied Share Price": self.implied_share_price_gordon,
                    "TV as % of EV": self.pv_tv_gordon / self.ev_gordon,
                },
                "Exit Multiple": {
                    "Terminal value (undiscounted)": self.tv_exit,
                    "PV of terminal value": self.pv_tv_exit,
                    "PV of explicit-period UFCF": self.pv_explicit_ufcf,
                    "Enterprise Value": self.ev_exit,
                    "Equity Value": self.equity_value_exit,
                    "Implied Share Price": self.implied_share_price_exit,
                    "TV as % of EV": self.pv_tv_exit / self.ev_exit,
                },
            }
        )


def run_dcf(
    company: Company,
    assumptions: dict,
    wacc_inputs: dict,
    terminal_growth_rate: float,
    exit_multiple: float,
) -> DCFResult:
    """
    End-to-end DCF: project UFCF, compute WACC, value the business under
    both terminal value methods, and bridge each to an implied share price.

    `wacc_inputs` is passed straight through to `compute_wacc` as kwargs,
    e.g. {"risk_free_rate": 0.04, "equity_risk_premium": 0.05,
          "cost_of_debt_pre_tax": 0.06, "beta": 1.1}
    """
    projection = project_ufcf(company, assumptions)
    wacc_breakdown = compute_wacc(company, **wacc_inputs)
    wacc_rate = wacc_breakdown["wacc"]
    n_years = len(projection)

    pv_explicit_ufcf = utils.discount_cash_flows(projection["ufcf"].tolist(), wacc_rate)

    final_ufcf = projection["ufcf"].iloc[-1]
    final_ebitda = projection["ebitda"].iloc[-1]

    tv_gordon = utils.terminal_value_gordon_growth(final_ufcf, wacc_rate, terminal_growth_rate)
    pv_tv_gordon = utils.present_value(tv_gordon, wacc_rate, n_years)
    ev_gordon = pv_explicit_ufcf + pv_tv_gordon
    equity_value_gordon, share_price_gordon = ev_to_share_price(ev_gordon, company)

    tv_exit = utils.terminal_value_exit_multiple(final_ebitda, exit_multiple)
    pv_tv_exit = utils.present_value(tv_exit, wacc_rate, n_years)
    ev_exit = pv_explicit_ufcf + pv_tv_exit
    equity_value_exit, share_price_exit = ev_to_share_price(ev_exit, company)

    return DCFResult(
        projection=projection,
        wacc_breakdown=wacc_breakdown,
        terminal_growth_rate=terminal_growth_rate,
        exit_multiple=exit_multiple,
        pv_explicit_ufcf=pv_explicit_ufcf,
        tv_gordon=tv_gordon,
        pv_tv_gordon=pv_tv_gordon,
        ev_gordon=ev_gordon,
        equity_value_gordon=equity_value_gordon,
        implied_share_price_gordon=share_price_gordon,
        tv_exit=tv_exit,
        pv_tv_exit=pv_tv_exit,
        ev_exit=ev_exit,
        equity_value_exit=equity_value_exit,
        implied_share_price_exit=share_price_exit,
    )


# ---------------------------------------------------------------------------
# 5. Sensitivity analysis
# ---------------------------------------------------------------------------


def sensitivity_grid(
    company: Company,
    projection: pd.DataFrame,
    wacc_values: Sequence[float],
    terminal_values: Sequence[float],
    method: str = "gordon",
    metric: str = "implied_share_price",
) -> pd.DataFrame:
    """
    2D sensitivity grid: rows = WACC, columns = terminal-value parameter
    (terminal growth rate for method="gordon", exit multiple for
    method="exit_multiple").

    The explicit-period UFCF projection doesn't depend on WACC or the
    terminal assumption, so it's computed once (passed in) and only the
    discounting/terminal-value step is re-run per grid cell.

    `metric`: "implied_share_price", "equity_value", or "enterprise_value".
    """
    if method not in ("gordon", "exit_multiple"):
        raise ValueError("method must be 'gordon' or 'exit_multiple'.")
    if metric not in ("implied_share_price", "equity_value", "enterprise_value"):
        raise ValueError("metric must be 'implied_share_price', 'equity_value', or 'enterprise_value'.")

    n_years = len(projection)
    final_ufcf = projection["ufcf"].iloc[-1]
    final_ebitda = projection["ebitda"].iloc[-1]

    grid = np.zeros((len(wacc_values), len(terminal_values)))

    for i, wacc_rate in enumerate(wacc_values):
        pv_explicit_ufcf = utils.discount_cash_flows(projection["ufcf"].tolist(), wacc_rate)
        for j, tparam in enumerate(terminal_values):
            try:
                if method == "gordon":
                    # Gordon Growth is undefined once WACC <= terminal growth rate (see
                    # utils.terminal_value_gordon_growth) -- a wide sensitivity range can
                    # legitimately include such a combination at its edge. Marking just
                    # that cell NaN (rather than letting the whole grid computation crash)
                    # is the useful behavior for a sensitivity table: the reader sees
                    # exactly which combinations are outside the model's valid domain.
                    tv = utils.terminal_value_gordon_growth(final_ufcf, wacc_rate, tparam)
                else:
                    tv = utils.terminal_value_exit_multiple(final_ebitda, tparam)
            except ValueError:
                grid[i, j] = np.nan
                continue

            pv_tv = utils.present_value(tv, wacc_rate, n_years)
            ev = pv_explicit_ufcf + pv_tv

            if metric == "enterprise_value":
                grid[i, j] = ev
            else:
                equity_value, share_price = ev_to_share_price(ev, company)
                grid[i, j] = share_price if metric == "implied_share_price" else equity_value

    row_label = "WACC"
    col_label = "Terminal Growth Rate" if method == "gordon" else "Exit Multiple"
    df = pd.DataFrame(
        grid,
        index=pd.Index([f"{w:.2%}" for w in wacc_values], name=row_label),
        columns=pd.Index(
            [f"{t:.2%}" for t in terminal_values] if method == "gordon" else [f"{t:.1f}x" for t in terminal_values],
            name=col_label,
        ),
    )
    return df


def plot_sensitivity_heatmap(grid: pd.DataFrame, title: str = "DCF Sensitivity", value_fmt: str = "${:,.2f}"):
    """
    Render a sensitivity grid (from `sensitivity_grid`) as an annotated
    matplotlib heatmap: WACC down the rows, terminal parameter across the
    columns, cell values annotated directly on the chart.
    """
    fig, ax = plt.subplots(figsize=(1.3 * len(grid.columns) + 2, 0.6 * len(grid.index) + 2))
    im = ax.imshow(grid.values, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel(grid.columns.name)
    ax.set_ylabel(grid.index.name)
    ax.set_title(title)

    # Annotate each cell; flip text color for readability against the
    # colormap's darker green/red extremes. Uses the cell's position
    # normalized to [0, 1] within the grid's actual value range, rather than
    # a fixed multiple of the midpoint -- the latter breaks (picks the wrong
    # color, or divides by ~0) whenever values span zero or a negative
    # range, which a wide sensitivity grid (e.g. implied equity value going
    # negative under a stress WACC/growth combination) can legitimately do.
    vmin, vmax = np.nanmin(grid.values), np.nanmax(grid.values)
    value_range = vmax - vmin
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.values[i, j]
            if np.isnan(value):
                ax.text(j, i, "N/A", ha="center", va="center", color="black", fontsize=9)
                continue
            norm_pos = (value - vmin) / value_range if value_range > 0 else 0.5
            color = "white" if norm_pos < 0.25 or norm_pos > 0.75 else "black"
            ax.text(j, i, value_fmt.format(value), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Value")
    fig.tight_layout()
    return fig
