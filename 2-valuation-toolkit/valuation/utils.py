"""
utils.py
========

Shared financial-math helpers used by two or more modules in the toolkit
(dcf.py, comps.py, precedent_transactions.py, credit_overlay.py).

Kept dependency-light (numpy only) so these functions can be unit tested
in isolation without needing a Company object or network access.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Discounting
# ---------------------------------------------------------------------------


def discount_factor(rate: float, period: float) -> float:
    """
    Standard present-value discount factor: 1 / (1 + r)^t

    We assume end-of-period cash flow timing (not mid-year convention).
    Mid-year convention is a common refinement in real DCF models (it
    assumes cash flows arrive evenly through the year rather than in one
    lump at year-end, which slightly raises PV), but end-of-period keeps
    the mechanics transparent for a teaching/portfolio model.
    """
    return 1.0 / ((1.0 + rate) ** period)


def present_value(cash_flow: float, rate: float, period: float) -> float:
    """PV of a single future cash flow, discounted at `rate` for `period` years."""
    return cash_flow * discount_factor(rate, period)


def discount_cash_flows(cash_flows: Sequence[float], rate: float, start_period: int = 1) -> float:
    """
    Sum of PVs for a stream of cash flows.

    `start_period` defaults to 1 because the first projected year of a DCF
    is one full year out from valuation date (period 0 = today, undiscounted).
    """
    return sum(
        present_value(cf, rate, start_period + i) for i, cf in enumerate(cash_flows)
    )


# ---------------------------------------------------------------------------
# Cost of capital
# ---------------------------------------------------------------------------


def capm_cost_of_equity(risk_free_rate: float, beta: float, equity_risk_premium: float) -> float:
    """
    Capital Asset Pricing Model: Re = Rf + beta * ERP

    Rf is compensation for time value of money; beta * ERP is compensation
    for the equity's systematic (non-diversifiable) risk relative to the
    market. This is the standard way to derive a cost of equity when a
    company doesn't have directly observable required-return data.
    """
    return risk_free_rate + beta * equity_risk_premium


def after_tax_cost_of_debt(pre_tax_cost_of_debt: float, tax_rate: float) -> float:
    """
    Kd_after_tax = Kd_pre_tax * (1 - tax_rate)

    Interest is tax-deductible, so the "true" economic cost of debt to the
    firm is lower than the coupon/yield it pays -- this is the debt tax
    shield, and it's why WACC uses after-tax cost of debt.
    """
    return pre_tax_cost_of_debt * (1.0 - tax_rate)


def wacc(
    cost_of_equity: float,
    cost_of_debt_pre_tax: float,
    tax_rate: float,
    market_value_equity: float,
    market_value_debt: float,
) -> float:
    """
    Weighted Average Cost of Capital:

        WACC = E/(E+D) * Re  +  D/(E+D) * Kd * (1 - tax_rate)

    We discount *unlevered* free cash flow (cash available to all capital
    providers, before financing effects) at WACC because WACC blends the
    required returns of both equity and debt holders in proportion to how
    the firm is actually financed -- this is what makes UFCF discounted at
    WACC land on Enterprise Value (the value belonging to *all* capital
    providers), not just equity value.
    """
    total_capital = market_value_equity + market_value_debt
    if total_capital <= 0:
        raise ValueError("Combined market value of equity and debt must be positive.")

    equity_weight = market_value_equity / total_capital
    debt_weight = market_value_debt / total_capital
    kd_after_tax = after_tax_cost_of_debt(cost_of_debt_pre_tax, tax_rate)

    return equity_weight * cost_of_equity + debt_weight * kd_after_tax


# ---------------------------------------------------------------------------
# Growth / projection helpers
# ---------------------------------------------------------------------------


def project_with_growth(base_value: float, growth_rates: Sequence[float]) -> list[float]:
    """
    Roll a base value forward year by year using a (possibly year-varying)
    sequence of growth rates.

    A single flat growth rate can be passed as [g, g, g, g, g]; a fading
    growth trajectory (common in DCF models, since high growth rarely
    persists indefinitely) can be passed as e.g. [0.10, 0.08, 0.06, 0.04, 0.03].
    """
    values = []
    current = base_value
    for g in growth_rates:
        current = current * (1.0 + g)
        values.append(current)
    return values


def cagr(begin_value: float, end_value: float, periods: float) -> float:
    """
    Compound Annual Growth Rate: (end/begin)^(1/periods) - 1

    Used to summarize historical growth as a single smoothed rate, e.g. for
    sanity-checking a DCF's assumed forward growth against realized history.
    """
    if begin_value <= 0 or periods <= 0:
        raise ValueError("begin_value and periods must be positive.")
    return (end_value / begin_value) ** (1.0 / periods) - 1.0


# ---------------------------------------------------------------------------
# Terminal value
# ---------------------------------------------------------------------------


def terminal_value_gordon_growth(final_year_ufcf: float, wacc_rate: float, terminal_growth_rate: float) -> float:
    """
    Gordon Growth (perpetuity growth) terminal value, valued as of the end
    of the final forecast year:

        TV = UFCF_final * (1 + g) / (WACC - g)

    This treats the business as a perpetuity growing at a constant rate `g`
    forever after the explicit forecast window. `g` should be conservative
    (typically <= long-run GDP/inflation growth) -- WACC must exceed g or
    the formula produces a nonsensical (negative/infinite) value.
    """
    if wacc_rate <= terminal_growth_rate:
        raise ValueError("WACC must exceed the terminal growth rate for Gordon Growth to be valid.")
    return final_year_ufcf * (1.0 + terminal_growth_rate) / (wacc_rate - terminal_growth_rate)


def terminal_value_exit_multiple(final_year_ebitda: float, exit_multiple: float) -> float:
    """
    Exit Multiple terminal value, valued as of the end of the final forecast
    year:

        TV = EBITDA_final * Exit EV/EBITDA multiple

    This grounds the terminal value in what similar businesses actually
    trade for at exit, rather than an assumed perpetual growth rate --
    it's the market-based cross-check to Gordon Growth, and the two are
    conventionally shown side by side because they can imply very
    different (and informative) growth/multiple assumptions.
    """
    return final_year_ebitda * exit_multiple


# ---------------------------------------------------------------------------
# Enterprise Value -> Equity Value -> Share Price bridge
# ---------------------------------------------------------------------------


def ev_to_share_price(enterprise_value: float, net_debt: float, shares_outstanding: float) -> tuple[float, float]:
    """
    Bridge Enterprise Value to implied share price:

        Equity Value = Enterprise Value - Net Debt
        Share Price  = Equity Value / Shares Outstanding

    EV belongs to all capital providers (equity + debt); subtracting net
    debt strips out the portion owed to lenders, leaving what's left for
    shareholders. Used by dcf.py, comps.py, and precedent_transactions.py --
    every valuation method eventually produces an EV and bridges it the
    same way.
    """
    equity_value = enterprise_value - net_debt
    share_price = equity_value / shares_outstanding
    return equity_value, share_price


# ---------------------------------------------------------------------------
# Multiple / distribution statistics (for comps and precedent transactions)
# ---------------------------------------------------------------------------


def multiple_stats(values: Iterable[float]) -> dict[str, float]:
    """
    Summary statistics for a set of peer/transaction multiples: mean,
    median, and quartiles.

    We report median alongside mean because trading and deal multiples are
    frequently skewed by one or two outlier peers/transactions -- median is
    the more robust "central" multiple to apply to a target company.
    """
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("values must contain at least one element.")

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": float(arr.size),
    }
