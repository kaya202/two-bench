"""
credit_overlay.py
==================

THE differentiator module. Every other module in this toolkit answers
"what is this company worth?" This one takes that answer -- an Enterprise
Value from dcf.py, comps.py, or precedent_transactions.py, it doesn't
matter which -- and asks the question a direct lender / leveraged finance
analyst actually asks: "what can this company support in debt, and what do
I get back if it doesn't work out?"

Two pieces:
  1. Leverage capacity -- at a lender's assumed Debt/EBITDA underwriting
     level, how big is the debt quantum, is EBITDA coverage of interest
     adequate, and how much equity cushion sits below the debt in today's
     valuation.
  2. Downside / recovery -- stress EBITDA (and optionally the exit
     multiple) down, recompute EV, and see what each leverage tranche
     actually recovers -- i.e. cents on the dollar if the business
     underperforms. This mirrors the collateral-recovery logic in the
     companion bond default model (see README) applied here to a single
     company's own capital structure instead of a bond portfolio.

Simplification stated explicitly: both functions treat the assumed debt
quantum at each leverage level as a single tranche sitting senior to all
equity (no subordination waterfall across multiple debt tranches) --
realistic enough for a "how much can this company support, in aggregate"
lens, which is the question being asked here.

Currency note: this module trusts that `enterprise_value` and
`company.latest("ebitda")` are already in the same currency -- true as
long as that EV came from dcf.py/comps.py/precedent_transactions.py,
which all now guard against currency-mismatched companies at the point
EV is computed (see Company.assert_single_currency). This module doesn't
re-check, since by the time an EV reaches here that check has already
happened upstream.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .company import Company

DEFAULT_LEVERAGE_LEVELS = (3.0, 4.0, 5.0, 6.0)  # typical direct-lending / leveraged-finance underwriting range
DEFAULT_COVERAGE_THRESHOLDS = (2.0, 2.5)  # common minimum-interest-coverage covenant levels


def _broadcast(value: float | Sequence[float], n: int) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * n
    values = list(value)
    if len(values) != n:
        raise ValueError(f"expected {n} values, got {len(values)}.")
    return [float(v) for v in values]


# ---------------------------------------------------------------------------
# 1. Leverage capacity + interest coverage + equity cushion
# ---------------------------------------------------------------------------


def leverage_analysis(
    company: Company,
    enterprise_value: float,
    leverage_levels: Sequence[float] = DEFAULT_LEVERAGE_LEVELS,
    interest_rate: float | Sequence[float] = 0.09,
    coverage_thresholds: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS,
) -> pd.DataFrame:
    """
    For each assumed leverage level (turns of Debt/EBITDA a lender might
    underwrite to):

        Debt quantum        = leverage_x * EBITDA
        Interest expense     = Debt quantum * assumed interest rate
        Interest coverage    = EBITDA / Interest expense       (higher is safer)
        Equity cushion        = Enterprise Value - Debt quantum (equity value
                                 sitting below the debt, at today's EV --
                                 the buffer a lender is protected by before
                                 taking a loss)
        Equity cushion % of EV = Equity cushion / Enterprise Value

    `interest_rate` can be a single rate applied at every leverage level, or
    a list of length `len(leverage_levels)` -- in practice higher leverage
    tranches typically price at a wider spread, so a rising list (e.g.
    [0.08, 0.09, 0.10, 0.12]) is often more realistic than a flat rate.

    Flags a `below_{t}x_covenant` column per threshold in
    `coverage_thresholds`, marking leverage levels where interest coverage
    would fail a common minimum-coverage covenant.
    """
    ebitda = company.latest("ebitda")
    rates = _broadcast(interest_rate, len(leverage_levels))

    rows = []
    for level, rate in zip(leverage_levels, rates):
        debt_quantum = level * ebitda
        interest_expense = debt_quantum * rate
        interest_coverage = ebitda / interest_expense if interest_expense > 0 else float("inf")
        equity_cushion = enterprise_value - debt_quantum

        row = {
            "leverage_x": level,
            "ebitda": ebitda,
            "debt_quantum": debt_quantum,
            "interest_rate": rate,
            "interest_expense": interest_expense,
            "interest_coverage": interest_coverage,
            "enterprise_value": enterprise_value,
            "equity_cushion": equity_cushion,
            "equity_cushion_pct_of_ev": equity_cushion / enterprise_value,
        }
        for threshold in coverage_thresholds:
            row[f"below_{threshold}x_covenant"] = interest_coverage < threshold
        rows.append(row)

    return pd.DataFrame(rows).set_index("leverage_x")


# ---------------------------------------------------------------------------
# 2. Downside stress + recovery
# ---------------------------------------------------------------------------


def downside_recovery(
    company: Company,
    leverage_levels: Sequence[float] = DEFAULT_LEVERAGE_LEVELS,
    ebitda_decline: float = 0.25,
    exit_multiple: float | None = None,
    base_enterprise_value: float | None = None,
    multiple_compression: float = 0.0,
) -> pd.DataFrame:
    """
    Stress EBITDA down by `ebitda_decline` (e.g. 0.25 = a 25% EBITDA
    decline), optionally compress the valuation multiple too
    (`multiple_compression` -- distressed businesses don't just earn less,
    they're also typically valued at a lower multiple), and recompute EV
    under that stress case. Then, for each leverage tranche sized in the
    base case:

        Debt quantum   = leverage_x * BASE-CASE EBITDA  (the loan was sized
                          before the downside happened -- it doesn't shrink
                          just because EBITDA later falls)
        Stressed EV    = (Base EBITDA * (1 - decline)) * (Base multiple * (1 - compression))
        Recovery value = min(Stressed EV, Debt quantum)   -- debt is repaid
                          first out of enterprise value, up to the stressed
                          EV; it can't recover more than the business is
                          actually worth
        Recovery %     = Recovery value / Debt quantum     -- "cents on the
                          dollar" a lender gets back in the downside
        Equity wiped out = Stressed EV < Debt quantum       -- equity holders
                          get nothing once EV falls below the debt claim

    This is the same recovery logic (claim vs. available collateral value)
    used in the companion bond default model, applied here to one
    company's own capital structure under a single stress scenario rather
    than a portfolio of bonds.

    Either supply `exit_multiple` directly, or supply `base_enterprise_value`
    and the base-case EV/EBITDA multiple is implied from it
    (`base_enterprise_value / base EBITDA`).
    """
    base_ebitda = company.latest("ebitda")

    if exit_multiple is None:
        if base_enterprise_value is None:
            raise ValueError("Provide either exit_multiple or base_enterprise_value.")
        exit_multiple = base_enterprise_value / base_ebitda

    stressed_ebitda = base_ebitda * (1.0 - ebitda_decline)
    stressed_multiple = exit_multiple * (1.0 - multiple_compression)
    stressed_ev = stressed_ebitda * stressed_multiple

    rows = []
    for level in leverage_levels:
        debt_quantum = level * base_ebitda
        recovery_value = min(stressed_ev, debt_quantum)
        recovery_pct = recovery_value / debt_quantum if debt_quantum > 0 else float("nan")
        equity_value_in_stress = max(stressed_ev - debt_quantum, 0.0)

        rows.append(
            {
                "leverage_x": level,
                "debt_quantum": debt_quantum,
                "base_ebitda": base_ebitda,
                "stressed_ebitda": stressed_ebitda,
                "exit_multiple": exit_multiple,
                "stressed_multiple": stressed_multiple,
                "stressed_ev": stressed_ev,
                "recovery_value": recovery_value,
                "recovery_pct": recovery_pct,
                "equity_value_in_stress": equity_value_in_stress,
                "equity_wiped_out": stressed_ev < debt_quantum,
            }
        )

    return pd.DataFrame(rows).set_index("leverage_x")


# ---------------------------------------------------------------------------
# 3. Combined summary + chart
# ---------------------------------------------------------------------------


def credit_summary_table(leverage_df: pd.DataFrame, downside_df: pd.DataFrame) -> pd.DataFrame:
    """
    The headline credit-overlay output: at each leverage tranche, base-case
    equity cushion and downside recovery side by side -- "at X turns of
    leverage, here's your equity cushion and here's recovery in a
    downside," in one table.
    """
    covenant_cols = [c for c in leverage_df.columns if c.startswith("below_") and c.endswith("_covenant")]
    return pd.DataFrame(
        {
            "debt_quantum": leverage_df["debt_quantum"],
            "interest_coverage": leverage_df["interest_coverage"],
            "equity_cushion_pct_of_ev": leverage_df["equity_cushion_pct_of_ev"],
            "recovery_pct": downside_df["recovery_pct"],
            "equity_wiped_out": downside_df["equity_wiped_out"],
            **{col: leverage_df[col] for col in covenant_cols},
        }
    )


def plot_leverage_recovery(
    summary: pd.DataFrame, title: str = "Leverage Capacity & Downside Recovery"
):
    """
    Grouped bar chart: base-case equity cushion (% of EV) vs. downside
    recovery (% of debt principal) at each leverage tranche -- the visual
    version of "here's your cushion, here's your downside" that a direct
    lending analyst would actually build into a credit memo. A dashed line
    at 100% marks par recovery; leverage levels that breach the interest
    coverage covenant are marked with an asterisk on the x-axis label.
    """
    leverage_labels = [f"{x:.1f}x" for x in summary.index]
    cushion_pct = (summary["equity_cushion_pct_of_ev"] * 100).values
    recovery_pct = (summary["recovery_pct"] * 100).values

    x = np.arange(len(leverage_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, cushion_pct, width, label="Equity cushion (% of EV, base case)", color="#4C72B0")
    ax.bar(x + width / 2, recovery_pct, width, label="Recovery (% of debt principal, downside)", color="#C44E52")

    ax.axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(-0.5, 101.5, "Par (100%)", fontsize=8, ha="left", color="black")
    ax.set_ylim(0, max(105, cushion_pct.max(), recovery_pct.max()) + 8)

    covenant_cols = [c for c in summary.columns if c.startswith("below_") and c.endswith("_covenant")]
    breach_any = summary[covenant_cols].any(axis=1) if covenant_cols else pd.Series(False, index=summary.index)
    xtick_labels = [lbl + (" *" if breach else "") for lbl, breach in zip(leverage_labels, breach_any)]

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Leverage (Debt / EBITDA)" + ("   (* = breaches an interest coverage covenant)" if breach_any.any() else ""))
    ax.set_ylabel("%")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
