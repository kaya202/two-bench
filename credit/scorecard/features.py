"""
Ratio formulas and the feature set used by the scorecard.

The source dataset (data/corporate_rating.csv) already ships pre-computed
ratios rather than raw balance-sheet/income-statement line items — there are
no columns for total debt, total equity, current assets, etc. to derive them
from. So this module does two separate things:

1. Defines the standard ratio formulas as small, pure, testable functions
   (below), taking raw financial components as arguments. These document the
   formulas and are what test_features.py checks against known inputs. They
   are not run against this particular dataset, since the dataset's own
   pre-computed columns are used directly instead (see FEATURE_COLUMNS) —
   recomputing them from scratch would risk silently diverging from the
   vendor's own methodology (e.g. average vs. point-in-time denominators).
2. Selects a focused subset of the dataset's pre-computed ratio columns as
   the model's feature set, grouped into the categories a credit analyst
   actually thinks in, and standardizes them for logistic regression.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Ratio formulas (pure functions, not used on this dataset directly)
# ---------------------------------------------------------------------------


def debt_equity_ratio(total_debt: float, total_equity: float) -> float:
    """Total debt / total equity. Core leverage measure: how much debt
    funding sits behind each dollar of equity cushion."""
    return total_debt / total_equity


def debt_ratio(total_debt: float, total_assets: float) -> float:
    """Total debt / total assets. Leverage measure independent of equity
    accounting quirks (e.g. buybacks or write-downs distorting equity)."""
    return total_debt / total_assets


def return_on_assets(net_income: float, total_assets: float) -> float:
    """Net income / total assets. How efficiently the whole balance sheet
    generates profit, regardless of how it's financed."""
    return net_income / total_assets


def operating_margin(operating_income: float, revenue: float) -> float:
    """Operating income / revenue. Profitability from core operations,
    before financing costs and taxes — a cleaner read on the business
    itself than net margin."""
    return operating_income / revenue


def current_ratio(current_assets: float, current_liabilities: float) -> float:
    """Current assets / current liabilities. Can short-term obligations be
    covered by short-term assets — the standard first liquidity check."""
    return current_assets / current_liabilities


def quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> float:
    """(Current assets - inventory) / current liabilities. Stricter
    liquidity check that excludes inventory, which may not convert to cash
    quickly or at book value in a stress scenario."""
    return (current_assets - inventory) / current_liabilities


def cash_flow_coverage(operating_cash_flow: float, revenue: float) -> float:
    """Operating cash flow / revenue. Used here as a coverage *proxy*: this
    dataset has no EBIT/interest-expense breakout for a true interest
    coverage ratio, so this measures how much cash the business throws off
    per dollar of sales — a rough read on capacity to service debt from
    operations."""
    return operating_cash_flow / revenue


# ---------------------------------------------------------------------------
# 2. Feature set selected from the dataset's own pre-computed columns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Feature:
    column: str
    category: str
    rationale: str


FEATURES = [
    Feature(
        "debtEquityRatio",
        "Leverage",
        "How much debt sits behind each dollar of equity — the core solvency cushion lenders size up first.",
    ),
    Feature(
        "debtRatio",
        "Leverage",
        "Debt as a share of total assets — a second leverage cut that isn't distorted by equity accounting.",
    ),
    Feature(
        "returnOnAssets",
        "Profitability",
        "Profit generated per dollar of assets — a going-concern's ability to earn its way out of its obligations.",
    ),
    Feature(
        "operatingProfitMargin",
        "Profitability",
        "Core operating profitability before financing costs — signals whether the underlying business, not just its capital structure, is healthy.",
    ),
    Feature(
        "currentRatio",
        "Liquidity",
        "Short-term assets versus short-term liabilities — can the company meet obligations due within a year.",
    ),
    Feature(
        "quickRatio",
        "Liquidity",
        "Current ratio excluding inventory — liquidity under a stricter, more conservative read.",
    ),
    Feature(
        "operatingCashFlowSalesRatio",
        "Coverage (cash-flow proxy)",
        "No true interest coverage ratio is available in this dataset (no EBIT/interest expense columns); "
        "operating cash flow per dollar of sales is used as the closest available proxy for debt-servicing capacity.",
    ),
]

FEATURE_COLUMNS = [f.column for f in FEATURES]


def get_feature_table() -> pd.DataFrame:
    """The feature set as a DataFrame, for display in the notebook README-style."""
    return pd.DataFrame(
        {
            "feature": [f.column for f in FEATURES],
            "category": [f.category for f in FEATURES],
            "rationale": [f.rationale for f in FEATURES],
        }
    )


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model's feature columns from the loaded dataset."""
    return df[FEATURE_COLUMNS].copy()


def winsorize_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clip each ratio to its [1st, 99th] percentile bounds, fit on the
    training set only.

    Ratio-based features can spike toward +/-infinity when a denominator sits
    near zero (e.g. a company with near-zero total assets produces a
    returnOnAssets in the thousands, or near-zero equity produces a
    debt/equity ratio in the thousands) — a mechanical property of ratios,
    not necessarily a data error, but a handful of these rows are extreme
    enough to dominate StandardScaler's mean/std for the whole column and
    degrade the model for the remaining well-behaved companies. Winsorizing
    keeps every row (unlike dropping outliers, which would shrink an already
    small dataset) while preventing that domination. Bounds are computed on
    the training set only and applied to both, so no test-set information
    leaks into training — same discipline as scale_features below.
    """
    lower = X_train.quantile(lower_q)
    upper = X_train.quantile(upper_q)
    X_train_clipped = X_train.clip(lower=lower, upper=upper, axis=1)
    X_test_clipped = X_test.clip(lower=lower, upper=upper, axis=1)
    return X_train_clipped, X_test_clipped


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize features to zero mean / unit variance.

    Logistic regression's coefficients are only comparable to each other,
    and its L2-regularized optimizer only converges reasonably, when
    features share a common scale — otherwise a ratio like debtEquityRatio
    (range ~0-10) would dominate one like operatingProfitMargin (range
    ~-1 to 1) purely due to units, not credit signal. The scaler is fit on
    the training set only and applied to both, so no test-set information
    leaks into training.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler
