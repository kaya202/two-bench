import pandas as pd
import pytest

from scorecard.features import (
    cash_flow_coverage,
    current_ratio,
    debt_equity_ratio,
    debt_ratio,
    operating_margin,
    quick_ratio,
    return_on_assets,
    winsorize_features,
)


def test_debt_equity_ratio():
    assert debt_equity_ratio(total_debt=50, total_equity=100) == 0.5
    assert debt_equity_ratio(total_debt=200, total_equity=100) == 2.0


def test_debt_ratio():
    assert debt_ratio(total_debt=30, total_assets=100) == 0.3


def test_return_on_assets():
    assert return_on_assets(net_income=10, total_assets=200) == 0.05


def test_operating_margin():
    assert operating_margin(operating_income=25, revenue=100) == 0.25
    assert operating_margin(operating_income=-10, revenue=100) == -0.1


def test_current_ratio():
    assert current_ratio(current_assets=200, current_liabilities=100) == 2.0


def test_quick_ratio():
    # Inventory is excluded from the numerator, unlike current_ratio.
    assert quick_ratio(current_assets=200, inventory=50, current_liabilities=100) == 1.5


def test_quick_ratio_never_exceeds_current_ratio():
    current_assets, inventory, current_liabilities = 200, 30, 100
    cr = current_ratio(current_assets, current_liabilities)
    qr = quick_ratio(current_assets, inventory, current_liabilities)
    assert qr <= cr


def test_cash_flow_coverage():
    assert cash_flow_coverage(operating_cash_flow=15, revenue=100) == 0.15


def test_ratio_functions_raise_on_zero_denominator():
    with pytest.raises(ZeroDivisionError):
        debt_equity_ratio(total_debt=50, total_equity=0)
    with pytest.raises(ZeroDivisionError):
        current_ratio(current_assets=100, current_liabilities=0)


def test_winsorize_features_clips_outliers_to_train_bounds():
    X_train = pd.DataFrame({"ratio": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
    X_test = pd.DataFrame({"ratio": [-500.0, 2.5, 9000.0]})

    train_clipped, test_clipped = winsorize_features(X_train, X_test, lower_q=0.10, upper_q=0.90)

    # The extreme value in the training set itself gets clipped down.
    assert train_clipped["ratio"].max() < 1000.0
    # Bounds are fit on train only, then applied to test — an in-range test
    # value passes through unchanged, and out-of-range ones get clipped to
    # the same bounds used for train.
    assert test_clipped["ratio"].iloc[1] == 2.5
    assert test_clipped["ratio"].min() == train_clipped["ratio"].min()
    assert test_clipped["ratio"].max() == train_clipped["ratio"].max()
