"""Tests for the pure-math helpers: discounting, CAPM/WACC, terminal value, multiple stats."""

import pytest

from valuation import utils


class TestDiscounting:
    def test_discount_factor(self):
        assert utils.discount_factor(0.10, 1) == pytest.approx(1 / 1.10)
        assert utils.discount_factor(0.0, 5) == 1.0

    def test_present_value(self):
        assert utils.present_value(110, 0.10, 1) == pytest.approx(100.0)

    def test_discount_cash_flows_matches_manual_sum(self):
        pv = utils.discount_cash_flows([100, 100, 100], 0.10)
        expected = 100 / 1.1 + 100 / 1.1**2 + 100 / 1.1**3
        assert pv == pytest.approx(expected)


class TestCostOfCapital:
    def test_capm_cost_of_equity(self):
        assert utils.capm_cost_of_equity(0.04, 1.2, 0.05) == pytest.approx(0.10)

    def test_after_tax_cost_of_debt(self):
        assert utils.after_tax_cost_of_debt(0.08, 0.25) == pytest.approx(0.06)

    def test_wacc_matches_manual_calc(self):
        w = utils.wacc(cost_of_equity=0.10, cost_of_debt_pre_tax=0.08, tax_rate=0.25, market_value_equity=800, market_value_debt=200)
        expected = 0.8 * 0.10 + 0.2 * 0.08 * 0.75
        assert w == pytest.approx(expected)

    def test_wacc_zero_capital_raises(self):
        with pytest.raises(ValueError):
            utils.wacc(0.1, 0.05, 0.25, 0, 0)


class TestGrowthProjection:
    def test_project_with_growth(self):
        assert utils.project_with_growth(100, [0.10, 0.10]) == pytest.approx([110.0, 121.0])

    def test_cagr(self):
        assert utils.cagr(100, 121, 2) == pytest.approx(0.10)

    def test_cagr_nonpositive_begin_raises(self):
        with pytest.raises(ValueError):
            utils.cagr(-100, 121, 2)


class TestTerminalValue:
    def test_gordon_growth(self):
        tv = utils.terminal_value_gordon_growth(100, 0.10, 0.03)
        assert tv == pytest.approx(100 * 1.03 / 0.07)

    def test_gordon_growth_invalid_when_wacc_below_growth(self):
        with pytest.raises(ValueError):
            utils.terminal_value_gordon_growth(100, 0.02, 0.03)

    def test_gordon_growth_invalid_when_wacc_equals_growth(self):
        with pytest.raises(ValueError):
            utils.terminal_value_gordon_growth(100, 0.03, 0.03)

    def test_exit_multiple(self):
        assert utils.terminal_value_exit_multiple(100, 10) == 1000


class TestEvToSharePrice:
    def test_bridges_correctly(self):
        equity, price = utils.ev_to_share_price(1000, 200, 40)
        assert equity == 800
        assert price == 20


class TestMultipleStats:
    def test_basic_stats(self):
        stats = utils.multiple_stats([8, 9, 10, 11, 12])
        assert stats["median"] == 10
        assert stats["mean"] == 10
        assert stats["min"] == 8
        assert stats["max"] == 12
        assert stats["count"] == 5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            utils.multiple_stats([])
