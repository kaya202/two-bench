"""Tests for the DCF engine: projection, WACC, full run_dcf, and sensitivity grid."""

import math

import pytest

from valuation import dcf


class TestProjectUfcf:
    def test_basic_projection(self, sample_company):
        assumptions = {
            "years": 3,
            "revenue_growth": 0.10,
            "ebitda_margin": 0.30,
            "da_pct_revenue": 0.05,
            "capex_pct_revenue": 0.10,
            "nwc_pct_revenue": 0.10,
            "tax_rate": 0.25,
        }
        proj = dcf.project_ufcf(sample_company, assumptions)
        assert list(proj.index) == [2025, 2026, 2027]  # latest historical year is 2024
        assert proj.loc[2025, "revenue"] == pytest.approx(1210 * 1.10)
        assert proj.loc[2026, "revenue"] == pytest.approx(1210 * 1.10**2)
        assert proj.loc[2025, "ebitda"] == pytest.approx(proj.loc[2025, "revenue"] * 0.30)

    def test_missing_required_assumption_raises(self, sample_company):
        with pytest.raises(ValueError):
            dcf.project_ufcf(sample_company, {"years": 3, "revenue_growth": 0.1})

    def test_wrong_length_trajectory_raises(self, sample_company):
        assumptions = {
            "years": 3,
            "revenue_growth": [0.1, 0.1],  # length 2, but years=3
            "ebitda_margin": 0.3,
            "da_pct_revenue": 0.05,
            "capex_pct_revenue": 0.1,
            "nwc_pct_revenue": 0.1,
        }
        with pytest.raises(ValueError):
            dcf.project_ufcf(sample_company, assumptions)

    def test_defaults_tax_rate_to_latest_historical(self, sample_company):
        assumptions = {
            "years": 1,
            "revenue_growth": 0.0,
            "ebitda_margin": 0.30,
            "da_pct_revenue": 0.05,
            "capex_pct_revenue": 0.1,
            "nwc_pct_revenue": 0.1,
        }
        proj = dcf.project_ufcf(sample_company, assumptions)
        assert proj.loc[2025, "tax_rate"] == pytest.approx(0.25)  # sample_company's latest historical rate


class TestComputeWacc:
    def test_manual_beta_no_network(self, sample_company):
        wb = dcf.compute_wacc(sample_company, risk_free_rate=0.04, equity_risk_premium=0.05, cost_of_debt_pre_tax=0.06, beta=1.0)
        assert wb["cost_of_equity"] == pytest.approx(0.09)
        assert 0 < wb["wacc"] < wb["cost_of_equity"]  # blended rate sits below pure cost of equity

    def test_currency_mismatch_raises(self, currency_mismatched_company):
        with pytest.raises(ValueError):
            dcf.compute_wacc(
                currency_mismatched_company, risk_free_rate=0.04, equity_risk_premium=0.05, cost_of_debt_pre_tax=0.06, beta=1.0
            )


class TestRunDcf:
    def test_end_to_end(self, sample_company):
        assumptions = {
            "years": 3,
            "revenue_growth": 0.08,
            "ebitda_margin": 0.28,
            "da_pct_revenue": 0.05,
            "capex_pct_revenue": 0.06,
            "nwc_pct_revenue": 0.10,
        }
        wacc_inputs = {"risk_free_rate": 0.04, "equity_risk_premium": 0.05, "cost_of_debt_pre_tax": 0.05, "beta": 1.0}
        result = dcf.run_dcf(sample_company, assumptions, wacc_inputs, terminal_growth_rate=0.02, exit_multiple=8.0)

        assert result.ev_gordon > 0
        assert result.ev_exit > 0
        assert result.implied_share_price_gordon == pytest.approx(result.equity_value_gordon / 100.0)
        assert result.implied_share_price_exit == pytest.approx(result.equity_value_exit / 100.0)

    def test_summary_table_has_both_methods(self, sample_company):
        assumptions = {
            "years": 2,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.25,
            "da_pct_revenue": 0.05,
            "capex_pct_revenue": 0.05,
            "nwc_pct_revenue": 0.1,
        }
        wacc_inputs = {"risk_free_rate": 0.04, "equity_risk_premium": 0.05, "cost_of_debt_pre_tax": 0.05, "beta": 1.0}
        result = dcf.run_dcf(sample_company, assumptions, wacc_inputs, terminal_growth_rate=0.02, exit_multiple=8.0)
        summary = result.summary()
        assert set(summary.columns) == {"Gordon Growth", "Exit Multiple"}
        assert "Implied Share Price" in summary.index


class TestSensitivityGrid:
    def test_invalid_gordon_cells_are_nan_not_a_crash(self, sample_company):
        assumptions = {
            "years": 2,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.25,
            "da_pct_revenue": 0.05,
            "capex_pct_revenue": 0.05,
            "nwc_pct_revenue": 0.1,
        }
        proj = dcf.project_ufcf(sample_company, assumptions)
        # WACC (3%) <= terminal growth (5%) is outside Gordon Growth's valid domain
        grid = dcf.sensitivity_grid(sample_company, proj, wacc_values=[0.03], terminal_values=[0.05], method="gordon")
        assert math.isnan(grid.iloc[0, 0])

    def test_valid_cells_produce_finite_values(self, sample_company):
        assumptions = {
            "years": 2,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.25,
            "da_pct_revenue": 0.05,
            "capex_pct_revenue": 0.05,
            "nwc_pct_revenue": 0.1,
        }
        proj = dcf.project_ufcf(sample_company, assumptions)
        grid = dcf.sensitivity_grid(sample_company, proj, wacc_values=[0.10], terminal_values=[0.02], method="gordon")
        assert math.isfinite(grid.iloc[0, 0])


def test_ev_to_share_price_wrapper(sample_company):
    equity, price = dcf.ev_to_share_price(1000.0, sample_company)
    assert equity == pytest.approx(1000.0 - sample_company.net_debt)
    assert price == pytest.approx(equity / sample_company.shares_outstanding)
