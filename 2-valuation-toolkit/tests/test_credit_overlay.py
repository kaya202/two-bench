"""Tests for the credit overlay: leverage capacity, interest coverage, downside recovery."""

import pytest

from valuation import credit_overlay as co


class TestLeverageAnalysis:
    def test_matches_hand_calc(self, sample_company):
        lev = co.leverage_analysis(sample_company, enterprise_value=5000.0, leverage_levels=(3.0, 5.0), interest_rate=0.10)
        # sample_company's 2024 EBITDA is 310
        assert lev.loc[3.0, "debt_quantum"] == pytest.approx(930.0)
        assert lev.loc[3.0, "interest_expense"] == pytest.approx(93.0)
        assert lev.loc[3.0, "interest_coverage"] == pytest.approx(310 / 93.0)
        assert lev.loc[3.0, "equity_cushion"] == pytest.approx(5000.0 - 930.0)

    def test_covenant_breach_flagged(self, sample_company):
        # debt=6*310=1860, interest=1860*0.15=279, coverage=310/279=1.11 < 2.0
        lev = co.leverage_analysis(
            sample_company, enterprise_value=5000.0, leverage_levels=(6.0,), interest_rate=0.15, coverage_thresholds=(2.0,)
        )
        assert bool(lev.loc[6.0, "below_2.0x_covenant"]) is True

    def test_no_breach_when_coverage_adequate(self, sample_company):
        lev = co.leverage_analysis(
            sample_company, enterprise_value=5000.0, leverage_levels=(2.0,), interest_rate=0.05, coverage_thresholds=(2.0,)
        )
        assert bool(lev.loc[2.0, "below_2.0x_covenant"]) is False

    def test_mismatched_rate_list_length_raises(self, sample_company):
        with pytest.raises(ValueError):
            co.leverage_analysis(sample_company, 5000.0, leverage_levels=(3.0, 4.0), interest_rate=[0.1])


class TestDownsideRecovery:
    def test_full_recovery_when_ev_comfortably_exceeds_debt(self, sample_company):
        down = co.downside_recovery(sample_company, leverage_levels=(1.0,), ebitda_decline=0.1, exit_multiple=20.0)
        assert down.loc[1.0, "recovery_pct"] == pytest.approx(1.0)
        assert bool(down.loc[1.0, "equity_wiped_out"]) is False

    def test_equity_wiped_out_under_severe_stress(self, sample_company):
        # debt=20*310=6200; stressed EBITDA=310*0.5=155; stressed EV=155*2=310 << debt
        down = co.downside_recovery(sample_company, leverage_levels=(20.0,), ebitda_decline=0.5, exit_multiple=2.0)
        assert down.loc[20.0, "recovery_pct"] == pytest.approx(310.0 / 6200.0)
        assert bool(down.loc[20.0, "equity_wiped_out"]) is True

    def test_recovery_never_exceeds_par(self, sample_company):
        # deliberately generous stress case -- recovery should cap at 1.0, never exceed it
        down = co.downside_recovery(sample_company, leverage_levels=(1.0, 2.0), ebitda_decline=0.0, exit_multiple=50.0)
        assert (down["recovery_pct"] <= 1.0).all()

    def test_requires_multiple_or_base_ev(self, sample_company):
        with pytest.raises(ValueError):
            co.downside_recovery(sample_company, leverage_levels=(3.0,))

    def test_multiple_implied_from_base_enterprise_value(self, sample_company):
        # base EBITDA=310, base_enterprise_value=3100 -> implied exit_multiple=10.0
        down = co.downside_recovery(sample_company, leverage_levels=(1.0,), ebitda_decline=0.0, base_enterprise_value=3100.0)
        assert down.loc[1.0, "exit_multiple"] == pytest.approx(10.0)


class TestCreditSummaryTable:
    def test_merges_covenant_columns(self, sample_company):
        lev = co.leverage_analysis(sample_company, 5000.0, leverage_levels=(3.0, 6.0), interest_rate=0.12, coverage_thresholds=(2.0, 2.5))
        down = co.downside_recovery(sample_company, leverage_levels=(3.0, 6.0), ebitda_decline=0.3, exit_multiple=10.0)
        summary = co.credit_summary_table(lev, down)
        # this exact bug (covenant flags silently dropped before reaching the chart) was
        # caught during development -- see LIMITATIONS.md / the credit_overlay module notes
        assert "below_2.0x_covenant" in summary.columns
        assert "below_2.5x_covenant" in summary.columns


def test_plot_leverage_recovery_renders(sample_company):
    import matplotlib

    matplotlib.use("Agg")
    lev = co.leverage_analysis(sample_company, 5000.0, leverage_levels=(3.0, 6.0), interest_rate=0.12, coverage_thresholds=(2.0, 2.5))
    down = co.downside_recovery(sample_company, leverage_levels=(3.0, 6.0), ebitda_decline=0.3, exit_multiple=10.0)
    summary = co.credit_summary_table(lev, down)
    fig = co.plot_leverage_recovery(summary)
    assert fig is not None
