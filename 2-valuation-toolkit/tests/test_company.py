"""Tests for Company: market data, historical UFCF build, and the data-quality/currency guards."""

import math

import pytest

from valuation.company import Company


def test_market_cap(sample_company):
    assert sample_company.market_cap == pytest.approx(5000.0)


def test_enterprise_value_market(sample_company):
    assert sample_company.enterprise_value_market == pytest.approx(5200.0)


def test_latest_year(sample_company):
    assert sample_company.latest_year == 2024


def test_latest_ebitda_margin(sample_company):
    assert sample_company.latest_ebitda_margin == pytest.approx(310 / 1210)


def test_compute_historical_ufcf_first_year_delta_nwc_is_nan(sample_company):
    ufcf = sample_company.compute_historical_ufcf()
    assert math.isnan(ufcf.loc[2022, "ufcf"])


def test_compute_historical_ufcf_matches_hand_calc(sample_company):
    ufcf = sample_company.compute_historical_ufcf()
    # 2023: NOPAT = 225*0.75=168.75, +55 D&A, -65 capex, -delta_nwc(10) = 148.75
    assert ufcf.loc[2023, "ufcf"] == pytest.approx(148.75)
    # 2024: NOPAT = 250*0.75=187.5, +60 D&A, -70 capex, -delta_nwc(8) = 169.50
    assert ufcf.loc[2024, "ufcf"] == pytest.approx(169.50)


def test_missing_required_column_raises_at_construction():
    with pytest.raises(ValueError):
        Company(ticker="X", name="X", sector="X", financials={2024: {"revenue": 100}})


def test_latest_raises_clearly_on_missing_value():
    financials = {
        2024: {"revenue": 100.0, "ebitda": 20.0, "ebit": 15.0, "d_and_a": 5.0, "capex": 5.0, "nwc": 10.0, "tax_rate": float("nan")}
    }
    c = Company(ticker="X", name="X", sector="X", financials=financials)
    with pytest.raises(ValueError, match="tax_rate"):
        c.latest("tax_rate")


class TestCurrencyGuard:
    def test_mismatch_detected(self, currency_mismatched_company):
        assert currency_mismatched_company.currency_mismatch is True

    def test_assert_single_currency_raises_on_mismatch(self, currency_mismatched_company):
        with pytest.raises(ValueError, match="USD.*EUR|EUR.*USD"):
            currency_mismatched_company.assert_single_currency()

    def test_enterprise_value_market_raises_on_mismatch(self, currency_mismatched_company):
        with pytest.raises(ValueError):
            currency_mismatched_company.enterprise_value_market

    def test_no_mismatch_when_currencies_unset(self):
        c = Company(ticker="X", name="X", sector="X")
        assert c.currency_mismatch is False
        c.assert_single_currency()  # should not raise

    def test_no_mismatch_when_currencies_match(self, sample_company):
        assert sample_company.currency_mismatch is False
        sample_company.assert_single_currency()  # should not raise
