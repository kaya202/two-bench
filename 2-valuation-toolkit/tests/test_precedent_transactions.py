"""Tests for precedent transaction analysis: the curated CSV, applying multiples, control premium."""

import pandas as pd
import pytest

from valuation import precedent_transactions as pt


class TestLoadPrecedentTransactions:
    def test_loads_real_curated_csv(self):
        txns = pt.load_precedent_transactions()
        assert len(txns) == 8
        assert {"announcement_date", "acquirer", "target", "ev_ebitda", "source_url"}.issubset(txns.columns)
        assert txns["ev_ebitda"].notna().all()

    def test_filter_by_cruise_sector(self):
        txns = pt.load_precedent_transactions()
        filtered = pt.filter_transactions(txns, sector="Cruise & Leisure Travel")
        assert len(filtered) == 2

    def test_filter_by_this_datasets_own_sector_label(self):
        txns = pt.load_precedent_transactions()
        filtered = pt.filter_transactions(txns, sector="Cybersecurity Software")
        assert len(filtered) == 1
        assert filtered.iloc[0]["target"].startswith("Symantec")

    def test_filter_by_date_range(self):
        txns = pt.load_precedent_transactions()
        filtered = pt.filter_transactions(txns, min_date="2020-01-01")
        assert len(filtered) == 2  # VMware (2022) and Citrix (2022) are the only post-2020 deals


class TestPrecedentValuation:
    def test_matches_hand_calc(self, sample_company):
        txn_stats = pd.DataFrame({"ev_ebitda": {"q1": 10.0, "median": 12.0, "q3": 14.0, "mean": 12.0}})
        valuation = pt.precedent_valuation(sample_company, txn_stats)
        row = valuation[valuation["stat"] == "median"].iloc[0]
        assert row["implied_ev"] == pytest.approx(310 * 12)  # sample_company's 2024 EBITDA is 310

    def test_currency_mismatch_raises(self, currency_mismatched_company):
        txn_stats = pd.DataFrame({"ev_ebitda": {"q1": 10.0, "median": 12.0, "q3": 14.0, "mean": 12.0}})
        with pytest.raises(ValueError):
            pt.precedent_valuation(currency_mismatched_company, txn_stats)


class TestControlPremium:
    def test_positive_premium(self):
        txn_stats = pd.DataFrame({"ev_ebitda": {"median": 15.0}})
        trading_stats = pd.DataFrame({"ev_ebitda": {"median": 10.0}})
        assert pt.control_premium(txn_stats, trading_stats) == pytest.approx(0.5)

    def test_negative_premium(self):
        # The real case observed running this toolkit end to end (see LIMITATIONS.md):
        # a vintage/growth-profile mismatch between the two inputs can flip the sign.
        txn_stats = pd.DataFrame({"ev_ebitda": {"median": 8.0}})
        trading_stats = pd.DataFrame({"ev_ebitda": {"median": 10.0}})
        assert pt.control_premium(txn_stats, trading_stats) == pytest.approx(-0.2)

    def test_apply_control_premium(self):
        assert pt.apply_control_premium(10.0, 0.2) == pytest.approx(12.0)
        assert pt.apply_control_premium(10.0, -0.1) == pytest.approx(9.0)
