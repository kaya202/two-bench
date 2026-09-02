"""Tests for trading comparables: multiple stats, applying multiples to a target, football field."""

import pandas as pd
import pytest

from valuation import comps


class TestSummarizePeerMultiples:
    def test_basic_stats_and_independent_dropna(self):
        peer_multiples = pd.DataFrame(
            {
                "ev_ebitda": [8.0, 9.0, 10.0, 11.0, 12.0],
                "ev_revenue": [2.0, 2.5, 3.0, 3.5, 4.0],
                "pe": [15.0, float("nan"), 17.0, 18.0, 19.0],
            },
            index=["A", "B", "C", "D", "E"],
        )
        stats = comps.summarize_peer_multiples(peer_multiples)
        assert stats.loc["median", "ev_ebitda"] == 10.0
        # one peer's missing P/E doesn't remove it from the other columns' stats
        assert stats.loc["count", "pe"] == 4
        assert stats.loc["count", "ev_ebitda"] == 5


class TestCompsValuation:
    def test_matches_hand_calc(self, sample_company):
        peer_stats = pd.DataFrame(
            {
                "ev_ebitda": {"q1": 10.0, "median": 12.0, "q3": 14.0, "mean": 12.0},
                "ev_revenue": {"q1": 2.0, "median": 2.5, "q3": 3.0, "mean": 2.5},
            }
        )
        valuation = comps.comps_valuation(sample_company, peer_stats)
        row = valuation[(valuation["metric"] == "ev_ebitda") & (valuation["stat"] == "median")].iloc[0]
        # ebitda(2024)=310, multiple=12 -> EV=3720; equity=3720-200=3520; price=3520/100=35.2
        assert row["implied_ev"] == pytest.approx(3720.0)
        assert row["implied_share_price"] == pytest.approx(35.2)

    def test_currency_mismatch_raises(self, currency_mismatched_company):
        peer_stats = pd.DataFrame({"ev_ebitda": {"q1": 8.0, "median": 10.0, "q3": 12.0, "mean": 10.0}})
        with pytest.raises(ValueError):
            comps.comps_valuation(currency_mismatched_company, peer_stats)


class TestImpliedSharePriceRange:
    def test_normal_ordering(self):
        valuation = pd.DataFrame(
            [
                {"metric": "ev_ebitda", "stat": "q1", "implied_share_price": 10.0},
                {"metric": "ev_ebitda", "stat": "q3", "implied_share_price": 20.0},
            ]
        )
        assert comps.implied_share_price_range(valuation, "ev_ebitda") == (10.0, 20.0)

    def test_negative_ebitda_case_gets_swapped(self):
        # A larger multiple on a NEGATIVE ebitda base implies a MORE negative
        # (lower) implied share price -- q1's multiple produces the higher price here.
        valuation = pd.DataFrame(
            [
                {"metric": "ev_ebitda", "stat": "q1", "implied_share_price": -5.0},
                {"metric": "ev_ebitda", "stat": "q3", "implied_share_price": -20.0},
            ]
        )
        assert comps.implied_share_price_range(valuation, "ev_ebitda") == (-20.0, -5.0)


class TestPlotFootballField:
    def test_empty_ranges_raises(self):
        with pytest.raises(ValueError):
            comps.plot_football_field({})

    def test_renders_without_error(self):
        import matplotlib

        matplotlib.use("Agg")
        fig = comps.plot_football_field({"Method A": (10.0, 20.0), "Method B": (15.0, 30.0)})
        assert fig is not None
