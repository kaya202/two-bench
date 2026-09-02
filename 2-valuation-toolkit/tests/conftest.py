"""
Shared pytest fixtures. Every fixture here is built from hand-verified
numbers (no network calls, no yfinance) so the whole suite runs fast and
deterministically -- see LIMITATIONS.md for why data/fetch.py itself isn't
unit tested (it talks to a live, unofficial external API; mocking it
meaningfully is out of scope for "basic tests for the pure-math functions").
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from valuation.company import Company


@pytest.fixture
def sample_company() -> Company:
    """
    A 3-year synthetic company, USD/USD (no currency mismatch), with
    numbers hand-verified during development (see module 1's UFCF check):
    2023 UFCF = 148.75, 2024 UFCF = 169.50.
    """
    financials = {
        2022: {"revenue": 1000.0, "ebitda": 250.0, "ebit": 200.0, "d_and_a": 50.0, "capex": 60.0, "nwc": 100.0, "tax_rate": 0.25},
        2023: {"revenue": 1100.0, "ebitda": 280.0, "ebit": 225.0, "d_and_a": 55.0, "capex": 65.0, "nwc": 110.0, "tax_rate": 0.25},
        2024: {"revenue": 1210.0, "ebitda": 310.0, "ebit": 250.0, "d_and_a": 60.0, "capex": 70.0, "nwc": 118.0, "tax_rate": 0.25},
    }
    return Company(
        ticker="TEST",
        name="Test Co",
        sector="Technology",
        financials=financials,
        share_price=50.0,
        shares_outstanding=100.0,
        net_debt=200.0,
        currency="USD",
        financial_currency="USD",
    )


@pytest.fixture
def currency_mismatched_company() -> Company:
    """A minimal company whose quote and financial-statement currencies differ, for guard tests."""
    return Company(
        ticker="MISMATCH",
        name="Mismatch Co",
        sector="Technology",
        financials={
            2024: {"revenue": 100.0, "ebitda": 20.0, "ebit": 15.0, "d_and_a": 5.0, "capex": 5.0, "nwc": 10.0, "tax_rate": 0.25}
        },
        share_price=10.0,
        shares_outstanding=100.0,
        net_debt=50.0,
        currency="USD",
        financial_currency="EUR",
    )
