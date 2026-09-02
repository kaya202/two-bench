"""
valuation-toolkit
==================

A modular valuation engine (DCF, trading comps, precedent transactions)
with a credit-analysis overlay that ties enterprise value into debt
capacity and downside recovery analysis.
"""

from .company import Company

__all__ = ["Company"]
