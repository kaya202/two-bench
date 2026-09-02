# Known Limitations & Open Questions

This file exists so that every simplification, scope boundary, and unresolved
decision in this toolkit is stated explicitly rather than discovered later.
It's organized in three parts: what was actually verified against live data,
what the toolkit deliberately does *not* do (and why), and what still needs
a decision from the project owner before this is "done."

## What has actually been tested

Every module was exercised against live yfinance data during development,
not just unit-level logic:

- `company.py` / `utils.py`: historical UFCF build, CAPM, WACC, both terminal
  value methods, discounting, multiple statistics — verified against a
  synthetic 3-year dataset with hand-checked arithmetic.
- `data/fetch.py`: full pull tested against MSFT, AAPL, SAP, ORCL; cache
  hit/miss timing confirmed (2.66s cold vs. 0.01s cached); graceful
  degradation confirmed against a deliberately invalid ticker.
- `dcf.py`: full `run_dcf` pipeline and sensitivity grid run against MSFT;
  heatmap rendering checked visually, including the NaN-handling path for
  WACC/growth combinations outside Gordon Growth's valid domain.
- `comps.py`: peer selection, multiple fetching, and the football field
  chart run against a real 7-name tech peer set; label-collision issues
  found and fixed by visual inspection, not assumed away.
- `precedent_transactions.py`: 6 real M&A deals individually researched and
  source-linked (see `data/precedent_transactions.csv`); control premium
  calculation run against the MSFT/tech-comps case, which produced a
  **negative** premium — a real, disclosed result (see below), not hidden.
- `credit_overlay.py`: leverage/coverage/recovery tables and chart run
  against an MSFT-derived DCF enterprise value; a real bug (covenant-breach
  flags silently dropped before reaching the chart) was caught this way and
  fixed.
- Currency-mismatch guard: confirmed SAP reports `currency="USD"` (its
  ADR's quote currency) but `financialCurrency="EUR"` (its statements) via
  direct inspection of yfinance's own output, then confirmed the new guard
  actually raises in `enterprise_value_market` and `dcf.compute_wacc` for
  SAP, and does *not* false-positive on MSFT/AAPL/ORCL (all USD/USD).

## Deliberate scope boundaries (not bugs — disclosed tradeoffs)

**Currency: detected and blocked, not converted.** A company whose quote
currency differs from its financial-statement currency (confirmed real for
SAP; likely true of other foreign issuers trading via a USD ADR) will raise
a clear `ValueError` from `Company.assert_single_currency()` rather than
silently blending two currencies into a nonsense multiple. The toolkit does
**not** perform FX conversion — if you want to value such a company, you
must supply currency-consistent overrides yourself. This was found through
direct testing, not assumed; before the fix, `comps.py`/`dcf.py` would have
silently produced numbers wrong by roughly the EUR/USD rate for a ticker
like SAP.

**Cost of debt is a required manual input**, not derived from the
company's own bonds/credit spread. Reliable pre-tax yield data isn't
reconstructable from income-statement-level data — an analyst reads this
off the company's actual debt in practice, and this toolkit expects the
same input, not an automated shortcut.

**Net debt (book value) is used as the proxy for market value of debt** in
WACC weighting. True market value of debt is rarely observable; this is
standard practice, but it means a net-cash company (net debt < 0) will
produce a negative debt weight in the WACC blend — a known point of debate
in corporate finance practice (some analysts use gross debt instead,
precisely to avoid this), not a defect in this implementation.

**EBITDA/EBIT/D&A are used as independently reported**, not forced to
algebraically reconcile (EBITDA − D&A is not asserted to equal EBIT). In
every ticker actually checked during development (MSFT, SAP, ORCL) these
reconciled exactly across all fetched years — but this was only checked for
those three large-cap tickers, not proven true in general, so treat it as a
residual risk on an untested ticker rather than a guaranteed non-issue.

**Capex sign handling assumes a net capital outflow.** `fetch_financials`
takes `abs()` of yfinance's reported capex figure to match this toolkit's
sign convention. For a company with a genuine net capex *inflow* in a given
year (e.g. a large one-off asset sale exceeding purchases), this would
incorrectly flip that inflow into an apparent outflow. This is a real
theoretical edge case, not one actually encountered in testing (all tickers
tested had ordinary net capex outflows).

**Credit overlay is a single-tranche, single-scenario, static model.**
`leverage_analysis`/`downside_recovery` treat the entire assumed debt
quantum as one tranche senior to all equity (no subordination waterfall
across multiple debt classes), and `downside_recovery` runs one stress
scenario per call rather than a distribution of outcomes. There is also no
feedback loop from leverage level back into enterprise value itself — real
financial-distress costs at very high leverage aren't modeled (a
Modigliani-Miller-style simplification, stated explicitly in the module's
own docstring). This is intentional scope for a valuation-toolkit overlay,
not a substitute for a full LBO or Monte Carlo credit model.

**Small sample sizes throughout.** Realistic peer sets (5–10 tickers) and
the curated precedent-transaction dataset (6 deals, per the build spec's
own "5–10 real, sourced transactions" ask) both produce quartile statistics
that are illustrative, not statistically robust — one data point can move
a quartile meaningfully at this N. `comps.summarize_peer_multiples` and
`precedent_transactions.summarize_transaction_multiples` both now emit an
explicit warning when a metric has fewer than 5 usable data points, so this
surfaces at runtime rather than only in this document.

**Precedent-transaction sector labels are a custom taxonomy**, not
yfinance's coarser GICS `sector` field. `filter_transactions(sector=...)`
expects values like `"Enterprise Infrastructure Software"` (this dataset's
own hand-curated labels), not `"Technology"` (what `fetch_sector()` / the
peer-selection sector filter in `comps.select_peers` returns) — passing one
where the other is expected will silently match zero rows. This is now
called out explicitly in `filter_transactions`'s docstring.

**Control premium can come out negative.** Tested directly: pairing the
curated 2014–2022 precedent-transaction set against a *current* (2026)
trading-comps peer set for MSFT produced a **−11.4%** "control premium" —
the peer set's richly-rated, high-growth names outvalued the historical
deal multiples. This isn't a bug in the formula; it's a vintage/growth-
profile mismatch between the two inputs, and `control_premium()`'s
docstring now says so explicitly rather than letting a negative number
read as a broken calculation.

**Cache has no automatic expiry.** `data/fetch.py` caches yfinance pulls
indefinitely until `refresh=True` is passed. A `cache_age(ticker)` helper
now exists to check staleness, but nothing calls it automatically — a
notebook re-run months from now will silently reuse old market data unless
someone remembers to check or refresh.

## Engineering scope not yet addressed

- **No automated test suite yet.** `tests/` is scheduled as the next build
  step. Until it lands, all verification above was manual/ad hoc, run and
  discarded — there is no regression protection against a future edit
  breaking something that currently works.
- **No linting or type-checking configured** (no ruff/mypy). Not requested
  by the original build spec (which asks only for pytest), so this wasn't
  added unilaterally — see open questions below.
- **No CI workflow** (e.g. GitHub Actions running pytest on push). Same
  reasoning — not in the original spec, flagged as an open question rather
  than assumed.
- **No LICENSE file.** A public GitHub repo with no license is, by default,
  all-rights-reserved — worth a deliberate choice, not an oversight.
- **`data/fetch.py` retains a `sys.path` insert** as a zero-install
  convenience, alongside the new `pyproject.toml` (which enables a proper
  `pip install -e .`). Not removed outright, to avoid risking the
  currently-working import path under time pressure — see open questions.
- **`requirements.txt` pins the exact versions this was built and tested
  against** (pandas 3.0.5, numpy 2.5.2, matplotlib 3.11.1, yfinance 1.7.0,
  pytest 9.1.1) except `jupyter`, left as a floor since it hasn't been
  installed/exercised yet this session — will pin once the notebook module
  is built and actually run end to end.

## Open questions for the project owner

1. **Walkthrough company for the notebook.** MSFT was used throughout
   development/testing. It tells a clean equity-valuation story, but its
   credit_overlay output is somewhat flat (100% recovery even at 6x
   leverage + a 25% EBITDA stress — interest coverage covenants bind well
   before recovery risk does, at MSFT's scale). Keep MSFT and lean into
   that as the narrative ("here's why credit doesn't lever mega-cap tech"),
   or switch to a more leverage-relevant mid-cap for a punchier
   credit_overlay demo?
2. **LICENSE.** No license currently exists. MIT is the conventional
   default for a portfolio piece meant to be read/reused freely — add it,
   pick something else, or skip it deliberately?
3. **Linting/type-checking (ruff/mypy).** Not in the original spec. Worth
   adding for extra polish, or keep scope to exactly what was asked
   (pytest only)?
4. **CI (GitHub Actions running pytest on push).** Same question — polish
   beyond the original spec, add or skip?
5. **Precedent-transaction dataset size.** Currently 6 real, sourced deals,
   matching the spec's own "5–10" ask. Acceptable as-is, or should more
   deals be researched to round it out toward the upper end of that range?

This file should be linked from the README once that's built, so it stays
visible to anyone reviewing the repo rather than buried in commit history.
