# Known Limitations & Open Questions

This file exists so that every simplification, scope boundary, and unresolved
decision in this toolkit is stated explicitly rather than discovered later.
It's organized in three parts: what was actually verified against live data,
what the toolkit deliberately does *not* do (and why), and what still needs
a decision from the project owner before this is "done."

## Tested

Every module was exercised against live yfinance data during development,
not just unit-level logic. The walkthrough notebook and README screenshots
run on **Carnival Corporation (CCL)** — swapped in from an initial MSFT
version specifically because MSFT's scale made the credit overlay's output
too flat to be a convincing demo (see "Walkthrough company" below); MSFT
was kept as a deliberate contrast point in both the notebook and this
document, not deleted.

- `company.py` / `utils.py`: historical UFCF build, CAPM, WACC, both terminal
  value methods, discounting, multiple statistics — verified against a
  synthetic 3-year dataset with hand-checked arithmetic, and now covered by
  67 automated pytest tests (`tests/`, ~1.4s, no network calls).
- `data/fetch.py`: full pull tested against MSFT, AAPL, SAP, ORCL, CCL,
  RCL, NCLH, VIK; cache hit/miss timing confirmed (2.66s cold vs. 0.01s
  cached); graceful degradation confirmed against a deliberately invalid
  ticker.
- `dcf.py`: full `run_dcf` pipeline and sensitivity grid run against both
  MSFT and CCL; heatmap rendering checked visually, including the
  NaN-handling path for WACC/growth combinations outside Gordon Growth's
  valid domain.
- `comps.py`: peer selection, multiple fetching, and the football field
  chart run against a real 8-name tech peer set (MSFT case) and a 3-name
  pure-play cruise peer set (CCL case: RCL, NCLH, VIK); label-collision
  issues found and fixed by visual inspection, not assumed away.
- `precedent_transactions.py`: 8 real M&A deals individually researched and
  source-linked (see `data/precedent_transactions.csv`) — 6 enterprise-
  software deals (2014–2022) plus 2 cruise-line deals (2014, 2018,
  added specifically for the CCL walkthrough). Control premium tested in
  both directions: MSFT/tech-comps produced a **negative** premium
  (−5.7%), CCL/cruise-comps produced a **positive** one (+1.3%) — both are
  real, disclosed results (see below), not hidden either way.
- `credit_overlay.py`: leverage/coverage/recovery tables and chart run
  against both an MSFT-derived and a CCL-derived DCF enterprise value. Two
  real bugs were caught this way, not in code review: covenant-breach flags
  silently dropped before reaching the chart (MSFT run), and the DCF's
  manually-overridden tax rate (2%, for CCL's real IRC §883 shipping-
  income exemption) silently diverging from `compute_wacc`'s own tax-rate
  default, which fell back to CCL's *raw fetched* 2025 rate (0.4%) instead
  — two different numbers for the same real-world fact, caught by actually
  reading the executed WACC build-up table rather than assuming the
  numbers were consistent.
- Currency-mismatch guard: confirmed SAP reports `currency="USD"` (its
  ADR's quote currency) but `financialCurrency="EUR"` (its statements) via
  direct inspection of yfinance's own output, then confirmed the new guard
  actually raises in `enterprise_value_market` and `dcf.compute_wacc` for
  SAP, and does *not* false-positive on MSFT/AAPL/ORCL/CCL/RCL/NCLH/VIK
  (all USD/USD).

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

**Auto-fetched beta can be unusually high for a volatile/cyclical sector,
and this toolkit doesn't correct it.** CCL's Yahoo-reported beta is ~2.3,
well above the ~1.5–2.0 more commonly cited as a "normalized" cruise/
leisure-travel beta — plausibly because Yahoo's undocumented lookback
window (see `fetch_beta`'s own docstring) still includes CCL's extreme
COVID-era volatility. The walkthrough notebook uses this fetched value as
the base case and flags it explicitly in markdown rather than silently
substituting a "more reasonable" number — the point of pulling beta live
is to show what the data actually says, including when it's a value worth
a human's skepticism.

**Manual tax-rate overrides need to be applied consistently across both
`project_ufcf` and `compute_wacc`, and nothing enforces that.** CCL's real
effective tax rate is minimal — Panamanian-incorporated cruise operators
qualify for the IRC §883 exemption on qualifying international shipping
income, well documented in Carnival's own 10-Ks — so the walkthrough
overrides the DCF's forward tax-rate assumption to 2%. `compute_wacc` has
its own independent `tax_rate`
parameter (used for the after-tax cost-of-debt calc) that, if left
unset, falls back to `company.latest("tax_rate")` — CCL's *raw fetched*
2025 rate (0.4%), a different number for the same underlying real-world
reason. This was caught during development by reading the executed WACC
table and noticing the tax rate shown didn't match the DCF's assumption;
the fix was passing `tax_rate` explicitly into both call sites, not a
code change, since both defaults are working exactly as documented — the
toolkit doesn't (and can't, in general) know that two separately-named
parameters in two different functions should represent "the same" real-
world quantity for a given company. Worth checking any time a manual
override matters to more than one function call.

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
the curated precedent-transaction dataset (8 deals total, per the build
spec's own "5–10 real, sourced transactions" ask, applied per sector — 6
enterprise-software, 2 cruise-line) both produce quartile statistics that
are illustrative, not statistically robust — one data point can move a
quartile meaningfully at this N. This is most acute for CCL's own
peer/deal sets: only 3 pure-play public cruise comps exist (RCL, NCLH,
VIK) and only 2 sourced cruise-sector precedent deals exist, because CCL,
RCL, and NCLH together represent the large majority of global ocean cruise
capacity — there simply isn't a larger universe of comparable public
companies or historical deals to draw from, not a research shortfall.
`comps.summarize_peer_multiples` and
`precedent_transactions.summarize_transaction_multiples` both emit an
explicit warning when a metric has fewer than 5 usable data points, so
this surfaces at runtime rather than only in this document.

**Precedent-transaction sector labels are a custom taxonomy**, not
yfinance's coarser GICS `sector` field. `filter_transactions(sector=...)`
expects values like `"Enterprise Infrastructure Software"` (this dataset's
own hand-curated labels), not `"Technology"` (what `fetch_sector()` / the
peer-selection sector filter in `comps.select_peers` returns) — passing one
where the other is expected will silently match zero rows. This is now
called out explicitly in `filter_transactions`'s docstring.

**Control premium can come out negative — or positive, depending on
vintage/growth-profile alignment.** Tested both ways: the MSFT/tech-comps
pairing (used in the notebook) produced **−5.7%**, because the curated
2014–2022 software deal set doesn't match a *current* (2026) trading-comps
peer set that includes richly-rated, higher-growth names like ServiceNow
and Apple. The CCL/cruise-comps pairing produced **+1.3%** — a much closer
match, since both the deal multiples (12.2x–14.0x) and the current peer
multiples (8.4x–16.2x, median 12.9x) happen to sit in a similar band. Note
this is close to coincidental (n=2 deals, n=3 peers), not proof the
cruise-sector pairing is more "correct" methodology than the software one
— it's the small-sample-size caveat above showing up again from a
different angle. Either sign is a legitimate answer, not a bug;
`control_premium()`'s docstring explains both cases explicitly.

**Cache has no automatic expiry.** `data/fetch.py` caches yfinance pulls
indefinitely until `refresh=True` is passed. A `cache_age(ticker)` helper
now exists to check staleness, but nothing calls it automatically — a
notebook re-run months from now will silently reuse old market data unless
someone remembers to check or refresh.

## Engineering scope

- **Automated tests: done.** 67 pytest tests across 6 files (`tests/`),
  all against hand-verified synthetic data, no network calls, ~1.4s.
- **Linting/type-checking (ruff/mypy): deliberately skipped**, per a direct
  decision — full mypy/ruff enforcement judged lower value for the effort
  than other polish, given the original spec only asked for pytest.
- **CI: added.** `.github/workflows/tests.yml` runs pytest on every push,
  matrix'd across Python 3.11/3.12, with no network calls (so it doesn't
  depend on yfinance's uptime). The README's CI badge needs the real GitHub
  `OWNER/REPO` path substituted in — see open questions.
- **LICENSE: added.** MIT (`LICENSE` at repo root) — the copyright line
  currently reads `[Your Name]`, a placeholder that needs the actual
  name filled in before this is truly finished — see open questions.
- **`data/fetch.py` retains a `sys.path` insert** as a zero-install
  convenience, alongside `pyproject.toml` (which enables a proper
  `pip install -e .`). Not removed outright, to avoid risking the
  currently-working import path under time pressure.
- **`requirements.txt` pins the exact versions this was built and tested
  against**, including `jupyter` now that the notebook has actually been
  built and executed end to end (pandas 3.0.5, numpy 2.5.2, matplotlib
  3.11.1, yfinance 1.7.0, pytest 9.1.1, jupyter 1.1.1).

## Status: all 9 build-order modules complete, decisions made

Everything through `README.md` (module 9) is built, tested, and executed
end to end. Five decisions that were open questions in an earlier version
of this document have since been made explicitly by the project owner:

1. **Walkthrough company: switched from MSFT to Carnival Corp (CCL).**
   MSFT's credit_overlay output was too flat (100% recovery even at 6x +
   severe stress) to be a convincing demo of the toolkit's differentiator.
   CCL has real, recent, sourced leverage history (peaked ~6.8x net
   debt/EBITDA at end of 2023, deleveraged to ~3.1x–3.4x since) that maps
   directly onto the credit overlay's 3x–6x sweep, and its downside
   scenario actually breaches covenant and wipes out equity at 6x — a
   materially different, more informative chart. MSFT's flat result is
   kept as a one-line contrast in both the notebook and this document,
   not deleted, because it's a genuinely sharp finding in its own right
   ("interest coverage binds before recovery risk for a firm this size").
2. **LICENSE: added, MIT.**
3. **Linting/type-checking (ruff/mypy): skipped deliberately** — judged
   lower value for the effort than other polish.
4. **CI: added** — GitHub Actions running pytest on every push.
5. **Precedent-transaction dataset: left at 6 for the enterprise-software
   set** (not padded toward 10 just to hit a round number — quality/
   sourcing mattered more than count). Separately, 2 more real cruise-
   sector deals were added specifically to support the CCL walkthrough,
   for the reason explained under "Small sample sizes" above.
