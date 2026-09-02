# credit-scorecard

A logistic regression scorecard that predicts whether a company's credit is
**investment grade** or **non-investment grade** from its financial ratios — the
same style of screen used in private credit and leveraged finance before a deal
gets deeper analyst attention.

## What it does

- Loads the Kaggle "Corporate Credit Rating" dataset (2,029 agency ratings across
  ~1,700 US-listed companies, S&P/Moody's/Fitch/Egan-Jones/DBRS) and recasts the
  letter-grade rating into a binary target: **BBB and above = investment grade**,
  **BB and below = non-investment grade** — the standard cutoff that determines
  eligibility for many institutional mandates.
- Builds a focused, 7-ratio feature set across leverage, profitability, liquidity,
  and coverage — the categories a credit analyst actually reasons in, not every
  column the dataset happens to have.
- Trains a logistic regression classifier and evaluates it with ROC/AUC, a
  threshold-adjustable confusion matrix, and — the main event — a coefficient
  table with plain-English interpretation of what's actually driving each score.

## Why logistic regression

Interpretability is the point. A logistic regression's coefficients can be read
out and defended one by one to a credit committee ("a one std-dev increase in
debt/assets lowers the log-odds of investment grade by X"); a boosted tree's
can't, not without a second layer of explainability tooling on top. For a
scorecard whose whole job is to be auditable, that trade is worth making even
though it costs a few points of accuracy against a more opaque model.

## Structure

```
data/fetch_data.py       load + clean the data, construct the IG/non-IG target
scorecard/features.py    ratio formulas, feature selection, winsorizing, scaling
scorecard/model.py       train/test split, logistic regression training
scorecard/evaluate.py    ROC/AUC, confusion matrix, coefficient interpretation
notebooks/walkthrough.ipynb   full narrative: data -> features -> model -> evaluation -> examples
tests/test_features.py   unit tests on the ratio formulas
```

## Running it

```bash
pip install -r requirements.txt
python data/fetch_data.py       # confirms the data loads and shows class balance
pytest tests/                   # ratio formula tests
jupyter notebook notebooks/walkthrough.ipynb   # full walkthrough
```

## Limitations

This is a ratios-only screen on ~2,000 historical ratings, not a rating
methodology. It has no access to the qualitative overlays, industry-specific
adjustments, and forward-looking judgment that go into a real agency rating —
see the notebook's closing section for the fuller discussion.

## License

MIT
