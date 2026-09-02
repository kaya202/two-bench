"""
Evaluation: ROC/AUC, confusion matrix, and the coefficient table.

Threshold choice: the default classification threshold of 0.5 isn't
necessarily the right one for this use case. A lender or investor screening
for investment-grade eligibility is typically more worried about a false
positive — the model saying "investment grade" for a company that's actually
non-IG — than a false negative, since the former risks putting a
sub-investment-grade credit into a mandate that isn't supposed to hold it.
That argues for a *higher* threshold than 0.5 (require more confidence before
calling something IG), trading some recall on true IG names for fewer costly
false positives. The threshold is a parameter here rather than hardcoded, so
this tradeoff can be shown explicitly in the notebook.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def compute_roc(y_true: pd.Series, y_proba: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    """False-positive rate, true-positive rate, and AUC across all thresholds."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    return fpr, tpr, auc


def plot_roc(y_true: pd.Series, y_proba: pd.Series, ax: plt.Axes = None) -> plt.Axes:
    fpr, tpr, auc = compute_roc(y_true, y_proba)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve: investment-grade classification")
    ax.legend()
    return ax


def confusion_at_threshold(y_true: pd.Series, y_proba: pd.Series, threshold: float = 0.5) -> pd.DataFrame:
    """Confusion matrix at a given probability threshold, labeled for readability."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Actual: non-IG", "Actual: IG"],
        columns=["Predicted: non-IG", "Predicted: IG"],
    )


def coefficient_table(model: LogisticRegression, feature_names: list[str]) -> pd.DataFrame:
    """Feature name, coefficient, and a plain-English interpretation.

    Coefficients are on standardized features, so each one reads as: "a one
    standard-deviation increase in this ratio changes the log-odds of being
    investment grade by this amount, holding the other ratios fixed."
    """
    coefs = model.coef_[0]
    rows = []
    for name, coef in zip(feature_names, coefs):
        direction = "increases" if coef > 0 else "decreases"
        rows.append(
            {
                "feature": name,
                "coefficient": round(coef, 3),
                "interpretation": (
                    f"A one std-dev increase in {name} {direction} the log-odds of "
                    f"investment-grade classification by {abs(coef):.3f}, holding other ratios fixed."
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
    return table.reset_index(drop=True)
