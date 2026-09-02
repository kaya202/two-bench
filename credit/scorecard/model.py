"""
Train/test split and the primary logistic regression model.

Class balance check (from data/fetch_data.py on the real data): 57.4% IG vs
42.6% non-IG. That's a mild skew, not a severe one — but it's skewed enough
that plain accuracy would flatter a model that leans toward predicting the
majority class, and skewed enough to be worth correcting for rather than
ignoring. class_weight='balanced' reweights each class inversely to its
frequency during training, so the minority (non-IG) class isn't
underweighted in the loss just because it's less common in the sample.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def make_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25) -> Split:
    """Stratified train/test split.

    Stratifying on y keeps the ~57/43 IG/non-IG ratio consistent between
    train and test — with a plain random split, an unlucky draw could
    noticeably shift that balance in the (smaller) test set and make the
    evaluation metrics less representative.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return Split(X_train, X_test, y_train, y_test)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Fit the primary logistic regression model.

    class_weight='balanced' addresses the mild IG/non-IG imbalance (see
    module docstring). Features must already be standardized (see
    scorecard.features.scale_features) before calling this.
    """
    model = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def predict_proba(model: LogisticRegression, X: pd.DataFrame) -> pd.Series:
    """Predicted probability of investment grade (class 1) for each row."""
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="p_investment_grade")
