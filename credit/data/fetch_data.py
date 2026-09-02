"""
Load the Corporate Credit Rating dataset and construct the binary
investment-grade (IG) / non-investment-grade (non-IG) target.

Source: "Corporate Credit Rating" (Kaggle, agewerc/corporate-credit-rating) —
2,029 credit ratings issued by five agencies (S&P, Moody's, Fitch, Egan-Jones,
DBRS) to US-listed companies, each paired with 25 financial ratios computed
from the company's financial statements.

Kaggle requires a personal API token to download programmatically, which
doesn't travel well into a shared repo or CI. Since the raw file is small and
static, a copy is bundled at data/corporate_rating.csv (fetched once from the
dataset author's own GitHub mirror) and loaded directly. Re-running the
Kaggle download isn't part of this script.

Rating scale check (confirmed against the actual file, not assumed): the
dataset uses whole-letter S&P-style grades only —
    AAA, AA, A, BBB, BB, B, CCC, CC, C, D
with no +/- modifiers, even for rows sourced from Moody's, which normally uses
Aaa/Aa/A/Baa/Ba/etc. notation. All five agencies' ratings have already been
normalized to this one scale, so a single mapping applies across the board.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
RAW_PATH = DATA_DIR / "corporate_rating.csv"

# Investment grade is the standard "BBB-/Baa3 and above" cutoff used by
# institutional mandates and regulatory capital treatment. This dataset has
# no +/- or numeric modifiers, so the cutoff collapses to: BBB and above is
# investment grade, BB and below is not.
INVESTMENT_GRADE_RATINGS = {"AAA", "AA", "A", "BBB"}


def load_data(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the bundled CSV and construct the binary IG target.

    Returns the raw columns plus one new column, `is_investment_grade`
    (1 = investment grade, 0 = non-investment grade).
    """
    df = pd.read_csv(raw_path)

    # Minimal cleaning: the source file has no missing values or duplicate
    # rows, but check rather than assume, since this script is the one place
    # a silent data-quality issue would otherwise slip through unnoticed.
    n_missing = df.isnull().sum().sum()
    if n_missing:
        df = df.dropna()
    n_dupes = df.duplicated().sum()
    if n_dupes:
        df = df.drop_duplicates()

    unknown_ratings = set(df["Rating"].unique()) - (
        INVESTMENT_GRADE_RATINGS | {"BB", "B", "CCC", "CC", "C", "D"}
    )
    if unknown_ratings:
        raise ValueError(f"Unexpected rating values not in the mapping: {unknown_ratings}")

    df["is_investment_grade"] = df["Rating"].isin(INVESTMENT_GRADE_RATINGS).astype(int)

    return df


def summarize(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df)}")
    print("\nRating distribution:")
    print(df["Rating"].value_counts())
    print("\nIG (1) vs non-IG (0) class balance:")
    print(df["is_investment_grade"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    data = load_data()
    summarize(data)
