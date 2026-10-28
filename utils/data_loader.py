"""Dataset loading, validation, splitting, and deterministic sample helpers."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "online_shoppers_intention.csv"

FEATURE_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
TARGET_COLUMN = "Revenue"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
RANDOM_STATE = 42


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise a clear error when the dataset cannot be used by this project."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Revenue contains missing values.")
    classes = set(df[TARGET_COLUMN].astype(int).unique())
    if classes != {0, 1}:
        raise ValueError("Revenue must contain both binary classes (True/False).")


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the untouched local CSV and validate its schema."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    validate_dataset(df)
    return df


def split_dataset(df: pd.DataFrame):
    """Return the single stratified split used by training, tests, and samples."""
    validate_dataset(df)
    X = df[FEATURE_COLUMNS].copy()
    X["Weekend"] = X["Weekend"].astype(bool)
    y = df[TARGET_COLUMN].astype(int)
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def get_test_sample(df: pd.DataFrame, purchase: bool) -> tuple[pd.Series, int]:
    """Select the first matching held-out record, independent of predictions."""
    _, X_test, _, y_test = split_dataset(df)
    matches = y_test[y_test == int(purchase)]
    if matches.empty:
        raise ValueError(f"No {'purchase' if purchase else 'non-purchase'} test sample found.")
    index = matches.index[0]
    return X_test.loc[index, FEATURE_COLUMNS].copy(), int(y_test.loc[index])

