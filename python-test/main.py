import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Path to your dataset (edit if needed)
CSV_PATH = Path("malwareproj.csv")


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load the CSV dataset into a DataFrame.

    dtype=str keeps values consistent (e.g., Label stays '0'/'1' strings)
    and helps avoid accidental float coercion.
    """
    return pd.read_csv(csv_path, dtype=str)


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate dataset structure + Label distribution.
    Returns a dict of computed values so printing is handled elsewhere.
    """
    n_rows = df.shape[0]
    n_cols = df.shape[1]
    columns = df.columns.tolist()
    has_label = "Label" in df.columns

    count_0 = None
    count_1 = None
    is_binary = None

    if has_label:
        # Normalize Label: string dtype + strip whitespace + keep NaN safe
        label_series = df["Label"].astype("string").str.strip()

        count_0 = int((label_series == "0").sum())
        count_1 = int((label_series == "1").sum())

        # Binary classification iff non-null unique values are exactly {'0','1'}
        unique_non_null = set(label_series.dropna().unique().tolist())
        is_binary = (unique_non_null == {"0", "1"})

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
        "has_label": has_label,
        "count_0": count_0,
        "count_1": count_1,
        "is_binary": is_binary,
    }


def show_summary(csv_path: Path, results: dict) -> None:
    """
    Print the same outputs as your current script, in the same order.
    """
    print("File type:", csv_path.suffix)
    print("Number of rows:", results["n_rows"])
    print("Number of columns:", results["n_cols"])
    print("Column names:", results["columns"])
    print("Has target column named 'Label'?", results["has_label"])

    if results["has_label"]:
        print("Count(Label == 0):", results["count_0"])
        print("Count(Label == 1):", results["count_1"])
        print("Binary classification dataset?", results["is_binary"])


def split_features_target(df: pd.DataFrame, target_col: str = "Label") -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y).
    """
    # Keep y as a clean string series ('0'/'1') to match earlier validation logic
    y = df[target_col].astype("string").str.strip()
    X = df.drop(columns=[target_col])
    return X, y


def holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Hold out 20% test set BEFORE any preprocessing/feature engineering.
    Stratify by y to preserve Label class ratio.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def print_split_summary(y_train: pd.Series, y_test: pd.Series) -> None:
    """
    Print train/test sizes and class counts.
    """
    print("Training rows:", int(y_train.shape[0]))
    print("Test rows:", int(y_test.shape[0]))

    # Show class counts as a dict-like mapping, e.g. {'0': 21116, '1': 29065}
    train_counts = y_train.value_counts(dropna=False).to_dict()
    test_counts = y_test.value_counts(dropna=False).to_dict()

    print("Training class counts:", train_counts)
    print("Test class counts:", test_counts)


def main() -> None:
    df = load_dataset(CSV_PATH)

    # Requirement 1: keep existing dataset validation outputs
    results = validate_dataset(df)
    show_summary(CSV_PATH, results)

    # Requirements 2-6: X/y split and 80/20 holdout BEFORE preprocessing, stratified
    if not results["has_label"]:
        # Without Label, we cannot split into X/y or stratify.
        return

    X, y = split_features_target(df, target_col="Label")
    X_train, X_test, y_train, y_test = holdout_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Requirement 7: print split sizes and class counts
    print_split_summary(y_train, y_test)


if __name__ == "__main__":
    main()