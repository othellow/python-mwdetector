from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Reuse existing, consistent logic from main.py
from main import load_dataset, split_features_target, holdout_split

CSV_PATH = Path("malwareproj.csv")
MODEL_PATH = Path("model.pkl")
PREPROCESSOR_PATH = Path("preprocessor.pkl")

# Columns to drop BEFORE preprocessing (IDs / timestamps / high-cardinality categoricals)
DROP_COLS = [
    "SHA1",
    "Identify",
    "FirstSeenDate",
    "ImportedDlls",
    "ImportedSymbols",
]


def infer_numeric_columns(X: pd.DataFrame, min_numeric_rate: float = 0.90) -> list[str]:
    """
    Infer numeric feature columns even if the CSV was loaded as strings.

    We attempt numeric coercion column-by-column and mark a column numeric if
    >= min_numeric_rate of non-null entries can be parsed as numbers.

    Note on leakage:
    - We will call this on X_train only (not X_test), so we do not "peek" at
      test-set content/formatting when deciding feature types.
    """
    numeric_cols: list[str] = []
    for col in X.columns:
        s = X[col]
        non_null = int(s.notna().sum())
        if non_null == 0:
            continue

        coerced = pd.to_numeric(s, errors="coerce")
        numeric_rate = int(coerced.notna().sum()) / non_null

        if numeric_rate >= min_numeric_rate:
            numeric_cols.append(col)

    return numeric_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Build ColumnTransformer + per-type pipelines.

    Leakage prevention:
    - We will FIT this preprocessor ONLY on X_train.
    - We will TRANSFORM X_test using the already-fitted preprocessor.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Dense output avoids sparse matrices that can be awkward with some downstream code.
    # If your scikit-learn version doesn't support sparse_output, use sparse=False instead.
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def main() -> None:
    print("Loading dataset...")
    df = load_dataset(CSV_PATH)

    print("Splitting train/test...")
    # Split X/y from full dataset
    X, y = split_features_target(df, target_col="Label")

    # Hold out test set FIRST (before any preprocessing/feature engineering)
    X_train, X_test, y_train, y_test = holdout_split(
        X, y, test_size=0.2, random_state=42
    )

    # Drop columns after split (dropping doesn't learn statistics, but we keep it explicit here)
    X_train = X_train.drop(columns=DROP_COLS, errors="ignore")
    X_test = X_test.drop(columns=DROP_COLS, errors="ignore")

    # Decide numeric vs categorical columns from TRAIN ONLY (avoid test peeking)
    numeric_cols = infer_numeric_columns(X_train, min_numeric_rate=0.90)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    # Apply numeric coercion using the TRAIN-derived numeric column list
    for c in numeric_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    # Keep categoricals as strings consistently
    for c in categorical_cols:
        X_train[c] = X_train[c].astype("string")
        X_test[c] = X_test[c].astype("string")

    print("Building preprocessor...")
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    print("Fitting preprocessor...")
    # Fit ONLY on training data to prevent leakage
    preprocessor.fit(X_train)

    # Transform train/test using fitted preprocessor
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)  # computed for completeness / future eval use

    print("Training model...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_proc, y_train)

    print("Saving model...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print(f"Saved model to: {MODEL_PATH.resolve()}")
    print(f"Saved preprocessor to: {PREPROCESSOR_PATH.resolve()}")
    print("Training rows:", len(y_train))
    print("Test rows:", len(y_test))


if __name__ == "__main__":
    main()