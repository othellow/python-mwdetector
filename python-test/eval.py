from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Reuse the same dataset logic so the test split matches training
from main import load_dataset, split_features_target, holdout_split

# Reuse training-time column rules and numeric inference helper
from train import DROP_COLS, infer_numeric_columns

CSV_PATH = Path("malwareproj.csv")
MODEL_PATH = Path("model.pkl")
PREPROCESSOR_PATH = Path("preprocessor.pkl")


def main() -> None:
    # 1) Load malwareproj.csv
    df = load_dataset(CSV_PATH)

    # 2) Split X/y using Label (same as training)
    X, y = split_features_target(df, target_col="Label")

    # 3) Same holdout_split(random_state=42) so the test set matches training
    X_train, X_test, y_train, y_test = holdout_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Drop the same columns used in train.py (must match training preprocessing)
    X_train = X_train.drop(columns=DROP_COLS, errors="ignore")
    X_test = X_test.drop(columns=DROP_COLS, errors="ignore")

    # Load saved artifacts
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    # --- Small tweak: match input types to training before transform ---
    # 1) Determine numeric columns from X_train (same as training logic)
    numeric_cols = infer_numeric_columns(X_train, min_numeric_rate=0.90)
    # 2) Categorical columns are the remaining columns
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    # 3) Convert X_test numeric columns to real numbers (invalid parses -> NaN)
    for c in numeric_cols:
        if c in X_test.columns:
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    # 4) Convert X_test categorical columns to string
    for c in categorical_cols:
        if c in X_test.columns:
            X_test[c] = X_test[c].astype("string")
    # --- End tweak ---

    # 5) Transform X_test using fitted preprocessor (no fitting here = no leakage)
    X_test_proc = preprocessor.transform(X_test)

    # 6) Predict using the trained model
    y_pred = model.predict(X_test_proc)

    # Ensure y_test is clean strings ('0'/'1') for consistent metrics/labels
    y_test = y_test.astype("string").str.strip()
    y_pred = pd.Series(y_pred).astype("string").str.strip()

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        pos_label="1",
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=["0", "1"])

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=["0", "1"], zero_division=0))

    print("Confusion matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm)


if __name__ == "__main__":
    main()

    