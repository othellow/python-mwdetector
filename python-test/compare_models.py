from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Use the exact same split logic for fairness/reproducibility
from main import load_dataset, split_features_target, holdout_split

# Reuse training rules so preprocessing input types match training
from train import DROP_COLS, infer_numeric_columns

# -----------------------------
# Paths / configuration
# -----------------------------
DATA_PATH = Path("data/malwareproj.csv")          # preferred canonical location
FALLBACK_DATA_PATH = Path("malwareproj.csv")      # fallback if you still keep it in root

MODELS_DIR = Path("models")
OUT_CSV = Path("compare_results.csv")

# Model filenames to compare (as requested)
SKLEARN_MODELS = {
    "random_forest": MODELS_DIR / "random_forest.pkl",
    "logistic_regression": MODELS_DIR / "logistic_regression.pkl",
    "decision_tree": MODELS_DIR / "decision_tree.pkl",
}
TORCH_MODEL_NAME = "pytorch_mlp"
TORCH_MODEL_PATH = MODELS_DIR / "pytorch_mlp.pt"


def pick_dataset_path() -> Path:
    """Pick the dataset path (data/ preferred, else root fallback)."""
    if DATA_PATH.exists():
        return DATA_PATH
    return FALLBACK_DATA_PATH


def pick_preprocessor_path(model_key: str) -> Path:
    """
    Choose the correct preprocessor for a given model.

    - Random Forest baseline: prefer models/random_forest_preprocessor.pkl (if you copied it)
    - Otherwise: prefer models/preprocessor.pkl (shared)
    - Fallback: root preprocessor.pkl (older layout)
    """
    if model_key == "random_forest":
        rf_pp = MODELS_DIR / "random_forest_preprocessor.pkl"
        if rf_pp.exists():
            return rf_pp

    shared_pp = MODELS_DIR / "preprocessor.pkl"
    if shared_pp.exists():
        return shared_pp

    root_pp = Path("preprocessor.pkl")
    return root_pp


def prepare_test_set(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Create the holdout test set with the exact same split logic and input typing
    expectations as training.

    Leakage prevention:
    - We do NOT fit anything here. We only split, drop columns, and coerce types.
    - Numeric/categorical column inference is done on X_train only to avoid test peeking.
    """
    # 1) Split X/y
    X, y = split_features_target(df, target_col="Label")

    # 2) Same holdout split for every model
    X_train, X_test, y_train, y_test = holdout_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Drop the same columns used in training
    X_train = X_train.drop(columns=DROP_COLS, errors="ignore")
    X_test = X_test.drop(columns=DROP_COLS, errors="ignore")

    # 4) Determine numeric/categorical columns from TRAIN only
    numeric_cols = infer_numeric_columns(X_train, min_numeric_rate=0.90)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    # 5) Make X_test types match training expectations before transform
    for c in numeric_cols:
        if c in X_test.columns:
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
    for c in categorical_cols:
        if c in X_test.columns:
            X_test[c] = X_test[c].astype("string")

    # Clean y_test to consistent strings for metric calculations
    y_test = y_test.astype("string").str.strip()

    return X_test, y_test, numeric_cols, categorical_cols


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Compute measurable metrics for binary classification ('0' vs '1').
    Returns metrics + confusion matrix.
    """
    # Ensure consistent string format
    y_true = y_true.astype("string").str.strip()
    y_pred = pd.Series(y_pred).astype("string").str.strip()

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label="1",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=["0", "1"])
    # cm = [[tn, fp],
    #       [fn, tp]]

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "confusion_matrix": cm.tolist(),
    }


def predict_sklearn(model_path: Path, preprocessor_path: Path, X_test: pd.DataFrame) -> np.ndarray:
    """Load sklearn model + preprocessor, transform X_test, predict labels."""
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    X_test_proc = preprocessor.transform(X_test)
    return model.predict(X_test_proc)


def predict_torch_mlp(preprocessor_path: Path, X_test: pd.DataFrame, weights_path: Path) -> np.ndarray:
    """
    Load PyTorch MLP weights and run inference.

    Architecture must match training in train_baselines.py:
      Linear(in_dim, 256) -> ReLU -> Dropout -> Linear(256, 64) -> ReLU -> Dropout -> Linear(64, 1)
    """
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyTorch is required to evaluate pytorch_mlp.pt. Install with: python -m pip install torch"
        ) from e

    preprocessor = joblib.load(preprocessor_path)
    X_test_proc = preprocessor.transform(X_test)

    X_np = np.asarray(X_test_proc, dtype=np.float32)
    X_t = torch.tensor(X_np, dtype=torch.float32)

    in_dim = X_np.shape[1]
    model = nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).view(-1).cpu().numpy()

    preds01 = (probs >= 0.5).astype(int)
    # Return string labels to match the rest of the pipeline ('0','1')
    return preds01.astype(str)


def main() -> None:
    # -----------------------------
    # Load dataset and create the shared test set once
    # -----------------------------
    dataset_path = pick_dataset_path()
    df = load_dataset(dataset_path)

    X_test, y_test, _, _ = prepare_test_set(df)

    results: list[dict] = []

    # -----------------------------
    # Evaluate sklearn models
    # -----------------------------
    for model_key, model_path in SKLEARN_MODELS.items():
        if not model_path.exists():
            # Skip missing models (so script still runs even if you haven't trained them yet)
            continue

        preprocessor_path = pick_preprocessor_path(model_key)
        if not preprocessor_path.exists():
            # If no preprocessor artifact exists, this model can't be evaluated fairly
            continue

        y_pred = predict_sklearn(model_path, preprocessor_path, X_test)
        metrics = compute_metrics(y_test, y_pred)

        results.append(
            {
                "model": model_key,
                "model_file": str(model_path),
                "preprocessor_file": str(preprocessor_path),
                **metrics,
            }
        )

    # -----------------------------
    # Evaluate PyTorch MLP (if present)
    # -----------------------------
    if TORCH_MODEL_PATH.exists():
        preprocessor_path = pick_preprocessor_path("mlp_torch")
        if preprocessor_path.exists():
            y_pred = predict_torch_mlp(preprocessor_path, X_test, TORCH_MODEL_PATH)
            metrics = compute_metrics(y_test, y_pred)
            results.append(
                {
                    "model": TORCH_MODEL_NAME,
                    "model_file": str(TORCH_MODEL_PATH),
                    "preprocessor_file": str(preprocessor_path),
                    **metrics,
                }
            )

    if not results:
        print("No models evaluated. Check that model files exist in ./models and preprocessors exist.")
        return

    # -----------------------------
    # Build ranked comparison table (best -> worst by F1)
    # -----------------------------
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="f1", ascending=False).reset_index(drop=True)

    # Print a clean comparison table
    display_cols = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    print("\n=== Model comparison (ranked by F1-score) ===")
    print(df_results[display_cols].to_string(index=False))

    # -----------------------------
    # Save results to CSV (measurable outputs)
    # -----------------------------
    df_results.to_csv(OUT_CSV, index=False)
    print(f"\nSaved results to: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()