"""
cross_validate_pytorch_mlp.py

Statistically correct model selection + evaluation workflow (mirrors cross_validate_models.py):
- Step 1: Create an untouched 20% holdout test set (stratified, random_state=42)
- Step 2: Use only the 80% training portion for model selection
- Step 3: Within training, run Stratified 10-Fold Cross-Validation + hyperparameter tuning
- Step 4: Retrain best model on full training set
- Step 5: Evaluate final tuned model on the untouched holdout test set

This script reuses:
- main.py: load_dataset, split_features_target, holdout_split
- train.py: DROP_COLS, infer_numeric_columns

It avoids leakage by fitting preprocessing inside each CV fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Project reuse (required)
from main import load_dataset, split_features_target, holdout_split
from train import DROP_COLS, infer_numeric_columns


# -----------------------------
# Paths / outputs (required)
# -----------------------------
DATA_PATH = Path("data/malwareproj.csv")
FALLBACK_DATA_PATH = Path("malwareproj.csv")

CV_OUT = Path("pytorch_mlp_cv_results.csv")
HOLDOUT_OUT = Path("pytorch_mlp_holdout_results.csv")


# -----------------------------
# Reproducibility / device
# -----------------------------
SEED = 42
DEVICE = torch.device("cpu")  # required CPU-only


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU only here, but keep deterministic knobs explicit:
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older torch versions may not support this; training remains seeded.
        pass


# -----------------------------
# Data / preprocessing utilities
# -----------------------------
def pick_dataset_path() -> Path:
    """Prefer data/ location; fallback to root if still present."""
    return DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH


def y_to_binary_int(y: pd.Series) -> np.ndarray:
    """Convert Label series ('0'/'1') to ints 0/1 for metrics."""
    y_clean = y.astype("string").str.strip()
    return (y_clean == "1").astype(int).to_numpy()


def build_preprocessor_from_train(
    X_train: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build ColumnTransformer based on TRAIN columns only.
    IMPORTANT: infer numeric/categorical from X_train only to avoid peeking.
    """
    numeric_cols = infer_numeric_columns(X_train, min_numeric_rate=0.90)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def enforce_column_dtypes(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> None:
    """
    Apply dtype normalization using TRAIN-derived column typing.
    Mutates the passed DataFrames (intended).
    """
    for c in numeric_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_other[c] = pd.to_numeric(X_other[c], errors="coerce")

    for c in categorical_cols:
        X_train[c] = X_train[c].astype("string")
        X_other[c] = X_other[c].astype("string")


def transform_to_tensors(
    preprocessor: ColumnTransformer,
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    X_val_df: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fit preprocessor on X_train_df only, transform both splits,
    and convert to CPU torch tensors.
    """
    X_train_np = preprocessor.fit_transform(X_train_df)
    X_val_np = preprocessor.transform(X_val_df)

    y_train_np = y_to_binary_int(y_train).astype(np.float32)
    y_val_np = y_to_binary_int(y_val).astype(np.float32)

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE).view(-1, 1)
    y_val_t = torch.tensor(y_val_np, dtype=torch.float32, device=DEVICE).view(-1, 1)

    return X_train_t, y_train_t, X_val_t, y_val_t


# -----------------------------
# PyTorch model / training
# -----------------------------
class MLPBinaryClassifier(nn.Module):
    """
    One-hidden-layer MLP with ReLU, optional dropout, sigmoid output.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class HParams:
    hidden_dim: int
    learning_rate: float
    batch_size: int
    epochs: int


def train_one_model(
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_val_t: torch.Tensor,
    hparams: HParams,
    seed: int,
) -> np.ndarray:
    """
    Train on (X_train_t, y_train_t) and return validation probabilities for X_val_t.
    Uses CPU only. Re-seeds per fold for reproducibility.
    """
    set_global_seeds(seed)

    input_dim = int(X_train_t.shape[1])
    model = MLPBinaryClassifier(
        input_dim=input_dim,
        hidden_dim=hparams.hidden_dim,
        dropout=0.2,
    ).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True)

    model.train()
    for _ in range(hparams.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_probs = model(X_val_t).detach().cpu().numpy().reshape(-1)

    return val_probs


# -----------------------------
# Main workflow
# -----------------------------
def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    set_global_seeds(SEED)

    # 1) Load dataset using project logic (required)
    df = load_dataset(pick_dataset_path())

    # 2) Split into X/y and create untouched holdout test set (80/20, stratified, rs=42) (required)
    X, y = split_features_target(df, target_col="Label")
    X_train_full, X_test, y_train_full, y_test = holdout_split(
        X, y, test_size=0.2, random_state=42
    )

    # Drop columns after split (matches project style; no learned stats here)
    X_train_full = X_train_full.drop(columns=DROP_COLS, errors="ignore")
    X_test = X_test.drop(columns=DROP_COLS, errors="ignore")

    # Hyperparameter grid (required slim grid)
    grid: list[HParams] = [
        HParams(hidden_dim=64, learning_rate=0.001, batch_size=32, epochs=20),
        HParams(hidden_dim=64, learning_rate=0.001, batch_size=32, epochs=30),
        HParams(hidden_dim=128, learning_rate=0.001, batch_size=32, epochs=20),
        HParams(hidden_dim=128, learning_rate=0.001, batch_size=32, epochs=30),
    ]

    # 8) Stratified 10-Fold CV on training set only (required)
    y_train_clean = y_train_full.astype("string").str.strip()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    cv_rows: list[dict] = []

    for hp in grid:
        fold_accs: list[float] = []
        fold_aucs: list[float] = []

        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train_full, y_train_clean), start=1):
            X_tr_df = X_train_full.iloc[tr_idx].copy()
            y_tr = y_train_full.iloc[tr_idx].copy()
            X_va_df = X_train_full.iloc[va_idx].copy()
            y_va = y_train_full.iloc[va_idx].copy()

            # Build fold-specific preprocessor from fold-train only (leakage-safe)
            preprocessor, num_cols, cat_cols = build_preprocessor_from_train(X_tr_df)
            enforce_column_dtypes(X_tr_df, X_va_df, num_cols, cat_cols)

            # 6) Convert transformed matrices into PyTorch tensors (required)
            X_tr_t, y_tr_t, X_va_t, y_va_t = transform_to_tensors(
                preprocessor=preprocessor,
                X_train_df=X_tr_df,
                y_train=y_tr,
                X_val_df=X_va_df,
                y_val=y_va,
            )

            # 10) Train fold model, predict val probabilities, compute Accuracy and ROC-AUC (required)
            # Seed is varied by fold+hp to keep runs reproducible but non-identical across folds.
            fold_seed = SEED + 1000 * fold_idx + 10 * hp.hidden_dim + hp.epochs
            va_probs = train_one_model(
                X_train_t=X_tr_t,
                y_train_t=y_tr_t,
                X_val_t=X_va_t,
                hparams=hp,
                seed=fold_seed,
            )

            y_va_int = y_to_binary_int(y_va)
            y_va_pred = (va_probs >= 0.5).astype(int)

            acc = float(accuracy_score(y_va_int, y_va_pred))
            # roc_auc_score requires both classes present; with stratification this should hold,
            # but guard anyway to avoid rare edge cases in tiny datasets.
            try:
                auc = float(roc_auc_score(y_va_int, va_probs))
            except ValueError:
                auc = float("nan")

            fold_accs.append(acc)
            fold_aucs.append(auc)

        cv_rows.append(
            {
                "model": "pytorch_mlp",
                "hidden_dim": hp.hidden_dim,
                "learning_rate": hp.learning_rate,
                "batch_size": hp.batch_size,
                "epochs": hp.epochs,
                "cv_accuracy_mean": float(np.nanmean(fold_accs)),
                "cv_accuracy_std": float(np.nanstd(fold_accs, ddof=1)),
                "cv_roc_auc_mean": float(np.nanmean(fold_aucs)),
                "cv_roc_auc_std": float(np.nanstd(fold_aucs, ddof=1)),
            }
        )

    # 11) Select best params using mean ROC-AUC (required)
    cv_df = pd.DataFrame(cv_rows)
    cv_df = cv_df.sort_values(
        by=["cv_roc_auc_mean", "cv_accuracy_mean"],
        ascending=False,
    ).reset_index(drop=True)

    best_row = cv_df.iloc[0].to_dict()
    best_hp = HParams(
        hidden_dim=int(best_row["hidden_dim"]),
        learning_rate=float(best_row["learning_rate"]),
        batch_size=int(best_row["batch_size"]),
        epochs=int(best_row["epochs"]),
    )

    print("\n=== A) Cross-validation results ===")
    print(
        cv_df[
            [
                "model",
                "hidden_dim",
                "learning_rate",
                "batch_size",
                "epochs",
                "cv_accuracy_mean",
                "cv_accuracy_std",
                "cv_roc_auc_mean",
                "cv_roc_auc_std",
            ]
        ].to_string(index=False)
    )

    # 12) Retrain best model on full 80% training set (required)
    preprocessor_full, num_cols_full, cat_cols_full = build_preprocessor_from_train(
        X_train_full.copy()
    )
    X_train_for_fit = X_train_full.copy()
    X_test_for_eval = X_test.copy()
    enforce_column_dtypes(X_train_for_fit, X_test_for_eval, num_cols_full, cat_cols_full)

    X_train_np = preprocessor_full.fit_transform(X_train_for_fit)
    X_test_np = preprocessor_full.transform(X_test_for_eval)

    y_train_int = y_to_binary_int(y_train_full).astype(np.float32)
    y_test_int = y_to_binary_int(y_test).astype(np.float32)

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train_int, dtype=torch.float32, device=DEVICE).view(-1, 1)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32, device=DEVICE)

    # Train final model
    set_global_seeds(SEED)
    final_model = MLPBinaryClassifier(
        input_dim=int(X_train_t.shape[1]),
        hidden_dim=best_hp.hidden_dim,
        dropout=0.2,
    ).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_hp.learning_rate)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=best_hp.batch_size, shuffle=True)

    final_model.train()
    for _ in range(best_hp.epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            preds = final_model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # 13) Evaluate on untouched 20% holdout test set (required)
    final_model.eval()
    with torch.no_grad():
        test_probs = final_model(X_test_t).detach().cpu().numpy().reshape(-1)

    y_test_pred = (test_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test_int, y_test_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_int,
        y_test_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    auc = float(roc_auc_score(y_test_int, test_probs))
    cm = confusion_matrix(y_test_int, y_test_pred, labels=[0, 1])

    holdout_df = pd.DataFrame(
        [
            {
                "model": "pytorch_mlp",
                "accuracy": acc,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": auc,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
                "best_params": {
                    "hidden_dim": best_hp.hidden_dim,
                    "learning_rate": best_hp.learning_rate,
                    "batch_size": best_hp.batch_size,
                    "epochs": best_hp.epochs,
                    "dropout": 0.2,
                },
            }
        ]
    )

    print("\n=== B) Tuned holdout test results ===")
    print(
        holdout_df[
            ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp", "best_params"]
        ].to_string(index=False)
    )

    # 15) Save CSV outputs (required)
    cv_df.to_csv(CV_OUT, index=False)
    holdout_df.to_csv(HOLDOUT_OUT, index=False)

    print(f"\nSaved CV results to: {CV_OUT.resolve()}")
    print(f"Saved tuned holdout results to: {HOLDOUT_OUT.resolve()}")


if __name__ == "__main__":
    main()
    