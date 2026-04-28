

from __future__ import annotations
"""
cross_validate_models.py
Statistically correct model selection + evaluation workflow:
- Step 1: Create an untouched 20% holdout test set (stratified, random_state=42)
- Step 2: Use only the 80% training portion for model selection
- Step 3: Within training, run Stratified 10-Fold Cross-Validation + hyperparameter tuning
- Step 4: Retrain best model on full training set
- Step 5: Evaluate final tuned model on the untouched holdout test set
This script reuses:
- main.py: load_dataset, split_features_target, holdout_split
- train.py: DROP_COLS, infer_numeric_columns
and avoids leakage by fitting preprocessing inside each CV fold via sklearn Pipeline.
"""
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Project reuse
from main import load_dataset, split_features_target, holdout_split
from train import DROP_COLS, infer_numeric_columns

# -----------------------------
# Paths / outputs
# -----------------------------
DATA_PATH = Path("data/malwareproj.csv")
FALLBACK_DATA_PATH = Path("malwareproj.csv")
CV_OUT = Path("cross_validation_results.csv")
HOLDOUT_OUT = Path("tuned_holdout_results.csv")

# -----------------------------
# Utilities
# -----------------------------
def pick_dataset_path() -> Path:
    """Prefer data/ location; fallback to root if still present."""
    return DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH

def build_preprocessor_from_train(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build the ColumnTransformer based on TRAIN columns only.
    IMPORTANT: We infer numeric/categorical columns from X_train only to avoid peeking at test.
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

def make_sklearn_pipeline(preprocessor: ColumnTransformer, estimator: BaseEstimator) -> Pipeline:
    """
    Leakage-safe pipeline: preprocessing + estimator.
    During CV, sklearn will fit the preprocessor on each training fold only.
    """
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

def y_to_binary_int(y: pd.Series) -> np.ndarray:
    """Convert Label series ('0'/'1') to ints 0/1 for ROC-AUC."""
    y_clean = y.astype("string").str.strip()
    return (y_clean == "1").astype(int).to_numpy()

def get_model_grids() -> dict[str, tuple[BaseEstimator, dict]]:
    """
    Models + hyperparameter grids for GridSearchCV.
    Notes:
    - We tune modest grids to keep runtime reasonable.
    - Add xgboost/lightgbm/catboost later if installed (this script is ready for extension).
    """
    grids: dict[str, tuple[BaseEstimator, dict]] = {}
    grids["random_forest"] = (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5],
        },
    )
    grids["logistic_regression"] = (
        LogisticRegression(max_iter=3000, solver="lbfgs"),
        {
            "model__C": [0.1, 1.0, 10.0],
        },
    )
    grids["decision_tree"] = (
        DecisionTreeClassifier(random_state=42),
        {
            "model__max_depth": [5, 10, 20, None],
            "model__min_samples_split": [2, 5, 10],
        },
    )
    return grids

# -----------------------------
# Main workflow
# -----------------------------
def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    # 1) Load dataset using project logic
    df = load_dataset(pick_dataset_path())
    # 2) Split into X/y and create untouched holdout test set (80/20, stratified, rs=42)
    X, y = split_features_target(df, target_col="Label")
    # Drop columns (same rule as training) BEFORE splitting is fine (dropping doesn't learn stats),
    # but we keep the exact workflow: split first then drop, matching your training approach.
    X_train, X_test, y_train, y_test = holdout_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.drop(columns=DROP_COLS, errors="ignore")
    X_test = X_test.drop(columns=DROP_COLS, errors="ignore")
    # 3) Build preprocessor definition based on training columns only
    preprocessor, numeric_cols, categorical_cols = build_preprocessor_from_train(X_train)
    # We do NOT manually fit/transform here for CV. The pipeline handles it per-fold.
    # Still, we ensure the raw DataFrame has consistent types similar to your training expectations:
    # - numeric columns become numeric
    # - categorical columns become string
    for c in numeric_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
    for c in categorical_cols:
        X_train[c] = X_train[c].astype("string")
        X_test[c] = X_test[c].astype("string")
    # 4) Stratified 10-fold CV on training set only (model selection)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # CV metrics to optimize/record
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }
    model_grids = get_model_grids()
    cv_rows: list[dict] = []
    holdout_rows: list[dict] = []
    # We need y in a format roc_auc understands; for sklearn scoring 'roc_auc' it expects
    # classes 0/1 OR labels that can be binarized. We'll provide y_train as '0'/'1' strings;
    # roc_auc scoring uses predict_proba/decision_function internally.
    y_train_clean = y_train.astype("string").str.strip()
    y_test_clean = y_test.astype("string").str.strip()
    for model_name, (estimator, param_grid) in model_grids.items():
        print(f"\n=== Tuning: {model_name} ===")
        pipe = make_sklearn_pipeline(preprocessor, estimator)
        # 5-6) GridSearchCV for hyperparameter tuning inside CV on training set only
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            refit="roc_auc",          # choose best params by ROC-AUC (common for malware detection)
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        gs.fit(X_train, y_train_clean)
        # Record CV results for the best model
        best_idx = int(gs.best_index_)
        mean_acc = float(gs.cv_results_["mean_test_accuracy"][best_idx])
        std_acc = float(gs.cv_results_["std_test_accuracy"][best_idx])
        mean_auc = float(gs.cv_results_["mean_test_roc_auc"][best_idx])
        std_auc = float(gs.cv_results_["std_test_roc_auc"][best_idx])
        cv_rows.append(
            {
                "model": model_name,
                "best_params": gs.best_params_,
                "cv_accuracy_mean": mean_acc,
                "cv_accuracy_std": std_acc,
                "cv_roc_auc_mean": mean_auc,
                "cv_roc_auc_std": std_auc,
            }
        )
        # 7) Retrain best model on full training set (GridSearchCV already refit on full train)
        best_model = gs.best_estimator_
        # 8) Evaluate tuned model on untouched holdout test set
        y_pred = best_model.predict(X_test)
        # For ROC-AUC: use predict_proba if available, else decision_function
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)[:, 1]
        else:
            y_score = best_model.decision_function(X_test)
        acc = float(accuracy_score(y_test_clean, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_clean,
            y_pred,
            average="binary",
            pos_label="1",
            zero_division=0,
        )
        auc = float(roc_auc_score(y_to_binary_int(y_test_clean), y_score))
        cm = confusion_matrix(y_test_clean, y_pred, labels=["0", "1"])
        holdout_rows.append(
            {
                "model": model_name,
                "accuracy": acc,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": auc,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
                "best_params": gs.best_params_,
            }
        )
    # 9A) Print CV results ranked by CV ROC-AUC (primary) then accuracy
    cv_df = pd.DataFrame(cv_rows)
    cv_df = cv_df.sort_values(by=["cv_roc_auc_mean", "cv_accuracy_mean"], ascending=False).reset_index(drop=True)
    print("\n=== A) Cross-validation results (ranked by CV ROC-AUC mean) ===")
    print(
        cv_df[
            ["model", "cv_accuracy_mean", "cv_accuracy_std", "cv_roc_auc_mean", "cv_roc_auc_std", "best_params"]
        ].to_string(index=False)
    )
    # 9B) Print holdout results ranked by holdout ROC-AUC then F1
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df = holdout_df.sort_values(by=["roc_auc", "f1"], ascending=False).reset_index(drop=True)
    print("\n=== B) Tuned holdout test results (ranked by holdout ROC-AUC) ===")
    print(
        holdout_df[
            ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp", "best_params"]
        ].to_string(index=False)
    )
    # 10) Save outputs to CSV
    cv_df.to_csv(CV_OUT, index=False)
    holdout_df.to_csv(HOLDOUT_OUT, index=False)
    print(f"\nSaved CV results to: {CV_OUT.resolve()}")
    print(f"Saved tuned holdout results to: {HOLDOUT_OUT.resolve()}")
    # 11) Optional: save tuned best estimators for reuse later (does not affect existing scripts)
    # If you want, uncomment:
    # Path("models/tuned").mkdir(parents=True, exist_ok=True)
    # for row in holdout_rows:
    #     pass

if __name__ == "__main__":
    main()
