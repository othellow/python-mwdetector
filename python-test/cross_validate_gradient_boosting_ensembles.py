from __future__ import annotations

"""
cross_validate_gradient_boosting_ensembles.py

Statistically correct model selection + evaluation workflow (mirrors cross_validate_models.py):
- Step 1: Create an untouched 20% holdout test set (stratified, random_state=42)
- Step 2: Use only the 80% training portion for model selection
- Step 3: Within training, run Stratified 10-Fold Cross-Validation + hyperparameter tuning
- Step 4: Retrain best model on full training set
- Step 5: Evaluate final tuned model on the untouched holdout test set

Models compared (tree-based gradient boosting ensembles):
- xgboost   (XGBClassifier)
- lightgbm  (LGBMClassifier)
- catboost  (CatBoostClassifier)
"""

from pathlib import Path
import warnings
import joblib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Project reuse
from main import load_dataset, split_features_target, holdout_split
from train import DROP_COLS, infer_numeric_columns


# -----------------------------
# Paths / outputs
# -----------------------------
DATA_PATH = Path("data/malwareproj.csv")
FALLBACK_DATA_PATH = Path("malwareproj.csv")

CV_OUT = Path("gradient_boosting_cv_results.csv")
HOLDOUT_OUT = Path("gradient_boosting_holdout_results.csv")
DATA_DIR = Path("data")
HOLDOUT_X_OUT = DATA_DIR / "holdout_X_test.csv"
HOLDOUT_Y_OUT = DATA_DIR / "holdout_y_test.csv"


# -----------------------------
# Utilities
# -----------------------------
def pick_dataset_path() -> Path:
    """Prefer data/ location; fallback to root if still present."""
    return DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH


def build_onehot_preprocessor_from_train(
    X_train: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Preprocessor for XGBoost & LightGBM:
    - numeric: median imputer
    - categorical: most_frequent imputer + OneHotEncoder
    Built from TRAIN columns only (avoid peeking).
    """
    numeric_cols = infer_numeric_columns(X_train, min_numeric_rate=0.90)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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


def build_catboost_preprocessor_from_train(
    X_train: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Preprocessor for CatBoost native categorical handling:
    - numeric: median imputer
    - categorical: most_frequent imputer (NO one-hot)
    We keep categoricals as strings and pass cat feature indices to CatBoost.
    Built from TRAIN columns only (avoid peeking).
    """
    numeric_cols = infer_numeric_columns(X_train, min_numeric_rate=0.90)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Ensure we preserve a pandas DataFrame output so CatBoost sees string categoricals.
    preprocessor.set_output(transform="pandas")
    return preprocessor, numeric_cols, categorical_cols


def make_sklearn_pipeline(preprocessor: ColumnTransformer, estimator: BaseEstimator) -> Pipeline:
    """
    Leakage-safe pipeline: preprocessing + estimator.
    During CV, sklearn will fit the preprocessor on each training fold only.
    """
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def _import_gb_ensembles() -> tuple[type, type, type, list[str]]:
    missing: list[str] = []

    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception:
        XGBClassifier = None  # type: ignore
        missing.append("xgboost")

    try:
        from lightgbm import LGBMClassifier  # type: ignore
    except Exception:
        LGBMClassifier = None  # type: ignore
        missing.append("lightgbm")

    try:
        from catboost import CatBoostClassifier  # type: ignore
    except Exception:
        CatBoostClassifier = None  # type: ignore
        missing.append("catboost")

    if missing:
        pkgs = " ".join(missing)
        raise RuntimeError(
            "Missing required packages: "
            + ", ".join(missing)
            + "\nInstall with:\n"
            + f"pip install {pkgs}"
        )

    return XGBClassifier, LGBMClassifier, CatBoostClassifier, missing


def get_model_grids(
    categorical_cols_for_catboost: list[str],
) -> dict[str, tuple[BaseEstimator, dict]]:
    """
    Models + lightweight hyperparameter grids for GridSearchCV.
    Best params are selected by mean ROC-AUC (refit='roc_auc').

    Labels MUST be exactly: xgboost, lightgbm, catboost
    """
    XGBClassifier, LGBMClassifier, CatBoostClassifier, _ = _import_gb_ensembles()

    grids: dict[str, tuple[BaseEstimator, dict]] = {}

    grids["xgboost"] = (
        XGBClassifier(
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="auc",
        ),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
        },
    )

    grids["lightgbm"] = (
        LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            device_type="cpu",
            verbosity=-1,
        ),
        {
            "model__n_estimators": [100, 200],
            "model__num_leaves": [31, 63],
            "model__learning_rate": [0.05, 0.1],
        },
    )

    # CatBoost: native categorical handling via cat_features; we keep preprocessing as imputers only.
    # NOTE: Using column names here works well with pandas inputs inside the estimator.
    grids["catboost"] = (
        CatBoostClassifier(
            random_seed=42,
            task_type="CPU",
            verbose=False,
        ),
        {
            "model__iterations": [100, 200],
            "model__depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
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
    X_train, X_test, y_train, y_test = holdout_split(X, y, test_size=0.2, random_state=42)

    # Drop columns AFTER split (dropping doesn't learn stats; mirrors existing workflow)
    X_train = X_train.drop(columns=DROP_COLS, errors="ignore")
    X_test = X_test.drop(columns=DROP_COLS, errors="ignore")

    # Clean y labels
    y_train_clean = y_train.astype("string").str.strip()
    y_test_clean = y_test.astype("string").str.strip()

    # Convert to numeric labels immediately (XGBoost requires numeric classes)
    y_train_num = (y_train_clean == "1").astype(int)
    y_test_num = (y_test_clean == "1").astype(int)

    # Preprocessors built from TRAIN ONLY
    onehot_preprocessor, onehot_numeric_cols, onehot_categorical_cols = build_onehot_preprocessor_from_train(
        X_train
    )
    cat_preprocessor, cat_numeric_cols, cat_categorical_cols = build_catboost_preprocessor_from_train(X_train)

    # Keep raw DataFrames with consistent types (based on TRAIN-derived column lists)
    for c in set(onehot_numeric_cols).union(set(cat_numeric_cols)):
        if c in X_train.columns:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        if c in X_test.columns:
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    for c in set(onehot_categorical_cols).union(set(cat_categorical_cols)):
        if c in X_train.columns:
            X_train[c] = X_train[c].astype("string")
        if c in X_test.columns:
            X_test[c] = X_test[c].astype("string")

    # Persist the untouched 20% holdout split for reproducible downstream checks.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_test.to_csv(HOLDOUT_X_OUT, index=False)
    y_test_clean.to_frame(name="Label").to_csv(HOLDOUT_Y_OUT, index=False)
    print("Saved: data/holdout_X_test.csv")
    print("Saved: data/holdout_y_test.csv")

    # 4) Stratified 10-fold CV on training set only (model selection)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }

    # Import models (and raise with exact pip install command if missing)
    model_grids = get_model_grids(categorical_cols_for_catboost=cat_categorical_cols)

    cv_rows: list[dict] = []
    holdout_rows: list[dict] = []
    best_lightgbm_params: dict | None = None

    for model_name, (estimator, param_grid) in model_grids.items():
        if model_name in {"xgboost", "lightgbm"}:
            preprocessor = onehot_preprocessor
        else:
            preprocessor = cat_preprocessor

        pipe = make_sklearn_pipeline(preprocessor, estimator)

        # 5-6) GridSearchCV for hyperparameter tuning inside CV on training set only
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        if model_name == "catboost":
            gs.fit(
                X_train,
                y_train_num,
                model__cat_features=cat_categorical_cols,
            )
        else:
            gs.fit(X_train, y_train_num)

        best_idx = int(gs.best_index_)
        if model_name == "lightgbm":
            best_lightgbm_params = dict(gs.best_params_)
        cv_rows.append(
            {
                "model": model_name,
                "best_params": gs.best_params_,
                "cv_accuracy_mean": float(gs.cv_results_["mean_test_accuracy"][best_idx]),
                "cv_accuracy_std": float(gs.cv_results_["std_test_accuracy"][best_idx]),
                "cv_roc_auc_mean": float(gs.cv_results_["mean_test_roc_auc"][best_idx]),
                "cv_roc_auc_std": float(gs.cv_results_["std_test_roc_auc"][best_idx]),
            }
        )

        # 7) Retrain best model on full training set (GridSearchCV already refit on full train)
        best_model = gs.best_estimator_

        # 8) Evaluate tuned model on untouched holdout test set
        y_pred = best_model.predict(X_test)
        y_pred = pd.Series(y_pred).astype(int).to_numpy()
        y_score = best_model.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test_num, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_num,
            y_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        auc = float(roc_auc_score(y_test_num, y_score))
        cm = confusion_matrix(y_test_num, y_pred, labels=[0, 1])

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

    cv_df = pd.DataFrame(cv_rows).sort_values(
        by=["cv_roc_auc_mean", "cv_accuracy_mean"], ascending=False
    ).reset_index(drop=True)

    holdout_df = pd.DataFrame(holdout_rows).sort_values(
        by=["roc_auc", "accuracy"], ascending=False
    ).reset_index(drop=True)

    print("\n=== A) Cross-validation results ===")
    print(
        cv_df[
            ["model", "cv_accuracy_mean", "cv_accuracy_std", "cv_roc_auc_mean", "cv_roc_auc_std", "best_params"]
        ].to_string(index=False)
    )

    print("\n=== B) Tuned holdout test results ===")
    print(
        holdout_df[
            ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp", "best_params"]
        ].to_string(index=False)
    )

    cv_df.to_csv(CV_OUT, index=False)
    holdout_df.to_csv(HOLDOUT_OUT, index=False)

    # Export production-ready LightGBM artifacts using the best CV parameters.
    if best_lightgbm_params is None:
        raise RuntimeError("LightGBM best parameters were not found during CV.")

    # Rebuild final LightGBM estimator from selected CV hyperparameters.
    _, LGBMClassifier, _, _ = _import_gb_ensembles()
    final_lightgbm_params = {
        k.replace("model__", "", 1): v for k, v in best_lightgbm_params.items()
    }
    final_lightgbm = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        device_type="cpu",
        verbosity=-1,
        **final_lightgbm_params,
    )

    # Fit preprocessing on full training data, transform it, then train final model.
    final_preprocessor, _, _ = build_onehot_preprocessor_from_train(X_train)
    X_train_transformed = final_preprocessor.fit_transform(X_train)
    final_lightgbm.fit(X_train_transformed, y_train_num)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_path = DATA_DIR / "LightGBMmodel.pkl"
    preprocessor_path = DATA_DIR / "LightGBM_preprocessor.pkl"
    feature_columns_path = DATA_DIR / "feature_columns.pkl"

    joblib.dump(final_lightgbm, model_path)
    joblib.dump(final_preprocessor, preprocessor_path)
    joblib.dump(list(X_train.columns), feature_columns_path)

    print("Saved: data/LightGBMmodel.pkl")
    print("Saved: data/LightGBM_preprocessor.pkl")
    print("Saved: data/feature_columns.pkl")


if __name__ == "__main__":
    main()