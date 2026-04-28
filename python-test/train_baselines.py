from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from main import load_dataset, split_features_target, holdout_split
from train import DROP_COLS, infer_numeric_columns  # reuse your exact training rules


DATA_PATH = Path("data/malwareproj.csv")
MODELS_DIR = Path("models")

LR_PATH = MODELS_DIR / "logistic_regression.pkl"
DT_PATH = MODELS_DIR / "decision_tree.pkl"
TORCH_PATH = MODELS_DIR / "pytorch_mlp.pt"

# (optional) reuse one shared preprocessor for all models
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Same preprocessing as train.py; fit on train only to avoid leakage."""
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
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def prepare_data(df: pd.DataFrame):
    """
    Fair comparison setup:
    - same split (random_state=42)
    - same columns dropped as RF
    - same numeric/cat inference from X_train
    - fit preprocessor ONLY on X_train
    """
    X, y = split_features_target(df, target_col="Label")
    X_train, X_test, y_train, y_test = holdout_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.drop(columns=DROP_COLS, errors="ignore")
    X_test = X_test.drop(columns=DROP_COLS, errors="ignore")

    numeric_cols = infer_numeric_columns(X_train, min_numeric_rate=0.90)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    for c in numeric_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    for c in categorical_cols:
        X_train[c] = X_train[c].astype("string")
        X_test[c] = X_test[c].astype("string")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)  # fit on train only (no leakage)

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return preprocessor, X_train_proc, X_test_proc, y_train, y_test


def train_logistic_regression(X_train_proc, y_train) -> LogisticRegression:
    # Baseline LR (good for a fast, strong baseline)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(X_train_proc, y_train)
    return clf


def train_decision_tree(X_train_proc, y_train) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(max_depth=20, random_state=42)
    clf.fit(X_train_proc, y_train)
    return clf


def train_torch_mlp(X_train_proc, y_train, epochs: int = 10, lr: float = 1e-3):
    """
    Minimal PyTorch MLP baseline.
    Saves only weights; compare_models.py can later reload and evaluate.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_np = np.asarray(X_train_proc, dtype=np.float32)
    y_np = (pd.Series(y_train).astype("string") == "1").astype(np.float32).to_numpy().reshape(-1, 1)

    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

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

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    return model


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data using same loader logic as main.py
    df = load_dataset(DATA_PATH) if DATA_PATH.exists() else load_dataset(Path("malwareproj.csv"))

    preprocessor, X_train_proc, X_test_proc, y_train, y_test = prepare_data(df)

    # Save shared preprocessor (useful for future compare_models.py)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # Train + save sklearn baselines
    lr_model = train_logistic_regression(X_train_proc, y_train)
    joblib.dump(lr_model, LR_PATH)

    dt_model = train_decision_tree(X_train_proc, y_train)
    joblib.dump(dt_model, DT_PATH)

    # Train + save torch baseline
    torch_model = train_torch_mlp(X_train_proc, y_train, epochs=10, lr=1e-3)
    import torch
    torch.save(torch_model.state_dict(), TORCH_PATH)

    print("Saved:", LR_PATH)
    print("Saved:", DT_PATH)
    print("Saved:", TORCH_PATH)
    print("Saved preprocessor:", PREPROCESSOR_PATH)


if __name__ == "__main__":
    main()