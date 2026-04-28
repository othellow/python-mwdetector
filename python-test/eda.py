"""
eda.py

Exploratory Data Analysis (EDA) for malwareproj.csv.

- Reuses the same dataset loading logic from main.py (load_dataset)
- Prints key dataset diagnostics
- Plots histograms for top numeric features
- Plots correlation heatmap for numeric features
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Reuse your existing loader (keeps behavior consistent with main.py)
from main import load_dataset

CSV_PATH = Path("malwareproj.csv")
TARGET_COL = "Label"


def coerce_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to numeric where possible (without crashing).
    This is helpful because main.py loads dtype=str for consistency.
    """
    X_num = X.copy()
    for col in X_num.columns:
        # Convert anything that looks numeric; non-numeric becomes NaN
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
    return X_num


def print_basic_info(df: pd.DataFrame) -> None:
    """1-3) Shape, columns, dtypes."""
    print("=== 1) Dataset shape ===")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n=== 2) Column names ===")
    print(df.columns.tolist())

    print("\n=== 3) Data types (pandas dtypes) ===")
    print(df.dtypes)


def print_missing_values(df: pd.DataFrame) -> None:
    """5) Missing/null values per column."""
    print("\n=== 5) Missing/null values per column ===")
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)

    missing_table = pd.DataFrame(
        {"missing_count": missing, "missing_pct": missing_pct.round(2)}
    )
    print(missing_table)


def print_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    4) Summary statistics for numeric columns.

    Returns the numeric-only DataFrame used for later plots/correlation.
    """
    print("\n=== 4) Summary statistics for numeric columns ===")

    # Exclude target for feature-only numeric analysis
    feature_df = df.drop(columns=[TARGET_COL], errors="ignore")

    # Convert "numeric-looking" strings to numeric values
    numeric_candidate = coerce_numeric_features(feature_df)

    # Keep only columns that have at least one numeric value
    numeric_cols = [c for c in numeric_candidate.columns if numeric_candidate[c].notna().any()]
    numeric_df = numeric_candidate[numeric_cols]

    if numeric_df.shape[1] == 0:
        print("No numeric columns detected after coercion.")
        return numeric_df

    # describe() for numeric summary
    print(numeric_df.describe().T)
    return numeric_df


def print_label_distribution(df: pd.DataFrame) -> pd.Series:
    """6-7) Label distribution + imbalance check."""
    print("\n=== 6) Target Label distribution (counts + %) ===")

    if TARGET_COL not in df.columns:
        print(f"Target column '{TARGET_COL}' not found.")
        return pd.Series(dtype="int64")

    # Keep Label consistent as strings '0'/'1'
    y = df[TARGET_COL].astype("string").str.strip()

    counts = y.value_counts(dropna=False)
    pct = (counts / len(y) * 100).round(2)

    dist = pd.DataFrame({"count": counts, "pct": pct})
    print(dist)

    print("\n=== 7) Class balance check ===")
    # Simple imbalance heuristic: minority share < 40% (adjust as you prefer)
    if set(y.dropna().unique().tolist()) >= {"0", "1"}:
        c0 = int((y == "0").sum())
        c1 = int((y == "1").sum())
        total = c0 + c1
        if total > 0:
            minority = min(c0, c1)
            minority_pct = minority / total * 100
            is_imbalanced = minority_pct < 40
            print(f"Minority class % (of non-null 0/1 labels): {minority_pct:.2f}%")
            print("Imbalanced?", is_imbalanced)
        else:
            print("No valid '0'/'1' labels found to assess balance.")
    else:
        print("Label values are not strictly {'0','1'}; cannot apply binary imbalance check.")

    return counts


def plot_histograms(numeric_df: pd.DataFrame, top_n: int = 12) -> None:
    """8) Histograms for top numeric features."""
    print("\n=== 8) Histograms for top numeric features ===")
    if numeric_df.shape[1] == 0:
        print("Skipping histograms: no numeric columns detected.")
        return

    # Pick "top numeric features" by variance (ignoring all-NaN columns)
    variances = numeric_df.var(numeric_only=True).sort_values(ascending=False)
    top_cols = variances.head(top_n).index.tolist()

    print(f"Plotting histograms for: {top_cols}")

    sns.set_style("whitegrid")
    n = len(top_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(top_cols):
        series = numeric_df[col].dropna()
        axes[i].hist(series, bins=30)
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(numeric_df: pd.DataFrame, max_features: int = 25) -> None:
    """9) Correlation heatmap (numeric features only)."""
    print("\n=== 9) Correlation heatmap (numeric features) ===")
    if numeric_df.shape[1] < 2:
        print("Skipping correlation heatmap: need at least 2 numeric columns.")
        return

    # Limit to top features to keep the heatmap readable
    variances = numeric_df.var(numeric_only=True).sort_values(ascending=False)
    cols = variances.head(max_features).index.tolist()
    corr = numeric_df[cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title(f"Correlation heatmap (top {len(cols)} numeric features by variance)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Load dataset using the same logic as main.py
    df = load_dataset(CSV_PATH)

    # 1-3) Basic structure info
    print_basic_info(df)

    # 4) Numeric summary stats
    numeric_df = print_numeric_summary(df)

    # 5) Missing values
    print_missing_values(df)

    # 6-7) Label distribution and imbalance check
    print_label_distribution(df)

    # 8) Histograms
    plot_histograms(numeric_df, top_n=12)

    # 9) Correlation heatmap
    plot_correlation_heatmap(numeric_df, max_features=25)


if __name__ == "__main__":
    main()