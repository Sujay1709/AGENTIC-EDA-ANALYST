"""
Deep missing-values analysis.

Goes beyond the raw null counts in the schema: per-column stats, dataset-wide
totals, fully-complete columns, heavily-missing columns, and co-missingness
(columns that tend to be null together). Also renders a deterministic
missing-values heatmap so the report always has one, independent of the LLM.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_missing(df) -> dict:
    """Return a structured missing-values report for `df`."""
    n_rows = len(df)
    total_cells = int(df.size)
    missing_per_col = df.isnull().sum()
    total_missing = int(missing_per_col.sum())

    per_column = {}
    for col in df.columns:
        count = int(missing_per_col[col])
        per_column[col] = {
            "count": count,
            "percent": round((count / n_rows * 100), 2) if n_rows else 0.0,
            "dtype": str(df[col].dtype),
        }

    affected = {c: v for c, v in per_column.items() if v["count"] > 0}
    complete_columns = [c for c, v in per_column.items() if v["count"] == 0]
    high_missing_columns = [c for c, v in per_column.items() if v["percent"] > 50]

    # Co-missingness: correlate the null-masks of affected columns.
    co_missing_pairs = []
    missing_cols = list(affected.keys())
    if len(missing_cols) >= 2:
        null_corr = df[missing_cols].isnull().astype(int).corr()
        for i, a in enumerate(missing_cols):
            for b in missing_cols[i + 1:]:
                val = null_corr.loc[a, b]
                if val == val and abs(val) >= 0.5:  # `val == val` filters out NaN
                    co_missing_pairs.append(
                        {"columns": [a, b], "correlation": round(float(val), 2)}
                    )
        co_missing_pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        co_missing_pairs = co_missing_pairs[:10]

    return {
        "n_rows": n_rows,
        "total_cells": total_cells,
        "total_missing": total_missing,
        "total_missing_percent": (
            round((total_missing / total_cells * 100), 2) if total_cells else 0.0
        ),
        "per_column": per_column,
        "affected_columns": affected,
        "complete_columns": complete_columns,
        "high_missing_columns": high_missing_columns,
        "co_missing_pairs": co_missing_pairs,
    }


def save_missing_heatmap(df, output_dir: str):
    """
    Save a missing-values heatmap to output_dir/missing_values.png.
    Returns the path, or None if the dataset has no missing values.
    """
    if not df.isnull().values.any():
        return None
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Missing Values Map")
    path = os.path.join(output_dir, "missing_values.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path
