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


def suggest_missing_strategies(df, schema: dict, missing_report: dict) -> str:
    """
    Deterministic, rule-based recommendations for handling missing values, with
    concrete pandas code per affected column. No LLM call — instant and reliable
    on any dataset.
    """
    affected = missing_report["affected_columns"]
    if not affected:
        return "No missing values detected - no imputation needed."

    numeric = set(schema.get("numeric_columns", []))
    categorical = set(schema.get("categorical_columns", []))
    datetime_cols = set(schema.get("datetime_columns", []))

    blocks = []
    for col, info in affected.items():
        pct = info["percent"]
        if pct > 50:
            strategy = (
                f"{pct}% missing - too sparse to impute reliably; drop the column "
                "(optionally keep a 'was_missing' flag first)."
            )
            code = f"df = df.drop(columns=['{col}'])"
        elif col in numeric:
            strategy = f"Numeric with {pct}% missing - impute with the median (robust to outliers)."
            code = f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())"
        elif col in datetime_cols:
            strategy = f"Datetime with {pct}% missing - fill in time order (forward then back)."
            code = f"df['{col}'] = df['{col}'].ffill().bfill()"
        elif col in categorical:
            strategy = (
                f"Categorical with {pct}% missing - impute the most frequent value "
                "or add an explicit 'Unknown' category."
            )
            code = f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode().iloc[0])"
        else:
            strategy = f"{pct}% missing - fill with a sensible default or drop the affected rows."
            code = f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode().iloc[0])"
        blocks.append(f"COLUMN: {col}\nSTRATEGY: {strategy}\nCODE: {code}")

    header = (
        "Rule-based recommendations to reduce missing values. Apply per column, "
        "then re-check with df.isnull().sum():\n"
    )
    tips = (
        "\nGeneral tips to reduce missingness: fix upstream data-collection gaps, "
        "join reference data to backfill, and prefer model-based imputation "
        "(e.g. sklearn IterativeImputer) when columns are correlated."
    )
    return header + "\n\n".join(blocks) + "\n" + tips


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
