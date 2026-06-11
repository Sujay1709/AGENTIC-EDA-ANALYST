"""
Deterministic EDA plotting fallback.

The coder agent relies on an LLM to generate visualization code, which is not
reliable with weaker local models — it can exhaust all retries. This module
provides hard-coded, known-good plotting so the pipeline always produces output
instead of crashing.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def generate_fallback_plots(df, schema: dict, output_dir: str) -> list:
    """
    Generate the standard EDA plot set deterministically.

    Returns a list of saved plot file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []
    numeric_cols = [c for c in schema["numeric_columns"] if c in df.columns]
    cat_cols = [c for c in schema["categorical_columns"] if c in df.columns]

    def _save(name):
        path = os.path.join(output_dir, name)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        saved_plots.append(path)

    # Distribution histogram per numeric column (NaNs are dropped by hist)
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        plt.figure()
        plt.hist(series, bins=30, edgecolor="black")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        _save(f"dist_{col}.png")

    # Correlation heatmap across numeric columns
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap")
        _save("correlation_heatmap.png")

    # Value-count bar chart per categorical column (top 20 categories)
    for col in cat_cols:
        counts = df[col].value_counts().head(20)
        if counts.empty:
            continue
        plt.figure()
        counts.plot(kind="bar")
        plt.title(f"Value Counts of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        _save(f"value_count_{col}.png")

    # Missing-values heatmap when any data is missing
    if df.isnull().values.any():
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
        plt.title("Missing Values")
        _save("missing_values.png")

    return saved_plots
