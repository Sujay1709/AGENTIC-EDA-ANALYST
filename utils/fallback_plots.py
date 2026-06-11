"""
Deterministic EDA plotting.

This is the primary, reliable plot generator for the pipeline: hard-coded,
known-good matplotlib/seaborn figures that work on any dataset without relying
on an LLM. (It also doubles as the fallback for the optional LLM coder agent.)

Produces, depending on what the data supports:
- distributions (histogram + KDE) and boxplots per numeric column
- a correlation heatmap and scatter plots for the most correlated numeric pairs
- trend lines of numeric columns over any detected datetime column
- value-count bars per categorical column
- a per-column missing-values bar chart

Plot filenames use stable prefixes (dist_, box_, scatter_, trend_,
value_count_, missing_bar) that output_manager relies on for captioning/ordering.
"""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Tunables
_TOP_CATEGORIES = 20        # cap categorical bars to the N most frequent values
_MAX_SCATTERS = 3           # how many top correlated pairs to scatter
_SCATTER_MIN_CORR = 0.3     # ignore weak correlations when picking scatters
_TREND_MAX_NUMERIC = 4      # cap numeric series drawn on a single trend chart
_TREND_RESAMPLE_OVER = 500  # resample to daily means above this many rows


def _slug(name: str) -> str:
    """Make a column name safe for use inside a filename."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_") or "col"


def generate_fallback_plots(df, schema: dict, output_dir: str) -> list:
    """
    Generate the EDA plot set deterministically.

    Returns an ordered list of saved plot file paths. Every figure is guarded
    against empty/insufficient data so this never raises on an odd dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    saved_plots = []
    numeric_cols = [c for c in schema.get("numeric_columns", []) if c in df.columns]
    cat_cols = [c for c in schema.get("categorical_columns", []) if c in df.columns]
    datetime_cols = [c for c in schema.get("datetime_columns", []) if c in df.columns]

    def _save(name):
        path = os.path.join(output_dir, name)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        saved_plots.append(path)

    # --- Distributions: histogram with KDE overlay -------------------------
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        plt.figure(figsize=(8, 5))
        # KDE needs variance; skip the curve for constant columns.
        sns.histplot(series, bins=30, kde=series.nunique() > 1, edgecolor="black")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        _save(f"dist_{_slug(col)}.png")

    # --- Boxplots: outliers and spread per numeric column ------------------
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=series, color="#69b3a2")
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        _save(f"box_{_slug(col)}.png")

    # --- Correlation heatmap -----------------------------------------------
    corr = None
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap")
        _save("correlation_heatmap.png")

    # --- Scatter plots for the most correlated numeric pairs ---------------
    if corr is not None:
        pairs = []
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1:]:
                val = corr.loc[a, b]
                if val == val and abs(val) >= _SCATTER_MIN_CORR:  # val==val drops NaN
                    pairs.append((abs(val), val, a, b))
        pairs.sort(reverse=True)
        for _, val, a, b in pairs[:_MAX_SCATTERS]:
            sub = df[[a, b]].dropna()
            if sub.empty:
                continue
            plt.figure(figsize=(7, 5))
            sns.scatterplot(x=sub[a], y=sub[b], alpha=0.6)
            plt.title(f"{a} vs {b} (r = {val:.2f})")
            plt.xlabel(a)
            plt.ylabel(b)
            _save(f"scatter_{_slug(a)}_vs_{_slug(b)}.png")

    # --- Trend lines of numeric columns over each datetime column ----------
    for date_col in datetime_cols:
        trend_numerics = numeric_cols[:_TREND_MAX_NUMERIC]
        if not trend_numerics:
            break
        ts = df[[date_col] + trend_numerics].dropna(subset=[date_col])
        ts = ts.sort_values(date_col)
        if ts.empty:
            continue
        if len(ts) > _TREND_RESAMPLE_OVER:
            ts = ts.set_index(date_col).resample("D").mean(numeric_only=True).dropna(how="all")
            x = ts.index
        else:
            ts = ts.set_index(date_col)
            x = ts.index
        plt.figure(figsize=(10, 5))
        plotted = False
        for col in trend_numerics:
            if col in ts.columns and ts[col].notna().any():
                plt.plot(x, ts[col], label=col)
                plotted = True
        if not plotted:
            plt.close()
            continue
        plt.title(f"Trend over {date_col}")
        plt.xlabel(date_col)
        plt.ylabel("Value")
        plt.legend(loc="best", fontsize=8)
        plt.xticks(rotation=30, ha="right")
        _save(f"trend_over_{_slug(date_col)}.png")

    # --- Value-count bars per categorical column ---------------------------
    for col in cat_cols:
        vc = df[col].value_counts()
        if vc.empty:
            continue
        counts = vc.head(_TOP_CATEGORIES)
        total = int(vc.shape[0])
        high_card = total > _TOP_CATEGORIES
        plt.figure(figsize=(8, 5))
        if high_card or counts.index.astype(str).str.len().max() > 12:
            # Horizontal bars read better for many / long category labels.
            counts.sort_values().plot(kind="barh", color="#4c72b0")
            plt.xlabel("Count")
            plt.ylabel(col)
        else:
            counts.plot(kind="bar", color="#4c72b0")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=30, ha="right")
        suffix = f" (top {_TOP_CATEGORIES} of {total})" if high_card else ""
        plt.title(f"Value Counts of {col}{suffix}")
        _save(f"value_count_{_slug(col)}.png")

    # --- Per-column missing-values bar chart -------------------------------
    miss_pct = (df.isnull().mean() * 100).round(2)
    miss_pct = miss_pct[miss_pct > 0].sort_values(ascending=True)
    if not miss_pct.empty:
        plt.figure(figsize=(8, max(3, 0.4 * len(miss_pct) + 1)))
        miss_pct.plot(kind="barh", color="#c44e52")
        plt.title("Missing Values by Column")
        plt.xlabel("% missing")
        plt.ylabel("Column")
        _save("missing_bar.png")

    return saved_plots
