from langchain_ollama import OllamaLLM


def build_analyst_prompt(schema: dict, stats_summary: str) -> str:
    """
    Constructs the prompt for the analyst agent.
    We pass actual computed statistics, not raw data.
    The LLM interprets numbers, it does not compute them.
    """
    return f"""
You are a senior data scientist reviewing an exploratory data analysis.

DATASET CONTEXT:
- Shape: {schema['shape'][0]} rows x {schema['shape'][1]} columns
- Numeric columns: {schema['numeric_columns']}
- Categorical columns: {schema['categorical_columns']}
- Missing value percentages: {schema['missing_percent']}

STATISTICAL SUMMARY:
{stats_summary}

Based on the above, provide a structured analysis covering:

1. KEY PATTERNS
   - Describe the distribution of each numeric column (normal, skewed, bimodal?)
   - Note any columns with unusually high or low variance
   - Identify dominant categories in categorical columns

2. DATA QUALITY ISSUES
   - Flag columns with significant missing data (>10%)
   - Identify potential outliers based on the min/max vs mean/std gap
   - Note any columns that may need type conversion

3. INTERESTING CORRELATIONS
   - Based on the column names and distributions, suggest which variables are likely correlated
   - Note any variables that appear redundant

4. MODELING RECOMMENDATIONS
   - Which columns should be dropped before training a model and why
   - What preprocessing steps are needed (scaling, encoding, imputation)
   - What type of ML problem does this dataset suggest (regression, classification, clustering)?

Be specific and reference actual column names and numbers from the statistics above.
Write in clear paragraphs, not bullet points. This text will appear in a PDF report.
"""


def run_analyst_agent(llm: OllamaLLM, df, schema: dict) -> str:
    """
    Runs the analyst agent: computes statistics and generates narrative insights.

    Args:
        llm: Initialized OllamaLLM instance
        df: Loaded pandas DataFrame
        schema: Output from schema_extractor

    Returns:
        insights (str): Narrative analysis text for the PDF report
    """
    print("[AnalystAgent] Computing statistical summary...")

    # Compute stats with pandas — LLM interprets, never computes
    stats_summary = df.describe(include="all").to_string()

    print("[AnalystAgent] Sending statistics to LLM for interpretation...")
    insights = llm.invoke(build_analyst_prompt(schema, stats_summary))

    print("[AnalystAgent] Analysis complete.")
    return insights.strip()


def build_basic_insights(df, schema: dict, missing_report: dict) -> str:
    """
    Deterministic narrative summary — no LLM call. Used by the fast default path
    so a detailed report can be produced in seconds on any dataset.
    """
    rows, cols = schema["shape"]
    numeric = [c for c in schema.get("numeric_columns", []) if c in df.columns]
    categorical = [c for c in schema.get("categorical_columns", []) if c in df.columns]

    lines = [
        f"The dataset has {rows} rows and {cols} columns "
        f"({len(numeric)} numeric, {len(categorical)} categorical)."
    ]

    if missing_report["affected_columns"]:
        worst_col, worst_info = max(
            missing_report["affected_columns"].items(),
            key=lambda kv: kv[1]["percent"],
        )
        lines.append(
            f"Overall {missing_report['total_missing_percent']}% of cells are missing; "
            f"the most affected column is {worst_col} ({worst_info['percent']}%)."
        )
    else:
        lines.append("No missing values were detected.")

    # Per-numeric-column shape (cap to keep the section readable).
    for col in numeric[:8]:
        s = df[col].dropna()
        if s.empty:
            continue
        skew = s.skew() if s.nunique() > 1 else 0.0
        shape = (
            "right-skewed" if skew > 1
            else "left-skewed" if skew < -1
            else "roughly symmetric"
        )
        lines.append(
            f"{col}: mean {s.mean():.2f}, median {s.median():.2f}, "
            f"std {s.std():.2f}, range [{s.min():.2f}, {s.max():.2f}] ({shape})."
        )

    # Strongest linear relationship among numeric columns.
    if len(numeric) >= 2:
        corr = df[numeric].corr().abs()
        best = (0.0, None, None)
        for i, a in enumerate(numeric):
            for b in numeric[i + 1:]:
                val = corr.loc[a, b]
                if val == val and val > best[0]:
                    best = (val, a, b)
        if best[1] is not None:
            lines.append(
                f"Strongest numeric correlation: {best[1]} and {best[2]} "
                f"(|r| = {best[0]:.2f})."
            )

    # Dominant categories.
    for col in categorical[:5]:
        vc = df[col].value_counts()
        if vc.empty:
            continue
        top = vc.index[0]
        share = round(vc.iloc[0] / vc.sum() * 100, 1)
        lines.append(f"{col}: most common value is '{top}' ({share}% of non-null rows).")

    return "\n".join(lines)