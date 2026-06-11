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