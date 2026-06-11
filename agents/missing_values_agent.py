"""
Missing-values agent.

Asks the LLM to recommend a concrete handling strategy for each column that has
missing data, together with the exact pandas code to apply it. Text-only — it
never executes code, so it cannot crash the pipeline.
"""
from langchain_ollama import OllamaLLM


def build_missing_values_prompt(schema: dict, missing_report: dict) -> str:
    affected = missing_report["affected_columns"]
    affected_lines = "\n".join(
        f"- {col} ({info['dtype']}): {info['count']} missing ({info['percent']}%)"
        for col, info in affected.items()
    ) or "None"

    return f"""
You are a senior data scientist advising on how to handle missing data.

DATASET: {schema['shape'][0]} rows x {schema['shape'][1]} columns
Numeric columns: {schema['numeric_columns']}
Categorical columns: {schema['categorical_columns']}

COLUMNS WITH MISSING VALUES:
{affected_lines}

Total missing cells: {missing_report['total_missing']} ({missing_report['total_missing_percent']}% of all cells)
Columns more than 50% missing: {missing_report['high_missing_columns']}

For EACH column with missing values, recommend ONE concrete strategy (median/mean
imputation, mode imputation, drop the column, drop the rows, or flag-and-fill) and
explain WHY in a single sentence based on its dtype and missing percentage. Then give
the exact one-line pandas code to apply it, assuming a DataFrame named df.

Format each column EXACTLY as:
COLUMN: <name>
STRATEGY: <one sentence>
CODE: <one line of pandas code>

Be concise. Use only the column names listed above. Plain ASCII only.
"""


def run_missing_values_agent(llm: OllamaLLM, schema: dict, missing_report: dict) -> str:
    """Return per-column missing-data recommendations as report-ready text."""
    if not missing_report["affected_columns"]:
        return "No missing values detected - no imputation needed."

    print("[MissingValuesAgent] Requesting handling recommendations from LLM...")
    result = llm.invoke(build_missing_values_prompt(schema, missing_report))
    print("[MissingValuesAgent] Recommendations complete.")
    return result.strip()
