import os
import inspect
from langchain_ollama import OllamaLLM
from utils.code_executor import extract_code, run_with_retry
from utils.fallback_plots import generate_fallback_plots


def build_coder_prompt(schema: dict) -> str:
    """
    Constructs the prompt for the coder agent.
    Explicit instructions prevent the most common LLM failure modes.
    """
    return f"""
You are an expert Python data scientist.

You have been given a dataset with the following structure:
- File: {schema['filepath']}
- Shape: {schema['shape'][0]} rows x {schema['shape'][1]} columns
- Columns: {schema['columns']}
- Data types: {schema['dtypes']}
- Numeric columns: {schema['numeric_columns']}
- Categorical columns: {schema['categorical_columns']}
- Missing value percentages: {schema['missing_percent']}
- Sample values: {schema['sample_values']}

The numeric columns are EXACTLY: {schema['numeric_columns']}
The categorical columns are EXACTLY: {schema['categorical_columns']}

Write Python code using pandas, matplotlib, and seaborn that:
1. Defines: numeric_cols = {schema['numeric_columns']} and cat_cols = {schema['categorical_columns']}
2. Plots a distribution histogram for each column in numeric_cols only
3. Plots a correlation heatmap computed ONLY on df[numeric_cols] (i.e. df[numeric_cols].corr())
4. Plots a bar chart of value counts for each column in cat_cols (skip 'date' type columns)
5. Plots a missing values heatmap using seaborn if any column has missing data
6. Saves every plot using: plt.savefig(os.path.join(output_dir, 'filename.png'), bbox_inches='tight', dpi=150)
7. Calls plt.close() after every savefig to free memory

STRICT RULES:
- The variable 'output_dir' is already defined in your execution environment. Do not redefine it.
- The DataFrame 'df' is already loaded in memory. DO NOT call pd.read_csv() under any circumstances. DO NOT import or reference the CSV filepath. Use 'df' directly.
- NEVER call .corr(), .mean(), .std(), .hist(), or any numeric/statistical operation on the full df or on a text column. These raise errors on string columns (e.g. names). ALWAYS subset to df[numeric_cols] first.
- Histograms and the correlation heatmap must use ONLY the columns in numeric_cols.
- For categorical columns use df[col].value_counts().plot(kind='bar') or sns.countplot(x=col, data=df).
- Do not call plt.show() anywhere — it will block execution.
- Do not use any library not in this list: pandas, numpy, matplotlib, seaborn, os
- Use only column names exactly as listed above
- Never use a pandas Series directly in a boolean context (e.g. `if df['col']:` or `if df.isnull():`). To test for missing data use `df.isnull().values.any()`, and for a single column use `df['col'].isnull().any()`.
- Missing values (NaN) are normal and expected. NEVER raise an error or stop when a column has missing data. matplotlib/seaborn handle NaN; for histograms call df[col].dropna() before plotting. Do not write validation code that raises.
- Only return executable Python code, no explanations, no markdown fences
"""


def run_coder_agent(llm: OllamaLLM, df, schema: dict, output_dir: str) -> tuple[str, list]:
    """
    Runs the coder agent: generates visualization code and executes it.

    Args:
        llm: Initialized OllamaLLM instance
        df: Loaded pandas DataFrame
        schema: Output from schema_extractor
        output_dir: Directory to save plots

    Returns:
        generated_code (str): The final working code
        saved_plots (list): Paths to all saved plot images
    """
    os.makedirs(output_dir, exist_ok=True)

    print("[CoderAgent] Generating visualization code...")
    raw_output = llm.invoke(build_coder_prompt(schema))
    generated_code = extract_code(raw_output)

    print("[CoderAgent] Code generated. Starting execution with retry loop...")
    try:
        final_code, saved_plots = run_with_retry(
            code=generated_code,
            df=df,
            output_dir=output_dir,
            llm=llm,
            schema=schema,
            max_retries=3
        )
    except RuntimeError as e:
        # The LLM never produced working code — fall back to deterministic plots
        # so the pipeline still produces a report instead of crashing.
        print(f"[CoderAgent] Retry loop exhausted ({e}).")
        print("[CoderAgent] Falling back to deterministic plotting...")
        saved_plots = generate_fallback_plots(df, schema, output_dir)
        # Surface the actual code that produced these plots so the report's
        # appendix matches the figures instead of claiming nothing was generated.
        final_code = (
            "# LLM code generation failed after all retries.\n"
            "# The plots below were produced by the deterministic fallback:\n\n"
            + inspect.getsource(generate_fallback_plots)
        )

    print(f"[CoderAgent] Done. {len(saved_plots)} plot(s) saved.")
    return final_code, saved_plots