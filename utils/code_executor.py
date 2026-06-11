import re
import sys
import io
import traceback
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for saving plots without a display
import matplotlib.pyplot as plt


def extract_code(llm_output: str) -> str:
    """
    Strip markdown fences that LLMs add despite being told not to.
    Handles ```python ... ``` and ``` ... ``` and plain code.
    """
    # Try to find a fenced code block first
    pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no fences found, return as-is (assume it's plain code)
    return llm_output.strip()

def sanitize_code(code: str, filepath: str = "") -> str:
    """
    Fix common LLM code generation mistakes before execution.
    Each rule targets a specific failure pattern we've observed.
    """
    lines = code.split("\n")
    clean_lines = []

    for line in lines:
        stripped = line.strip()

        # Rule 1: Block any read_csv call — df is already loaded
        if "read_csv" in line:
            clean_lines.append(f"# REMOVED: {stripped}  # df already loaded")
            continue

        # Rule 2: Block plt.show() — freezes execution
        if "plt.show()" in line:
            clean_lines.append(f"# REMOVED: {stripped}  # show() disabled")
            continue

        clean_lines.append(line)

    code = "\n".join(clean_lines)

    # Rule 3: Fix sns.countplot(x='col') missing data= argument
    code = re.sub(
        r"sns\.countplot\(x=['\"](\w+)['\"](?!.*data=)\)",
        r"sns.countplot(x='\1', data=df)",
        code
    )
    code = re.sub(
        r"sns\.countplot\(x=['\"](\w+)['\"](?!.*data=),",
        r"sns.countplot(x='\1', data=df,",
        code
    )

    return code



def execute_code(code: str, df, output_dir: str) -> tuple[bool, str, list]:
    """
    Execute LLM-generated code in a controlled namespace.

    Returns:
        success (bool): Whether execution completed without error
        message (str): stdout output if success, traceback if failure
        saved_plots (list): List of filepaths for any plots saved
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    saved_plots = []

    # --- Plot saving hook ---
    # We monkey-patch plt.savefig to track what gets saved
    original_savefig = plt.savefig
    def tracked_savefig(fname, *args, **kwargs):
        original_savefig(fname, *args, **kwargs)
        saved_plots.append(str(fname))
    plt.savefig = tracked_savefig

    # --- Build the sandbox namespace ---
    # The LLM's generated code runs inside this dict, not in global scope
    namespace = {
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "os": os,
        "output_dir": output_dir,
    }

    # --- Capture stdout ---
    stdout_capture = io.StringIO()

    try:
        sys.stdout = stdout_capture
        code = sanitize_code(code)
        exec(code, namespace)
        sys.stdout = sys.__stdout__

        plt.savefig = original_savefig  # Restore original
        plt.close("all")               # Free memory

        return True, stdout_capture.getvalue(), saved_plots

    except Exception:
        sys.stdout = sys.__stdout__
        plt.savefig = original_savefig
        plt.close("all")

        error_message = traceback.format_exc()
        return False, error_message, []


def run_with_retry(code: str, df, output_dir: str, llm, schema: dict, max_retries: int = 3) -> tuple[str, list]:
    """
    Execute code, and if it fails, send the error back to the LLM for correction.
    This is the core feedback loop that makes it 'agentic'.

    Returns:
        final_code (str): The code that eventually succeeded
        saved_plots (list): Paths to all saved plot files
    """
    current_code = code

    for attempt in range(1, max_retries + 1):
        print(f"\n[Executor] Attempt {attempt}/{max_retries}")
        success, message, saved_plots = execute_code(current_code, df, output_dir)

        if success:
            print(f"[Executor] Code executed successfully on attempt {attempt}")
            return current_code, saved_plots

        print(f"[Executor] Execution failed:\n{message}")

        if attempt == max_retries:
            raise RuntimeError(
                f"Code execution failed after {max_retries} attempts.\n"
                f"Last error:\n{message}"
            )

        # --- Build the correction prompt ---
        correction_prompt = f"""
The following Python code failed to execute.

ORIGINAL CODE:
{current_code}

ERROR:
{message}

DATASET SCHEMA:
- Columns: {schema['columns']}
- Dtypes: {schema['dtypes']}
- Numeric columns: {schema['numeric_columns']}
- Categorical columns: {schema['categorical_columns']}

COMMON CAUSE OF THIS ERROR: calling a numeric/statistical operation (.corr(), .mean(),
.std(), .hist(), histogram) on the full df or on a text column. Modern pandas raises
on string columns instead of dropping them.

Fix the code so it runs without errors. Requirements:
- Compute correlation and histograms ONLY on the numeric columns: df[{schema['numeric_columns']}].
- Never call a numeric operation on a text/categorical column; use value_counts() for those.
- Never use a pandas Series in a boolean context; use .any()/.empty for conditions.
- Save all plots using plt.savefig(os.path.join(output_dir, 'filename.png')).
- The variable 'output_dir' is already defined in the execution environment.
- Only return the corrected Python code, no explanations.
"""
        print(f"[Executor] Sending error back to LLM for correction...")
        raw_output = llm.invoke(correction_prompt)
        current_code = sanitize_code(extract_code(raw_output))

    # Should never reach here due to the raise above, but satisfies type checker
    raise RuntimeError("Unexpected exit from retry loop")