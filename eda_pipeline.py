import os
import sys
import inspect
import argparse
import platform
import subprocess
from langchain_ollama import OllamaLLM

# Ensure imports work regardless of where the script is called from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.schema_extractor import extract_schema
from utils.output_manager import build_report
from utils.missing_analyzer import analyze_missing, save_missing_heatmap
from utils.fallback_plots import generate_fallback_plots
from agents.coder_agent import run_coder_agent
from agents.analyst_agent import run_analyst_agent
from agents.missing_values_agent import run_missing_values_agent


def run_pipeline(
    input_path: str,
    model: str = "mistral",
    output_dir: str = "outputs",
    use_llm_plots: bool = False,
) -> dict:
    """
    Run the full EDA pipeline end to end.

    Shared entry point for both the CLI (main) and the Streamlit app. Does NOT
    auto-open the PDF — callers decide what to do with the result.

    Args:
        use_llm_plots: When True, let the LLM coder agent generate the plots
            (with deterministic fallback). When False (default), use the
            deterministic plotter directly — reliable for any dataset.

    Returns:
        dict with keys: report_path, saved_plots, insights, missing_report
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Initialize LLM ---
    print("[Pipeline] Initializing LLM...")
    llm = OllamaLLM(model=model)

    # --- Extract schema ---
    print("[Pipeline] Extracting dataset schema...")
    df, schema = extract_schema(input_path)
    print(f"[Pipeline] Dataset loaded: {schema['shape'][0]} rows x {schema['shape'][1]} columns")

    # --- Deep missing-values analysis ---
    print("[Pipeline] Analyzing missing values...")
    missing_report = analyze_missing(df)
    save_missing_heatmap(df, output_dir)
    missing_suggestions = run_missing_values_agent(llm, schema, missing_report)

    # --- Visualizations ---
    if use_llm_plots:
        # LLM-generated plots, with deterministic fallback inside the agent.
        generated_code, saved_plots = run_coder_agent(
            llm=llm, df=df, schema=schema, output_dir=output_dir
        )
    else:
        # Deterministic plotter as the primary, reliable path for any dataset.
        print("[Pipeline] Generating plots (deterministic)...")
        saved_plots = generate_fallback_plots(df, schema, output_dir)
        generated_code = (
            "# Plots produced by the deterministic plotter:\n\n"
            + inspect.getsource(generate_fallback_plots)
        )
        print(f"[Pipeline] Done. {len(saved_plots)} plot(s) saved.")

    # --- Analyst agent (narrative insights) ---
    insights = run_analyst_agent(llm=llm, df=df, schema=schema)

    # --- Build PDF report ---
    print("[Pipeline] Building PDF report...")
    report_path = build_report(
        schema=schema,
        insights=insights,
        saved_plots=saved_plots,
        output_dir=output_dir,
        generated_code=generated_code,
        missing_report=missing_report,
        missing_suggestions=missing_suggestions,
    )

    return {
        "report_path": report_path,
        "saved_plots": saved_plots,
        "insights": insights,
        "missing_report": missing_report,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Agentic AI Pipeline for Automated EDA"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file (e.g. data/sample_stock.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Ollama model to use (default: mistral)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for plots and PDF report (default: outputs/)"
    )
    parser.add_argument(
        "--use-llm-plots",
        action="store_true",
        help="Generate plots with the LLM coder agent instead of the "
             "deterministic plotter (default: deterministic)."
    )
    args = parser.parse_args()

    # --- Enforce working directory ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"\n{'='*50}")
    print(f"  Agentic EDA Pipeline")
    print(f"  Input : {args.input}")
    print(f"  Model : {args.model}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*50}\n")

    result = run_pipeline(
        args.input,
        model=args.model,
        output_dir=args.output_dir,
        use_llm_plots=args.use_llm_plots,
    )
    report_path = result["report_path"]
    saved_plots = result["saved_plots"]

    # --- Done ---
    print(f"\n{'='*50}")
    print(f"  Pipeline complete.")
    print(f"  Report: {report_path}")
    print(f"  Plots : {len(saved_plots)} saved")
    print(f"{'='*50}\n")

    # Auto-open PDF on macOS/Linux desktops — disabled in Docker
    in_docker = os.path.exists("/.dockerenv")
    if not in_docker:
        if platform.system() == "Darwin":
            subprocess.run(["open", report_path])
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", report_path])


if __name__ == "__main__":
    main()
