"""
Streamlit upload page for the Agentic EDA pipeline.

Run with:  streamlit run app.py

Upload a CSV and the agent analyzes it (including a deep missing-values report
with pandas remediation code), generates visualizations, and produces a
downloadable PDF. Requires Ollama running locally with the chosen model pulled.
"""
import os
import sys

import streamlit as st

# Make the pipeline importable regardless of where Streamlit is launched from.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from eda_pipeline import run_pipeline

st.set_page_config(
    page_title="Agentic EDA Report Generator",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Agentic EDA Report Generator")
st.write(
    "Upload a CSV and an AI agent will analyze it — digging into missing values, "
    "generating visualizations, and producing a downloadable PDF report with the "
    "Python code it used."
)

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama model", value="mistral")
    st.caption(
        "Requires [Ollama](https://ollama.com) running locally with the model "
        "pulled (e.g. `ollama pull mistral`)."
    )

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is not None and st.button("Generate Report", type="primary"):
    # Persist the upload so the pipeline can read it by path.
    upload_dir = os.path.join(APP_DIR, "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    input_path = os.path.join(upload_dir, uploaded.name)
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Keep each dataset's artifacts in its own output folder.
    dataset_name = os.path.splitext(uploaded.name)[0]
    output_dir = os.path.join(APP_DIR, "outputs", dataset_name)

    try:
        with st.spinner(
            "The agent is analyzing your data — this can take 30–90s with a local LLM..."
        ):
            result = run_pipeline(input_path, model=model, output_dir=output_dir)
    except Exception as e:  # noqa: BLE001 — surface any failure to the user
        st.error(
            f"Pipeline failed: {e}\n\n"
            f"Is Ollama running and is the model available? Try `ollama pull {model}`."
        )
    else:
        st.success("Report generated.")

        report_path = result["report_path"]
        with open(report_path, "rb") as f:
            st.download_button(
                "⬇️ Download PDF Report",
                data=f.read(),
                file_name=os.path.basename(report_path),
                mime="application/pdf",
            )

        plots = [p for p in result["saved_plots"] if os.path.exists(p)]
        if plots:
            st.subheader("Visualizations")
            cols = st.columns(2)
            for i, plot_path in enumerate(plots):
                caption = (
                    os.path.splitext(os.path.basename(plot_path))[0]
                    .replace("_", " ")
                    .title()
                )
                cols[i % 2].image(plot_path, caption=caption, use_container_width=True)

        with st.expander("AI-generated insights"):
            st.write(result["insights"])
