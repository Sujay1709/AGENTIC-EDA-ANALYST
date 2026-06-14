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
import streamlit.components.v1 as components

# Make the pipeline importable regardless of where Streamlit is launched from.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from eda_pipeline import run_pipeline

st.set_page_config(
    page_title="Agentic EDA Report Generator",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Landing page
#
# The first thing visitors see is the marketing landing page (landing.html).
# Its "Launch app" buttons link to ?app=1, which flips us into the tool view.
# A session flag keeps the user in the tool across reruns.
# ---------------------------------------------------------------------------
if st.query_params.get("app") == "1":
    st.session_state.entered = True


def render_landing() -> None:
    """Render the static landing page full-bleed, with no Streamlit chrome."""
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"], [data-testid="stHeader"] { display: none; }
          .block-container { padding: 0 !important; max-width: 100% !important; }
          [data-testid="stAppViewContainer"] { background: #121C30; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    landing_path = os.path.join(APP_DIR, "landing.html")
    with open(landing_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=4400, scrolling=True)


if not st.session_state.get("entered"):
    render_landing()
    st.stop()


# ---------------------------------------------------------------------------
# Tool view
# ---------------------------------------------------------------------------
top_l, top_r = st.columns([6, 1])
with top_r:
    if st.button("← Home"):
        st.session_state.entered = False
        st.query_params.clear()
        st.rerun()

st.title("📊 Agentic EDA Report Generator")
st.write(
    "Upload a dataset (CSV, Excel, JSON, or TSV) and an AI agent will analyze it — "
    "digging into missing values, generating visualizations, and producing a "
    "downloadable PDF report with the Python code it used."
)

with st.sidebar:
    st.header("Settings")
    ai_narrative = st.checkbox(
        "Add AI narrative (slower)",
        value=False,
        help="Use a local LLM for missing-value suggestions and insights. "
             "Off = fast deterministic report in seconds (no Ollama needed).",
    )
    model = st.text_input("Ollama model", value="mistral", disabled=not ai_narrative)
    if ai_narrative:
        st.caption(
            "Requires [Ollama](https://ollama.com) running locally with the model "
            "pulled (e.g. `ollama pull mistral`)."
        )
    else:
        st.caption("Fast mode: deterministic analysis, plots, and suggestions — no LLM call.")

uploaded = st.file_uploader(
    "Upload a dataset", type=["csv", "tsv", "txt", "xlsx", "xls", "json"]
)

# When a different file is uploaded, drop any previously generated report so we
# don't show stale results for the wrong dataset.
if uploaded is not None and st.session_state.get("uploaded_name") != uploaded.name:
    st.session_state.uploaded_name = uploaded.name
    for key in ("result", "report_path", "ai_narrative_used"):
        st.session_state.pop(key, None)

# Gate the button on having a file, but keep it as its own widget. Nesting the
# results inside `if st.button(...)` makes them vanish on the next rerun (e.g.
# when the download button is clicked), so we persist results in session_state
# and render them independently below.
generate = st.button("Generate Report", type="primary", disabled=uploaded is None)

if generate and uploaded is not None:
    # Persist the upload so the pipeline can read it by path.
    upload_dir = os.path.join(APP_DIR, "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    input_path = os.path.join(upload_dir, uploaded.name)
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Keep each dataset's artifacts in its own output folder.
    dataset_name = os.path.splitext(uploaded.name)[0]
    output_dir = os.path.join(APP_DIR, "outputs", dataset_name)

    spinner_msg = (
        "The agent is analyzing your data — this can take 30–90s with a local LLM..."
        if ai_narrative
        else "Analyzing your data and building the report (a few seconds)..."
    )
    try:
        with st.spinner(spinner_msg):
            result = run_pipeline(
                input_path,
                model=model,
                output_dir=output_dir,
                ai_narrative=ai_narrative,
            )
    except Exception as e:  # noqa: BLE001 — surface any failure to the user
        for key in ("result", "report_path", "ai_narrative_used"):
            st.session_state.pop(key, None)
        st.error(
            f"Pipeline failed: {e}\n\n"
            f"Is Ollama running and is the model available? Try `ollama pull {model}`."
        )
    else:
        # Stash results so they survive reruns triggered by other widgets
        # (notably the download button below).
        st.session_state.result = result
        st.session_state.report_path = result["report_path"]
        st.session_state.ai_narrative_used = ai_narrative

# Render the generated report from session_state. This runs on every rerun, so
# clicking the download button no longer wipes the visualizations and insights.
if st.session_state.get("result"):
    result = st.session_state.result
    report_path = st.session_state.report_path

    st.success("Report generated.")

    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button(
                "⬇️ Download PDF Report",
                data=f.read(),
                file_name=os.path.basename(report_path),
                mime="application/pdf",
                key="download_report",
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

    insights_label = (
        "AI-generated insights"
        if st.session_state.get("ai_narrative_used")
        else "Insights"
    )
    with st.expander(insights_label):
        st.write(result["insights"])
