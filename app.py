"""
Streamlit app for the Agentic EDA pipeline.

Run with:  streamlit run app.py

Two tools behind a login:
  • EDA Report   — upload a dataset → missing-values analysis, charts, PDF report.
  • Chat with Docs — local RAG (EmbedChain + Ollama) over URLs / PDFs / text.

Requires Ollama running locally with the chosen model pulled for the AI features.
"""
import os
import sys

import streamlit as st
import streamlit.components.v1 as components

# Make the pipeline importable regardless of where Streamlit is launched from.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from eda_pipeline import run_pipeline
from auth import require_login, logout
from rag_chat import render_chat_with_docs

st.set_page_config(
    page_title="Agentic EDA Analyst",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 1. Login gate — nothing below runs until the user signs in.
# ---------------------------------------------------------------------------
require_login()


# ---------------------------------------------------------------------------
# 2. Landing page
#
# The marketing landing page (landing.html) is shown first. Its "Launch app"
# buttons link to ?app=1, which flips us into the tool view. A session flag
# keeps the user in the tool across reruns.
# ---------------------------------------------------------------------------
if st.query_params.get("app") == "1":
    st.session_state.entered = True


def render_landing() -> None:
    """Render the static landing page full-bleed, with no Streamlit chrome.

    The landing HTML is embedded in a single-viewport iframe that scrolls its
    own content. Previously a fixed 4400px height produced a large empty area
    below the content (the "empty sliding pages"); pinning the iframe to the
    viewport height removes that dead space regardless of content length.
    """
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"], [data-testid="stHeader"] { display: none; }
          .block-container { padding: 6px 12px 0 !important; max-width: 100% !important; }
          [data-testid="stAppViewContainer"] { background: #121C30; }
          /* Full-height landing frame beneath the native launch bar, no gap. */
          .stApp iframe { height: calc(100vh - 58px) !important; min-height: 460px; border: 0; }
          div[data-testid="stButton"] > button {
              background: #2A8C7E !important; color: #fff !important; border: none !important;
              font-weight: 600 !important;
          }
          div[data-testid="stButton"] > button:hover { background: #38B2A2 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # The marketing page is embedded in a sandboxed component iframe, so its own
    # "Launch app" links can't navigate the top window. This native Streamlit
    # button is the reliable way into the tool view.
    _, launch_col = st.columns([5, 1])
    with launch_col:
        if st.button("Launch app →", type="primary", key="enter_app"):
            st.session_state.entered = True
            st.rerun()

    landing_path = os.path.join(APP_DIR, "landing.html")
    with open(landing_path, "r", encoding="utf-8") as f:
        # scrolling=True → the landing scrolls inside a single full-height frame.
        components.html(f.read(), height=900, scrolling=True)


if not st.session_state.get("entered"):
    render_landing()
    st.stop()


# ---------------------------------------------------------------------------
# 3. Shared top bar + navigation
# ---------------------------------------------------------------------------
# Touch-friendly styling: a large tappable file-uploader dropzone and chunky
# primary buttons that respond well to touch.
st.markdown(
    """
    <style>
      /* Big, obviously-tappable upload dropzone */
      [data-testid="stFileUploaderDropzone"] {
          min-height: 150px;
          border: 2px dashed #2A8C7E;
          border-radius: 12px;
          background: rgba(42,140,126,.06);
          cursor: pointer;
          transition: background .15s, border-color .15s;
          align-items: center;
      }
      [data-testid="stFileUploaderDropzone"]:hover,
      [data-testid="stFileUploaderDropzone"]:active {
          background: rgba(42,140,126,.14);
          border-color: #38B2A2;
      }
      [data-testid="stFileUploaderDropzone"] button {
          min-height: 46px; font-weight: 600;
          padding: 10px 22px; touch-action: manipulation;
      }
      /* Chunkier primary buttons for touch */
      .stButton > button, .stDownloadButton > button {
          min-height: 44px; touch-action: manipulation;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

top_l, top_r = st.columns([6, 1])
with top_r:
    if st.button("Logout"):
        logout()

# Sidebar: shared settings + navigation.
with st.sidebar:
    user = st.session_state.get("auth_user", "")
    if user:
        st.caption(f"Signed in as **{user}**")
    page = st.radio("View", ["📊 EDA Report", "💬 Chat with Docs"], index=0)

    st.header("Settings")
    ai_narrative = st.checkbox(
        "Add AI narrative (slower)",
        value=False,
        help="Use a local LLM for missing-value suggestions and insights. "
             "Off = fast deterministic report in seconds (no Ollama needed).",
    )
    model = st.text_input(
        "Ollama model",
        value=os.getenv("OLLAMA_MODEL", "mistral"),
        help="Used for the AI narrative and for Chat with Docs. "
             "Pull it first: `ollama pull mistral`.",
    )
    if ai_narrative:
        st.caption(
            "Requires [Ollama](https://ollama.com) running locally with the model "
            "pulled (e.g. `ollama pull mistral`)."
        )

    if st.button("← Back to landing"):
        st.session_state.entered = False
        st.query_params.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# 4. Pages
# ---------------------------------------------------------------------------
def render_eda_report() -> None:
    """The original EDA upload → report tool."""
    st.title("📊 Agentic EDA Report Generator")
    st.write(
        "Upload a dataset (CSV, Excel, JSON, or TSV) and an AI agent will analyze "
        "it — digging into missing values, generating visualizations, and producing "
        "a downloadable PDF report with the Python code it used."
    )

    uploaded = st.file_uploader(
        "Upload a dataset", type=["csv", "tsv", "txt", "xlsx", "xls", "json"]
    )

    # When a different file is uploaded, drop any previously generated report so
    # we don't show stale results for the wrong dataset.
    if uploaded is not None and st.session_state.get("uploaded_name") != uploaded.name:
        st.session_state.uploaded_name = uploaded.name
        for key in ("result", "report_path", "ai_narrative_used"):
            st.session_state.pop(key, None)

    # Keep the button as its own widget. Nesting the results inside
    # `if st.button(...)` makes them vanish on the next rerun (e.g. when the
    # download button is clicked), so we persist results in session_state and
    # render them independently below.
    generate = st.button("Generate Report", type="primary", disabled=uploaded is None)

    if generate and uploaded is not None:
        upload_dir = os.path.join(APP_DIR, "data", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        input_path = os.path.join(upload_dir, uploaded.name)
        with open(input_path, "wb") as f:
            f.write(uploaded.getbuffer())

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
            st.session_state.result = result
            st.session_state.report_path = result["report_path"]
            st.session_state.ai_narrative_used = ai_narrative

    # Render the generated report from session_state. This runs on every rerun,
    # so clicking the download button no longer wipes the charts and insights.
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


if page.endswith("Chat with Docs"):
    render_chat_with_docs(model)
else:
    render_eda_report()
