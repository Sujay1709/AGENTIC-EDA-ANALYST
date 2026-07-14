"""
"Chat with Docs" — a local RAG chatbox built on EmbedChain.

The user adds sources (web URLs, PDFs, or pasted text) to a knowledge base,
then asks natural-language questions. Everything runs locally:

    LLM       → Ollama  (model chosen in the sidebar, e.g. mistral / llama3)
    Embedder  → Ollama  (nomic-embed-text)
    Vector DB → Chroma  (persisted under ./data/embedchain)

This keeps the "no paid APIs" promise: no OpenAI key required.

Prerequisites (once):
    ollama pull mistral            # or your chosen chat model
    ollama pull nomic-embed-text   # embeddings

EmbedChain is imported lazily so the EDA app still runs if it isn't installed.
"""
import os

import streamlit as st

# Where Chroma persists the vector store. Overridable for Docker volumes.
_DB_DIR = os.getenv("EMBEDCHAIN_DB_DIR", os.path.join("data", "embedchain"))
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_EMBED_MODEL = os.getenv("EMBEDCHAIN_EMBED_MODEL", "nomic-embed-text")


def _build_app(chat_model: str):
    """
    Construct an EmbedChain App wired to local Ollama.

    Raises ImportError if embedchain isn't installed — the caller surfaces a
    friendly message with install instructions.
    """
    from embedchain import App  # lazy import

    config = {
        "app": {"config": {"collect_metrics": False}},
        "llm": {
            "provider": "ollama",
            "config": {
                "model": chat_model,
                "temperature": 0.3,
                "stream": False,
                "base_url": _OLLAMA_HOST,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": _EMBED_MODEL,
                "base_url": _OLLAMA_HOST,
            },
        },
        "vectordb": {
            "provider": "chroma",
            "config": {"dir": _DB_DIR, "allow_reset": True},
        },
    }
    return App.from_config(config=config)


def _get_app(chat_model: str):
    """Cache one App per chat model in session_state."""
    key = f"_ec_app::{chat_model}"
    if key not in st.session_state:
        st.session_state[key] = _build_app(chat_model)
    return st.session_state[key]


def render_chat_with_docs(chat_model: str) -> None:
    """Render the full 'Chat with Docs' page. `chat_model` is the Ollama model."""
    st.subheader("💬 Chat with Docs")
    st.caption(
        "Add web pages, PDFs, or text to a private knowledge base, then ask "
        "questions. Runs fully locally via Ollama — no paid API."
    )

    # --- Lazy import guard ---------------------------------------------------
    try:
        app = _get_app(chat_model)
    except ImportError:
        st.error(
            "EmbedChain isn't installed. Add it with:\n\n"
            "```\npip install embedchain\n```\n"
            "Then pull the embedding model: `ollama pull nomic-embed-text`."
        )
        return
    except Exception as e:  # noqa: BLE001 — surface Ollama/DB setup errors
        st.error(
            f"Couldn't start the RAG engine: {e}\n\n"
            "Make sure Ollama is running and the models are pulled "
            f"(`ollama pull {chat_model}` and `ollama pull {_EMBED_MODEL}`)."
        )
        return

    st.session_state.setdefault("rag_sources", [])
    st.session_state.setdefault("rag_history", [])

    # --- Add sources ---------------------------------------------------------
    with st.expander("➕ Add sources to the knowledge base", expanded=not st.session_state["rag_sources"]):
        src_type = st.radio(
            "Source type", ["Web URL", "PDF file", "Text"], horizontal=True
        )

        if src_type == "Web URL":
            url = st.text_input("Page URL", placeholder="https://example.com/article")
            if st.button("Add URL", disabled=not url):
                _add_source(app, url, data_type="web_page", label=url)

        elif src_type == "PDF file":
            pdf = st.file_uploader("Upload a PDF", type=["pdf"], key="rag_pdf")
            if pdf is not None and st.button("Add PDF"):
                os.makedirs(_DB_DIR, exist_ok=True)
                pdf_path = os.path.join(_DB_DIR, pdf.name)
                with open(pdf_path, "wb") as fh:
                    fh.write(pdf.getbuffer())
                _add_source(app, pdf_path, data_type="pdf_file", label=pdf.name)

        else:  # Text
            text = st.text_area("Paste text", height=140)
            if st.button("Add text", disabled=not text.strip()):
                _add_source(app, text, data_type="text", label="(pasted text)")

    if st.session_state["rag_sources"]:
        st.markdown("**Knowledge base:** " + ", ".join(
            f"`{s}`" for s in st.session_state["rag_sources"]
        ))
    else:
        st.info("Add at least one source above to start chatting.")

    st.divider()

    # --- Chat ----------------------------------------------------------------
    for role, msg in st.session_state["rag_history"]:
        with st.chat_message(role):
            st.markdown(msg)

    question = st.chat_input(
        "Ask a question about your documents…",
        disabled=not st.session_state["rag_sources"],
    )
    if question:
        st.session_state["rag_history"].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = app.query(question)
                except Exception as e:  # noqa: BLE001
                    answer = f"Query failed: {e}"
            st.markdown(answer)
        st.session_state["rag_history"].append(("assistant", answer))


def _add_source(app, source: str, data_type: str, label: str) -> None:
    """Add one source to the knowledge base and record it in the UI list."""
    with st.spinner(f"Indexing {label}…"):
        try:
            app.add(source, data_type=data_type)
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't add {label}: {e}")
            return
    st.session_state["rag_sources"].append(label)
    st.success(f"Added {label} to the knowledge base.")
    st.rerun()
