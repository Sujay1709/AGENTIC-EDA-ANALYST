# Agentic EDA Analyst

An autonomous Exploratory Data Analysis pipeline powered by a local LLM (Mistral via Ollama).
Sign in, upload a CSV through the web app (or pass one on the CLI), and the pipeline analyzes
it, performs an in-depth missing-values analysis, generates visualizations, and produces a
downloadable PDF report. It also ships a local **Chat with Docs** (RAG) assistant. No paid
APIs, no cloud services required.

**Live app:** https://agentic-eda-analyst.streamlit.app (fast/deterministic features only — see
[Deploy to Streamlit Community Cloud](#deploy-to-streamlit-community-cloud) for why the AI
features need a self-hosted Ollama).

## Features

- **Login gate** — a simple env-based sign-in (`APP_USERNAME` / `APP_PASSWORD`) protects the app.
- **EDA report** — schema extraction, deep missing-values analysis with a heatmap, priority-ranked
  charts, optional AI narrative, and a page-capped PDF report.
- **Touch-friendly upload** — a large tap-target dropzone for CSV / Excel / JSON / TSV.
- **Chat with Docs (EmbedChain)** — add web pages, PDFs, or text to a private knowledge base and
  ask questions. Fully local: Ollama for the LLM **and** embeddings (`nomic-embed-text`), Chroma
  vector store, no API keys.
- **Dockerized** — one `docker compose up` brings up the app plus an Ollama sidecar.

## Architecture

```
eda_pipeline/
├── app.py                   # Streamlit app — login, landing, EDA tool, Chat with Docs
├── auth.py                  # Env-based login gate (APP_USERNAME / APP_PASSWORD)
├── rag_chat.py              # "Chat with Docs" — EmbedChain RAG over URLs/PDFs/text
├── eda_pipeline.py          # Orchestrator — run_pipeline() (used by CLI and web)
├── landing.html             # Marketing landing page (embedded in the app)
├── agents/
│   ├── coder_agent.py       # Generates and self-corrects Python visualization code
│   ├── analyst_agent.py     # Interprets statistics and writes narrative insights
│   └── missing_values_agent.py  # Recommends per-column missing-data handling + pandas code
├── utils/
│   ├── schema_extractor.py  # Extracts dataset metadata (columns, types, nulls)
│   ├── code_executor.py     # Safely executes LLM-generated code with retry logic
│   ├── fallback_plots.py    # Deterministic plots if the LLM never produces working code
│   ├── missing_analyzer.py  # Deep missing-values stats + heatmap
│   └── output_manager.py    # Saves plots and compiles the final PDF report
├── .streamlit/config.toml   # Theme + server config
├── requirements.txt         # Core dependencies
├── requirements-rag.txt     # Optional Chat-with-Docs dependencies (EmbedChain)
├── Dockerfile               # App image
├── docker-compose.yml       # App + Ollama sidecar
├── data/                    # Uploads (data/uploads/) and vector store (data/embedchain/)
└── outputs/                 # Generated plots and PDF reports
```

## How it works

1. **Schema extraction** — reads only dataset metadata, not raw rows, to keep the LLM prompt efficient.
2. **Missing-values analysis** — per-column stats, co-missingness, and a heatmap; the missing-values
   agent recommends handling strategies with concrete pandas code.
3. **Coder agent** — uses the schema to generate pandas/matplotlib/seaborn visualization code.
4. **Safe execution** — runs generated code in a sandboxed namespace with retry; falls back to
   deterministic plots if all retries fail.
5. **Analyst agent** — sends summary statistics to the LLM for pattern recognition and recommendations.
6. **PDF report** — compiles the missing-values analysis, plots, insights, and generated code into a
   single downloadable PDF.

## Run with Docker Compose (recommended)

This runs the app and a local Ollama server together, with no Python setup on your machine.

```bash
cd eda_pipeline

# 1. (Optional) set your own login credentials
cp .env.example .env
# edit APP_USERNAME / APP_PASSWORD in .env

# 2. Build and start the stack (app + ollama sidecar) in the background
docker compose up --build -d

# 3. Pull the models into the Ollama container (first run only)
docker compose exec ollama ollama pull mistral
docker compose exec ollama ollama pull nomic-embed-text

# 4. Open the app
#    → http://localhost:8501
```

Useful commands:

```bash
docker compose logs -f eda-pipeline   # tail app logs
docker compose restart eda-pipeline   # restart just the app
docker compose down                   # stop everything (named volumes persist data + models)
```

Notes:
- Uploads, reports, and the vector store persist in named Docker volumes (`eda_data`, `eda_outputs`),
  and pulled models persist in `ollama_models`, so they survive restarts.
- The **EDA report in fast mode** works immediately. The **AI narrative** and **Chat with Docs**
  need the two models pulled in step 3.
- On macOS, named volumes avoid the "mounts denied / path not shared" Docker Desktop error. If you'd
  rather have reports written to a folder on your host, switch the `eda-pipeline` volumes in
  `docker-compose.yml` to bind mounts (`./data:/app/data`) and add the folder under
  Docker Desktop → Settings → Resources → File Sharing.

## Run locally (without Docker)

Requires Python 3.11 or 3.12 (EmbedChain supports Python ≤ 3.13.2; the core EDA app runs on any
modern Python).

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: Chat with Docs (RAG)
pip install -r requirements-rag.txt

# Pull the local models (requires Ollama installed and running)
ollama pull mistral
ollama pull nomic-embed-text

# Set login credentials (defaults to admin/admin with a warning)
export APP_USERNAME=admin
export APP_PASSWORD=change-me

# Web app
streamlit run app.py            # → http://localhost:8501

# …or the CLI
python eda_pipeline.py --input data/your_file.csv
```

## Deploy to Streamlit Community Cloud

The app is set up to deploy under a unique name:

- **App name / URL:** `agentic-eda-analyst` → **https://agentic-eda-analyst.streamlit.app**
- **Main file path:** `app.py`
- **Python version:** 3.12

Steps:

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → **New app**, pick the repo/branch, set the main file to `app.py`.
3. In **Advanced settings → App URL**, set the subdomain to `agentic-eda-analyst`.
4. In **Advanced settings → Secrets**, add your login credentials:

   ```toml
   APP_USERNAME = "admin"
   APP_PASSWORD = "your-strong-password"
   ```

5. Deploy.

⚠️ **Important:** Streamlit Community Cloud does not run Ollama. On the hosted URL, only the
**fast/deterministic EDA report** works. The **AI narrative** and **Chat with Docs** features need a
reachable Ollama server, so for the full experience run the Docker Compose stack (above) or another
self-hosted deployment. To point the hosted app at a remote Ollama instance, set the `OLLAMA_HOST`
secret to that server's URL.

## Configuration reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_USERNAME` | `admin` | Login username |
| `APP_PASSWORD` | `admin` | Login password (change this!) |
| `OLLAMA_HOST` | `http://localhost:11434` (local) / `http://ollama:11434` (Docker) | Ollama server URL |
| `OLLAMA_MODEL` | `mistral` | Chat / narrative model |
| `EMBEDCHAIN_EMBED_MODEL` | `nomic-embed-text` | Embedding model for Chat with Docs |
| `EMBEDCHAIN_DB_DIR` | `data/embedchain` | Chroma vector store location |

## Requirements

- Python 3.11 / 3.12 (local runs); Docker for the containerized stack
- [Ollama](https://ollama.com) for the AI features (~8 GB RAM for Mistral 7B)

## Why this project

Built as part of learning to design and implement LLM-powered agentic systems. Demonstrates:
prompt engineering, safe code execution, error-recovery loops, local LLM integration, retrieval
-augmented generation, and automated report generation.

## License

MIT
