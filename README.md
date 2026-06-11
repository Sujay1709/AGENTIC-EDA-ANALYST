# Agentic AI Pipeline for EDA

An autonomous Exploratory Data Analysis pipeline powered by a local LLM (Mistral via Ollama).
Upload a CSV through the web page (or pass one on the CLI) and the pipeline analyzes it,
performs an in-depth missing-values analysis, generates visualizations, and produces a PDF
report — no paid APIs, no cloud services.

## Architecture
eda_pipeline/
├── app.py                   # Streamlit web page — upload a CSV, download the PDF
├── eda_pipeline.py          # Orchestrator — `run_pipeline()` (used by CLI and web)
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
├── data/                    # Place your CSV files here (uploads land in data/uploads/)
└── outputs/                 # Generated plots and PDF report saved here
## How It Works

1. **Schema Extraction** — reads only dataset metadata, not raw rows, to keep the LLM prompt efficient
2. **Missing-Values Analysis** — computes per-column stats, co-missingness, and a heatmap, then the missing-values agent recommends handling strategies with concrete pandas code
3. **Coder Agent** — uses the schema to generate pandas/matplotlib/seaborn visualization code
4. **Safe Execution** — runs the generated code in a sandboxed namespace with error capture and retry; falls back to deterministic plots if all retries fail
5. **Analyst Agent** — sends `df.describe()` output to the LLM for pattern recognition and recommendations
6. **PDF Report** — compiles the missing-values analysis, plots, insights, and the generated code into a single downloadable PDF

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Pull the local LLM (requires Ollama installed)
ollama pull mistral
```

## Usage

### Web page (upload a CSV)

```bash
streamlit run app.py
```

Then open the browser tab, upload a CSV, and click **Generate Report** to download the PDF.

### Command line

```bash
python eda_pipeline.py --input data/your_file.csv
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- ~8GB RAM for Mistral 7B

## Why This Project

Built as part of learning to design and implement LLM-powered agentic systems.
Demonstrates: prompt engineering, safe code execution, error recovery loops,
local LLM integration, and automated report generation.

## License

MIT