# AI-Driven Retail Data Analytics Platform

A BCA final-project platform for exploring retail sales data, detecting anomalies, forecasting trends, and generating local AI-assisted business insights.

## What it does

- Uploads time-based retail CSV files for time-and-sales and regional analysis.
- Detects anomalies and handles missing numeric values.
- Produces trend, seasonality, regional-sales, and forecasting visualizations.
- Generates local LLM insights with Ollama.
- Optionally sends traces to a locally running OPIK instance.

## Project layout

```text
app/                 Earlier SQLite-backed Streamlit dashboard
data/raw/            Versioned sample retail dataset
database/            Locally generated SQLite database (ignored)
dynamic_app/         Current upload-based Streamlit dashboard
processed/           Generated outputs (ignored)
src/                 Ingestion, cleaning, analytics, and LLM helpers
```

## Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/) for AI-generated insights (optional)
- Docker Desktop plus an OPIK stack for tracing (optional)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For local AI insights, start Ollama and download the model used by the app:

```powershell
ollama pull llama3.2:1b
```

## Run the current dashboard

```powershell
streamlit run dynamic_app/dashboard.py
```

Upload `data/raw/retail_store_sales.csv` to try it with the included sample data.

The dashboard expects Ollama at `http://localhost:11434`. It also attempts to trace through OPIK at `http://localhost:5173/api`; start the OPIK Docker Compose stack before running it if you want observability. OPIK is optional for the analytics concept, but the current dashboard imports it at startup.

## Run the earlier SQLite workflow

The scripts below rebuild the local SQLite database from the versioned raw dataset and then launch the earlier dashboard:

```powershell
python src/ingest_data.py
python src/clean_data.py
python src/analytics_predict.py
python src/generate_insights.py
streamlit run app/dashboard.py
```

`database/retail_analytics.db` is generated locally and intentionally excluded from Git.

## Repository policy

Tracked:

- Application and analysis source code.
- `data/raw/retail_store_sales.csv`, a small reproducible sample dataset.
- This README, `requirements.txt`, the final report PDF, and other documentation.

Not tracked:

- SQLite databases, processed/output data, Python environments and caches.
- Secrets or local `.env` files.
- Docker/OPIK local volumes, logs, and other runtime state.

Before publishing a different dataset, verify that it contains no personal, confidential, licensed, or otherwise restricted data.

## Final report

The project report is included as `Ai-driven Retail Data Analytics Platform – Final Project Report.pdf`.
