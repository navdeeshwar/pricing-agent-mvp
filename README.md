# Autonomous Dynamic Pricing Agent MVP

An end-to-end, neuro-symbolic dynamic pricing system that combines machine learning demand forecasting (LightGBM), a constrained non-convex optimization engine with deterministic business guardrails, a FastAPI REST microservice, a 3-tab Streamlit Merchant Command Center, and an autonomous AI agent orchestrator built in n8n.

---

## 📌 Architecture Overview

```text
[ Raw Transaction Logs ] (1M+ rows)
         │
         ▼
[ 1. Data & ML Pipeline ] (src/1_prep_and_train.py)
   ├── Data Hygiene (drop cancellations, fees, bulk outliers)
   ├── Weekly Demand Aggregation & VWAP Feature Engineering
   └── LightGBM Regressor (Demand Elasticity Modeling)
         │
         ▼ Models & Catalog (.joblib, .json)
[ 2. Constrained Solver ] (src/2_solver_and_guardrails.py)
   ├── Guardrails: Margin Floor (≥15%) & Velocity Cap (±15%)
   └── 1D Grid Search Optimizer (Max Profit / Max Revenue + 50-pt Curve)
         │
         ▼
[ 3. FastAPI Microservice ] (src/3_api_and_agent.py)
   ├── Resilient Pydantic Input Sanitizers (strips currency symbols)
   ├── Pricing Tools: /products, /pricing/recommend, /pricing/simulate
   └── HITL Governance: /pricing/portfolio, /pricing/approve, /pricing/audit
         │
         ├──────────────────────────────────────────┐
         ▼                                          ▼
[ 4A. Merchant Command Center ]            [ 4B. n8n AI Agent ]
(src/4_dashboard.py · Streamlit :8501)     (Docker Compose :5678)
   ├── Interactive Price Slider & Curve       ├── LangChain Agent + Memory
   ├── Portfolio Opportunity Queue            └── Dynamic $fromAI HTTP Tools
   └── 1-Click Approval & Audit Log
```

---

## 🚀 Key Features

* **Empirical Demand & Elasticity Modeling**: Aggregates 1M+ retail transactions into weekly product demand buckets; computes Volume-Weighted Average Prices (VWAP) and normalizes price ratios against historical baselines.
* **Non-Linear Response Surface**: Uses LightGBM gradient boosted trees to model complex non-linear price sensitivity and seasonal shopping patterns.
* **Deterministic Business Guardrails**:
  * **Margin Floor**: Mathematically forbids pricing below cost plus a minimum 15% gross margin.
  * **Price Velocity Cap**: Restricts single-cycle adjustments to ±15% to protect customer trust.
  * **Historical Corridor**: Bounds candidate searches to historically observed price ranges to prevent out-of-distribution tree extrapolation.
* **1D Grid Search Optimization**: Evaluates candidate prices across non-differentiable tree step functions in ~40–47ms per SKU to find the global profit/revenue optimum.
* **Merchant Command Center UI (`src/4_dashboard.py`)**: 3-tab Streamlit + Plotly dashboard featuring an interactive price slider over the shaded guardrail corridor, a portfolio-wide profit-lift queue, and a live governance audit log.
* **Human-in-the-Loop (HITL) Governance**: Tiered autonomy (`≤3%` auto-approved, `3–15%` manager-reviewed, out-of-corridor blocked) with immutable logging to `models/price_audit_log.csv`.
* **Autonomous n8n Agent**: Connects an LLM agent to the pricing engine via HTTP tools for multi-turn product search, optimization, and what-if simulation.
* **Fully Containerized**: Packaged with Docker Compose to run FastAPI, Streamlit UI, and n8n together.

---

## 🛠️ Quickstart

### Option 1: Launch Full Stack with Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/<your-username>/pricing-agent-mvp.git
cd pricing-agent-mvp

# Start FastAPI, Streamlit Merchant UI, and n8n together
docker compose up -d --build
```

* **Merchant Command Center UI**: Visit `http://localhost:8501`
* **FastAPI Docs**: Visit `http://localhost:8000/docs`
* **n8n Canvas**: Visit `http://localhost:5678` (Import `n8n_pricing_workflow.json` and attach an LLM credential)

### Option 2: Run Locally (Without Docker)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start the FastAPI server
uvicorn src.3_api_and_agent:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start the Streamlit Merchant Command Center
streamlit run src/4_dashboard.py
```

---

## 📁 Project Structure

```text
pricing-agent-mvp/
├── data/                              # Raw transaction dataset (ignored by git)
├── models/                            # Serialized LightGBM model, catalog & audit log
│   ├── demand_model.joblib
│   ├── product_catalog.json
│   └── feature_cols.json
├── src/                               # Core application logic
│   ├── 1_prep_and_train.py            # Data cleaning & ML training pipeline
│   ├── 2_solver_and_guardrails.py     # Business guardrails & 1D grid optimizer
│   ├── 3_api_and_agent.py             # FastAPI microservice & HITL approval endpoints
│   └── 4_dashboard.py                 # Streamlit + Plotly Merchant Command Center UI
├── Dockerfile                         # Container definition for Python API & UI
├── docker-compose.yml                 # Multi-service stack (API + UI + n8n)
├── n8n_pricing_workflow.json          # Pre-configured n8n AI Agent workflow
├── Summary_and_metrics.py             # End-to-end portfolio metrics & audit script
├── Pricing_Agent_Interview_Guide.pdf  # In-depth technical & conceptual guide
└── requirements.txt                   # Project dependencies
```

---

## 📖 Technical Deep Dive & Interview Guide

A comprehensive, deep-dive interview preparation guide is included in this repository:
* **[Pricing_Agent_Interview_Guide.pdf](./Pricing_Agent_Interview_Guide.pdf)**: Detailed breakdown of the 4-layer architecture, V2 Merchant UI & HITL governance, mathematical equations, CS/AI analogies, real debugging case studies, and high-probability interview questions.
