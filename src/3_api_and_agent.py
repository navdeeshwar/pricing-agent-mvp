"""
===================================================================================
MODULE 3: REST API MICROSERVICE & LLM TOOL INTERFACE
File: src/3_api_and_agent.py
===================================================================================
WHAT THIS SCRIPT DOES:
1. Launches a high-performance web API using FastAPI to serve our pricing engine.
2. Keeps the Machine Learning model and product catalog permanently loaded in RAM 
   so optimization requests return in under 10 milliseconds.
3. Implements resilient Pydantic input sanitizers that clean messy inputs from LLMs 
   (e.g., automatically stripping currency symbols like '£0.50' or turning natural phrases
   like 'maximize our gross profit' into clean internal code).
4. Exposes HTTP endpoints for n8n AI Agent tools:
   - GET /products: Catalog search by keyword or SKU.
   - GET/POST /pricing/recommend: Runs optimization under business guardrails.
   - GET/POST /pricing/simulate: Runs what-if scenarios and checks for guardrail violations.
===================================================================================
"""

import os
import re
import sys
from typing import Optional, Literal, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import importlib

# ---------------------------------------------------------------------------------
# DYNAMIC MODULE IMPORT
# ---------------------------------------------------------------------------------
# Why importlib instead of standard 'from src.2_solver_and_guardrails import ...'?
# In Python, standard module names cannot start with a number (like '2_solver...').
# The Python grammar parser treats '2_' as an invalid numeric literal. 
# Using importlib.import_module() allows us to load the module cleanly by its string name 
# without needing to rename the numbered file.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

solver_module = importlib.import_module("src.2_solver_and_guardrails")
PricingOptimizer = solver_module.PricingOptimizer


# ---------------------------------------------------------------------------------
# FASTAPI APPLICATION & CORS INITIALIZATION
# ---------------------------------------------------------------------------------
app = FastAPI(
    title="Dynamic Pricing Agent API",
    description="Autonomous pricing optimization, demand simulation, and guardrail enforcement MVP",
    version="1.0.0"
)

# CORS (Cross-Origin Resource Sharing) Middleware:
# By default, browsers and external containers block network requests coming from 
# different ports or domains for security reasons. Adding CORSMiddleware tells FastAPI 
# to allow incoming API calls from our n8n Docker container, web dashboards, or mobile apps.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow requests from any origin (n8n, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],      # Allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# SINGLETON INITIALIZATION (Loading once into RAM):
# We create a single instance of PricingOptimizer() when the server boots up.
# This loads the LightGBM model and the 100-product catalog into computer memory once.
# If we re-loaded the model on every single user question, each reply would take 2-5 seconds.
# Keeping it in RAM means predictions happen in under 10 milliseconds!
optimizer = PricingOptimizer()


# ---------------------------------------------------------------------------------
# DATA VALIDATION SCHEMAS (CLEANING MESSY LLM INPUTS)
# ---------------------------------------------------------------------------------
# In an AI Agent workflow, Large Language Models (LLMs) like GPT-4 often pass imperfect data.
# For example, instead of passing a pure float like 0.50, an LLM might pass "£0.50".
# Or instead of passing the exact code "max_profit", it might pass "maximize our gross profit".
# Pydantic models act as a security checkpoint, cleaning these inputs before running math.
# ---------------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """Schema for price recommendation requests."""
    stock_code: str = Field(..., example="15036", description="Product StockCode")
    objective: Optional[str] = Field("max_profit", description="Optimization goal: max_profit or max_revenue")
    month: Optional[int] = Field(11, ge=1, le=12, description="Target calendar month")
    week: Optional[int] = Field(45, ge=1, le=53, description="Target calendar week")
    quarter: Optional[int] = Field(4, ge=1, le=4, description="Target calendar quarter")

    @field_validator("stock_code", mode="before")
    def clean_stock_code(cls, v: Any) -> str:
        """Coerces integer SKUs (e.g. 15036) or padded strings (' 15036 ') into clean strings."""
        return str(v).strip()

    @field_validator("objective", mode="before")
    def normalize_objective(cls, v: Any) -> str:
        """
        Translates natural language intent into strict internal enum codes.
        If the LLM says 'revenue', 'max revenue', or 'maximize sales', it maps to 'max_revenue'.
        Otherwise, it safely defaults to 'max_profit'.
        """
        if not v:
            return "max_profit"
        val = str(v).lower()
        return "max_revenue" if "rev" in val else "max_profit"


class SimulateRequest(BaseModel):
    """Schema for what-if price simulation requests."""
    stock_code: str = Field(..., example="15036", description="Product StockCode")
    proposed_price: float = Field(..., gt=0, description="Hypothetical price to evaluate")
    month: Optional[int] = Field(11, ge=1, le=12)
    week: Optional[int] = Field(45, ge=1, le=53)
    quarter: Optional[int] = Field(4, ge=1, le=4)

    @field_validator("stock_code", mode="before")
    def clean_stock_code(cls, v: Any) -> str:
        return str(v).strip()

    @field_validator("proposed_price", mode="before")
    def clean_price(cls, v: Any) -> float:
        """
        Strips away currency symbols (£, $, €), commas, and spaces using regex.
        Turns '£0.50' or '$ 1.25' into clean floats (0.50, 1.25) so math calculations don't crash.
        """
        if isinstance(v, (int, float)):
            return float(v)
        cleaned = re.sub(r"[^\d.]", "", str(v))
        if not cleaned:
            raise ValueError(f"Could not parse numeric price from '{v}'")
        return float(cleaned)


class AgentQueryRequest(BaseModel):
    """Schema for the direct natural language query endpoint."""
    query: str = Field(..., example="What is the optimal price for product 15036 to maximize profit?")


# ----------------- API Endpoints ----------------- #

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "loaded_products": len(optimizer.catalog),
        "model_loaded": optimizer.model is not None
    }


@app.get("/products")
def list_products(limit: int = 20, search: Optional[str] = None):
    results = []
    for code, info in optimizer.catalog.items():
        if search and search.lower() not in info["description"].lower() and search.lower() not in code.lower():
            continue
        results.append(info)
        if len(results) >= limit:
            break
    return {"count": len(results), "products": results}


@app.get("/pricing/recommend")
def recommend_price_get(
    stock_code: str,
    objective: Optional[str] = "max_profit",
    month: int = 11,
    week: int = 45,
    quarter: int = 4
):
    print(f"[n8n GET /pricing/recommend] stock_code={stock_code!r}, objective={objective!r}")
    req = RecommendRequest(
        stock_code=stock_code,
        objective=objective,
        month=month,
        week=week,
        quarter=quarter
    )
    return recommend_price(req)


@app.post("/pricing/recommend")
def recommend_price(req: RecommendRequest):
    if req.stock_code not in optimizer.catalog:
        raise HTTPException(status_code=404, detail=f"StockCode '{req.stock_code}' not found in catalog.")
    
    result = optimizer.optimize_price(
        stock_code=req.stock_code,
        objective=req.objective,
        month=req.month,
        week=req.week,
        quarter=req.quarter
    )
    return result


@app.get("/pricing/simulate")
def simulate_price_get(
    stock_code: str,
    proposed_price: str,
    month: int = 11,
    week: int = 45,
    quarter: int = 4
):
    print(f"[n8n GET /pricing/simulate] stock_code={stock_code!r}, proposed_price={proposed_price!r}")
    req = SimulateRequest(
        stock_code=stock_code,
        proposed_price=proposed_price,
        month=month,
        week=week,
        quarter=quarter
    )
    return simulate_price(req)


@app.post("/pricing/simulate")
def simulate_price(req: SimulateRequest):
    if req.stock_code not in optimizer.catalog:
        raise HTTPException(status_code=404, detail=f"StockCode '{req.stock_code}' not found in catalog.")
    
    item = optimizer.catalog[req.stock_code]
    base_price = item["base_price"]
    cost = item["estimated_cost"]

    # Compute demand and financials at proposed price
    demand = optimizer.predict_demand(
        stock_code=req.stock_code,
        price=req.proposed_price,
        ref_price=base_price,
        month=req.month,
        week=req.week,
        quarter=req.quarter
    )
    revenue = req.proposed_price * demand
    profit = (req.proposed_price - cost) * demand
    margin_pct = ((req.proposed_price - cost) / req.proposed_price * 100) if req.proposed_price > 0 else 0.0

    # Validate against guardrails
    p_min, p_max = optimizer.guardrails.compute_feasible_price_range(
        base_price=base_price,
        unit_cost=cost,
        min_hist_price=item["min_observed_price"],
        max_hist_price=item["max_observed_price"]
    )

    violations = []
    if req.proposed_price < p_min:
        violations.append(f"Price £{req.proposed_price:.2f} is below the allowed safety floor £{p_min:.2f} (margin or velocity guardrail).")
    if req.proposed_price > p_max:
        violations.append(f"Price £{req.proposed_price:.2f} exceeds the allowed ceiling £{p_max:.2f} (velocity guardrail).")

    return {
        "stock_code": req.stock_code,
        "description": item["description"],
        "proposed_price": round(req.proposed_price, 2),
        "baseline_price": base_price,
        "unit_cost": cost,
        "projected_demand": round(demand, 1),
        "projected_revenue": round(revenue, 2),
        "projected_profit": round(profit, 2),
        "gross_margin_pct": round(margin_pct, 1),
        "guardrail_status": "APPROVED" if not violations else "FLAGGED",
        "violations": violations,
        "allowed_corridor": [p_min, p_max]
    }


@app.post("/agent/query")
def agent_query(req: AgentQueryRequest):
    """
    Parses natural language requests, triggers the appropriate tool, and formats an executive summary.
    """
    query = req.query.strip()
    
    # Extract StockCode from query (looks for matches in our catalog or 4-6 char alphanumerics)
    target_code = None
    for code in optimizer.catalog.keys():
        if re.search(rf"\b{re.escape(code)}\b", query, re.IGNORECASE):
            target_code = code
            break
            
    if not target_code:
        # Fallback regex search
        match = re.search(r"\b([0-9]{4,6}[A-Za-z]?)\b", query)
        if match and match.group(1) in optimizer.catalog:
            target_code = match.group(1)

    if not target_code:
        sample_codes = list(optimizer.catalog.keys())[:5]
        return {
            "query": query,
            "response": f"I couldn't identify a known StockCode in your request. Please specify a valid product code (e.g., {', '.join(sample_codes)})."
        }

    # Check if this is a simulation ("what if price is X") or recommendation
    sim_match = re.search(r"(?:price\s+(?:at|to|is)|what\s+if)\s*[£$]?\s*([0-9]+(?:\.[0-9]+)?)", query, re.IGNORECASE)
    
    if sim_match:
        proposed_price = float(sim_match.group(1))
        sim = simulate_price(SimulateRequest(stock_code=target_code, proposed_price=proposed_price))
        status_text = "complies with all guardrails" if sim["guardrail_status"] == "APPROVED" else f"triggers guardrails: {'; '.join(sim['violations'])}"
        
        narrative = (
            f"Scenario Analysis for {sim['description']} (StockCode: {target_code}):\n"
            f"- Setting price to £{proposed_price:.2f} (Base: £{sim['baseline_price']:.2f})\n"
            f"- Expected Weekly Demand: {sim['projected_demand']} units\n"
            f"- Projected Weekly Revenue: £{sim['projected_revenue']:.2f} | Gross Profit: £{sim['projected_profit']:.2f} ({sim['gross_margin_pct']}% margin)\n"
            f"- Guardrail Audit: {sim['guardrail_status']} ({status_text})"
        )
        return {"query": query, "type": "simulation", "result": sim, "narrative": narrative}

    # Otherwise, run optimization
    objective = "max_revenue" if "revenue" in query.lower() else "max_profit"
    rec = optimizer.optimize_price(stock_code=target_code, objective=objective)
    
    rec_data = rec["recommendation"]
    base_data = rec["baseline"]
    narrative = (
        f"Pricing Recommendation for {rec['description']} (StockCode: {target_code}):\n"
        f"- Recommended Price: £{rec_data['recommended_price']:.2f} ({rec_data['price_change_pct']:+}% vs baseline £{base_data['price']:.2f})\n"
        f"- Projected Weekly Profit: £{rec_data['projected_profit']:.2f} ({rec_data['profit_lift_pct']:+}% lift)\n"
        f"- Projected Weekly Revenue: £{rec_data['projected_revenue']:.2f} ({rec_data['revenue_lift_pct']:+}% lift)\n"
        f"- Expected Weekly Demand: {rec_data['projected_demand']} units\n"
        f"- Guardrails Enforced: Corridor [£{rec['guardrails_applied']['allowed_price_corridor'][0]:.2f}, £{rec['guardrails_applied']['allowed_price_corridor'][1]:.2f}] with 15% margin floor."
    )
    return {"query": query, "type": "recommendation", "result": rec, "narrative": narrative}


if __name__ == "__main__":
    import uvicorn
    print("Starting Pricing Agent API server on http://0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
