"""
===================================================================================
MODULE 4: MERCHANT COMMAND CENTER UI (STREAMLIT + PLOTLY)
File: src/4_dashboard.py
===================================================================================
Run locally: streamlit run src/4_dashboard.py
Or via Docker: http://localhost:8501
===================================================================================
"""

import os
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Autonomous Pricing Command Center",
    page_icon="📊",
    layout="wide"
)

st.title("Autonomous Pricing Command Center")
st.caption("Neuro-Symbolic Pricing Engine · LightGBM Elasticity + Deterministic Guardrails + HITL Governance")


@st.cache_data(ttl=30)
def fetch_products():
    resp = requests.get(f"{API_BASE_URL}/products", params={"limit": 100}, timeout=10)
    resp.raise_for_status()
    return resp.json()["products"]


@st.cache_data(ttl=60)
def fetch_portfolio(objective: str):
    resp = requests.get(
        f"{API_BASE_URL}/pricing/portfolio",
        params={"limit": 15, "objective": objective},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["opportunities"]


def fetch_recommendation(stock_code: str, objective: str):
    resp = requests.get(
        f"{API_BASE_URL}/pricing/recommend",
        params={"stock_code": stock_code, "objective": objective},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def fetch_simulation(stock_code: str, proposed_price: float):
    resp = requests.get(
        f"{API_BASE_URL}/pricing/simulate",
        params={"stock_code": stock_code, "proposed_price": proposed_price},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()


try:
    products = fetch_products()
except Exception as e:
    st.error(f"Cannot connect to Pricing API at `{API_BASE_URL}`. Make sure FastAPI is running. ({e})")
    st.stop()

product_map = {f"{p['stock_code']} — {p['description']}": p for p in products}

tab1, tab2, tab3 = st.tabs([
    "🎯 SKU Simulator & Guardrails",
    "📈 Portfolio Opportunity Queue",
    "🛡️ Governance & Audit Log"
])

# ---------------------------------------------------------------------------------
# TAB 1: SINGLE-SKU SIMULATOR & GUARDRAIL VISUALIZER
# ---------------------------------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1, 2.2], gap="large")

    with col_left:
        selected_label = st.selectbox("Select SKU", options=list(product_map.keys()))
        item = product_map[selected_label]
        stock_code = item["stock_code"]

        objective = st.radio(
            "Optimization Objective",
            options=["max_profit", "max_revenue"],
            format_func=lambda x: "Maximize Gross Profit" if x == "max_profit" else "Maximize Revenue",
            horizontal=True
        )

        rec_data = fetch_recommendation(stock_code, objective)
        rec = rec_data["recommendation"]
        base = rec_data["baseline"]
        p_min, p_max = rec_data["guardrails_applied"]["allowed_price_corridor"]

        slider_min = round(min(p_min * 0.75, item["min_observed_price"] * 0.8), 2)
        slider_max = round(max(p_max * 1.25, item["max_observed_price"] * 1.1), 2)

        proposed_price = st.slider(
            "Test Custom Price (£)",
            min_value=float(slider_min),
            max_value=float(max(slider_max, slider_min + 0.50)),
            value=float(rec["recommended_price"]),
            step=0.01,
            format="£%.2f"
        )

        sim = fetch_simulation(stock_code, proposed_price)

        if sim["guardrail_status"] == "APPROVED":
            st.success(f"GUARDRAIL STATUS: APPROVED · Safe Corridor [£{p_min:.2f}, £{p_max:.2f}]")
        else:
            st.error(f"GUARDRAIL STATUS: FLAGGED · {'; '.join(sim['violations'])}")

        st.divider()
        st.subheader("Human-in-the-Loop Approval")
        approver = st.text_input("Approver Name", value="Category Manager")
        reason = st.text_input("Approval Note", value=f"UI review ({objective})")

        if st.button("✅ Approve & Apply Price", type="primary", use_container_width=True):
            approve_resp = requests.post(
                f"{API_BASE_URL}/pricing/approve",
                json={
                    "stock_code": stock_code,
                    "approved_price": proposed_price,
                    "approver": approver,
                    "reason": reason
                },
                timeout=10
            )
            if approve_resp.status_code == 200:
                res_json = approve_resp.json()
                st.toast(f"Price updated to £{res_json['new_price']:.2f} ({res_json['status']})")
                st.success(f"{res_json['status']}: £{res_json['old_price']:.2f} → £{res_json['new_price']:.2f} ({res_json['change_pct']:+.1f}%)")
                st.cache_data.clear()
            else:
                st.error(approve_resp.json().get("detail", "Approval blocked by guardrails."))

    with col_right:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Simulated Price", f"£{sim['proposed_price']:.2f}", f"{((sim['proposed_price'] - base['price']) / base['price'] * 100):+.1f}% vs base")
        k2.metric("Projected Demand", f"{sim['projected_demand']:.1f} /wk", f"{sim['projected_demand'] - base['projected_demand']:+.1f} units")
        k3.metric("Projected Profit", f"£{sim['projected_profit']:.2f}", f"{sim['projected_profit'] - base['projected_profit']:+.2f} vs base")
        k4.metric("Gross Margin", f"{sim['gross_margin_pct']:.1f}%", f"Cost: £{sim['unit_cost']:.2f}")

        curve_df = pd.DataFrame(rec_data["curve"])
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_vrect(
            x0=p_min, x1=p_max,
            fillcolor="rgba(34, 197, 94, 0.14)", line_width=0,
            annotation_text="Safe Guardrail Corridor", annotation_position="top left"
        )
        fig.add_trace(
            go.Scatter(x=curve_df["price"], y=curve_df["profit"], name="Weekly Profit (£)", line=dict(color="#2563eb", width=3)),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=curve_df["price"], y=curve_df["demand"], name="Weekly Demand (Units)", line=dict(color="#f59e0b", width=2, dash="dash")),
            secondary_y=True
        )
        fig.add_vline(x=rec["recommended_price"], line_dash="dot", line_color="#16a34a", annotation_text=f"Optimal £{rec['recommended_price']:.2f}")
        fig.add_trace(
            go.Scatter(x=[sim["proposed_price"]], y=[sim["projected_profit"]], mode="markers", name="Your Slider Price", marker=dict(size=12, color="#dc2626", symbol="diamond")),
            secondary_y=False
        )

        fig.update_layout(
            title=f"Price-Demand & Profit Response Curve — {stock_code}",
            xaxis_title="Unit Price (£)",
            hovermode="x unified",
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="Projected Weekly Profit (£)", secondary_y=False)
        fig.update_yaxes(title_text="Projected Demand (Units)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------------
# TAB 2: PORTFOLIO OPPORTUNITY QUEUE
# ---------------------------------------------------------------------------------
with tab2:
    st.subheader("Top Profit-Lift Opportunities Across Catalog")
    c_obj, c_ref = st.columns([3, 1])
    with c_obj:
        port_obj = st.selectbox("Ranking Objective", ["max_profit", "max_revenue"], key="port_obj")
    with c_ref:
        st.write("")
        if st.button("🔄 Refresh Portfolio Scan", use_container_width=True):
            st.cache_data.clear()

    opps = fetch_portfolio(port_obj)
    opp_df = pd.DataFrame(opps)
    st.dataframe(opp_df, use_container_width=True, hide_index=True)

    quick_sku = st.selectbox("Quick-Approve SKU from Queue", options=[o["stock_code"] for o in opps])
    chosen = next(o for o in opps if o["stock_code"] == quick_sku)
    if st.button(f"Approve Recommended £{chosen['recommended_price']:.2f} for {quick_sku}"):
        r = requests.post(
            f"{API_BASE_URL}/pricing/approve",
            json={
                "stock_code": quick_sku,
                "approved_price": chosen["recommended_price"],
                "approver": "Category Manager",
                "reason": "Approved from Portfolio Queue"
            },
            timeout=10
        )
        if r.status_code == 200:
            st.success(f"Approved {quick_sku} at £{chosen['recommended_price']:.2f} ({r.json()['status']})")
            st.cache_data.clear()
        else:
            st.error(r.json().get("detail", "Approval failed."))

# ---------------------------------------------------------------------------------
# TAB 3: GOVERNANCE & AUDIT LOG
# ---------------------------------------------------------------------------------
with tab3:
    st.subheader("Immutable Price Change Audit Log")
    audit_resp = requests.get(f"{API_BASE_URL}/pricing/audit", params={"limit": 100}, timeout=10)
    records = audit_resp.json()["records"]
    if not records:
        st.info("No price approvals recorded yet. Approve a price in Tab 1 or Tab 2 to populate the audit trail.")
    else:
        audit_df = pd.DataFrame(records)
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Audit Log CSV",
            data=audit_df.to_csv(index=False),
            file_name="price_audit_log.csv",
            mime="text/csv"
        )
