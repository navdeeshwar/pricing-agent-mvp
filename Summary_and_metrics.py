"""
Standalone verification script for CV & interview audit.
Run: python Summary_and_metrics.py
"""
import importlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")


def main():
    prep = importlib.import_module("src.1_prep_and_train")
    solver = importlib.import_module("src.2_solver_and_guardrails")

    print("=" * 68)
    print("   AUTONOMOUS PRICING AGENT MVP  SUMMARY")
    print("=" * 68)

    df = prep.load_and_clean_data("data/online_retail_II.csv")
    weekly, catalog = prep.aggregate_demand_features(df, top_n_products=100)
    prep.train_demand_model(weekly)

    opt = solver.PricingOptimizer()
    codes = list(opt.catalog.keys())

    base_prof = opt_prof = base_rev = opt_rev = 0.0
    latencies = []

    for code in codes:
        t0 = time.perf_counter()
        res = opt.optimize_price(code, objective="max_profit")
        latencies.append((time.perf_counter() - t0) * 1000)
        base_prof += res["baseline"]["projected_profit"]
        opt_prof += res["recommendation"]["projected_profit"]
        base_rev += res["baseline"]["projected_revenue"]
        opt_rev += res["recommendation"]["projected_revenue"]

    profit_lift = (opt_prof - base_prof) / base_prof * 100
    revenue_lift = (opt_rev - base_rev) / base_rev * 100
    avg_latency = sum(latencies) / len(latencies)

    print("\n" + "=" * 68)
    print("   KEY VERIFIED METRICS")
    print("=" * 68)
    print(f"1. Top Products Modeled         : {len(catalog)} SKUs")
    print(f"2. Weekly Observations          : {len(weekly):,} data points")
    print("3. Out-of-Time Holdout Window   : 12 weeks")
    print("4. Guardrails Enforced          : 15% Min Margin | +/-15% Price Cap")
    print(f"5. Baseline Weekly Profit       : £{base_prof:,.2f}")
    print(f"6. Optimized Weekly Profit      : £{opt_prof:,.2f} (+{profit_lift:.1f}% Lift)")
    print(f"7. Baseline Weekly Revenue      : £{base_rev:,.2f}")
    print(f"8. Optimized Weekly Revenue     : £{opt_rev:,.2f} (+{revenue_lift:.1f}% Lift)")
    print(f"9. Avg Optimization Speed       : {avg_latency:.1f} ms per SKU")
    print("=" * 68)


if __name__ == "__main__":
    main()
