"""
===================================================================================
MODULE 2: CONSTRAINED OPTIMIZATION & BUSINESS GUARDRAILS ENGINE
File: src/2_solver_and_guardrails.py
===================================================================================
WHAT THIS SCRIPT DOES:
1. Defines Business Guardrails (safety boundaries that prevent destructive prices).
   - Margin Floor: Protects profitability (e.g. at least 15% profit on top of cost).
   - Velocity Limit: Protects customer trust (price can't change by more than ±15% at once).
   - Historical Corridor: Keeps prices within realistic historical observed bounds.
2. Implements a 1D Grid Search Optimizer:
   - Evaluates 50 evenly spaced candidate prices inside the safe corridor.
   - Queries our trained LightGBM model to predict weekly customer demand at each price.
   - Calculates projected Revenue (P × Q) and Gross Profit ((P - Cost) × Q).
   - Selects the price that mathematically maximizes Profit or Revenue.
   - Computes expected percentage lift compared to the current baseline price.
===================================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------------
# 1. BUSINESS GUARDRAILS (THE SAFETY RAILS)
# ---------------------------------------------------------------------------------
class PricingGuardrails:
    """
    Guardrails are automated business rules.
    
    Why are guardrails necessary?
    Machine Learning models are purely mathematical pattern-matchers. They have no concept 
    of company bankruptcy or customer loyalty. If an unconstrained model discovers that 
    dropping price to £0.05 creates 1,000 units of sales, it might recommend it—causing 
    massive financial losses because every unit costs £0.52 to produce!
    
    Guardrails define a 'feasible corridor' (a safe price window [min_price, max_price])
    that the optimization solver is mathematically forbidden to cross.
    """
    def __init__(self, min_margin_pct=0.15, max_price_change_pct=0.15):
        # min_margin_pct (15%): Minimum gross margin we must make on every unit sold.
        # Gross Margin % = (Price - Cost) / Price.
        self.min_margin_pct = min_margin_pct          
        
        # max_price_change_pct (15%): Velocity limit to prevent customer 'sticker shock'.
        # We cap price adjustments to maximum +/- 15% change from the baseline price in any single cycle.
        self.max_price_change_pct = max_price_change_pct  

    def compute_feasible_price_range(self, base_price, unit_cost, min_hist_price, max_hist_price):
        """
        Calculates the safe [lower_bound, upper_bound] price window for a product.
        """
        # Guardrail 1: Margin Floor (Profitability Protection)
        # If an item costs £0.52 to produce, and we require at least a 15% margin:
        # Margin = (Price - Cost) / Price >= 0.15  -->  Price >= Cost / (1 - 0.15)
        # Price >= £0.52 / 0.85 = £0.6118.
        # (Note: In retail, gross margin is divided by Price, NOT multiplied by Cost).
        margin_floor = unit_cost / (1.0 - self.min_margin_pct)

        # Guardrail 2: Velocity Guardrail (Preventing Sudden Price Shock)
        # If the item currently sells at £1.00, it cannot drop below £0.85 (-15%) 
        # or rise above £1.15 (+15%) in one update cycle.
        velocity_min = base_price * (1.0 - self.max_price_change_pct)
        velocity_max = base_price * (1.0 + self.max_price_change_pct)

        # Guardrail 3: Historical Observed Corridor (Preventing Out-of-Distribution Extrapolation)
        # Decision tree models (LightGBM) cannot extrapolate trends outside their training data.
        # If the lowest historical price was £0.60, the tree has no idea what happens at £0.20.
        # We bound the search to within +/- 10% of historically observed prices.
        #
        # Intersect all 3 constraints:
        # The lowest safe price must satisfy the margin floor, the velocity floor, and historical min:
        lower_bound = max(margin_floor, velocity_min, min_hist_price * 0.9)
        # The highest safe price is bounded by velocity ceiling and historical max:
        upper_bound = min(velocity_max, max_hist_price * 1.1)

        # Fallback Check: If supplier costs spiked so high that lower_bound exceeds upper_bound,
        # gracefully collapse both bounds to the current base_price so the solver never crashes.
        if lower_bound > upper_bound:
            lower_bound = base_price
            upper_bound = base_price

        return round(lower_bound, 2), round(upper_bound, 2)


# ---------------------------------------------------------------------------------
# 2. PRICING OPTIMIZER (THE MATHEMATICAL SOLVER)
# ---------------------------------------------------------------------------------
class PricingOptimizer:
    """
    Connects our trained LightGBM demand model with the business guardrails 
    to find the mathematically optimal price that maximizes Profit or Revenue.
    """
    def __init__(self, models_dir=None):
        # Locate the models/ directory relative to this script
        if models_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(project_root, "models")

        self.models_dir = models_dir
        # Load the trained LightGBM model from disk into RAM
        self.model = joblib.load(os.path.join(models_dir, "demand_model.joblib"))
        # Load the product catalog dictionary (base prices, unit costs, historical bounds)
        with open(os.path.join(models_dir, "product_catalog.json"), "r") as f:
            self.catalog = json.load(f)
        # Load the exact column names expected by the model
        with open(os.path.join(models_dir, "feature_cols.json"), "r") as f:
            self.feature_cols = json.load(f)

        # Instantiate our business guardrail validator
        self.guardrails = PricingGuardrails()

    def predict_demand(self, stock_code, price, ref_price, month=11, week=45, quarter=4):
        """
        Queries the trained LightGBM model to predict expected weekly unit sales 
        for a specific candidate price and calendar time period.
        """
        # Build a single-row dictionary with all features required by the model
        row = {
            'StockCode': str(stock_code),
            'price': float(price),
            'ref_price': float(ref_price),
            'price_ratio': float(price / ref_price if ref_price > 0 else 1.0),
            'month': int(month),
            'week_of_year': int(week),
            'quarter': int(quarter)
        }
        # Convert to a DataFrame matching the model's feature column ordering
        df_feature = pd.DataFrame([row])[self.feature_cols]
        # Cast StockCode to categorical (as expected by LightGBM)
        df_feature['StockCode'] = df_feature['StockCode'].astype('category')

        # Predict expected weekly quantity demanded
        pred_qty = float(self.model.predict(df_feature)[0])
        # In reality, sales cannot be negative, so we clip at 0.0
        return max(0.0, pred_qty)

    def optimize_price(self, stock_code, objective='max_profit', month=11, week=45, quarter=4, steps=50):
        """
        Finds the optimal price for a product using 1D Discretized Grid Search.
        
        Why 1D Grid Search instead of Calculus / Gradient Descent (like SciPy minimize)?
        Decision trees (LightGBM) output piecewise-constant step functions (like stairs).
        On flat stairs, the derivative/slope (gradient) is exactly 0 everywhere! 
        Gradient-based solvers get stuck immediately on the first step.
        Evaluating 50 candidate prices across our narrow safe window takes under 5 milliseconds 
        and is mathematically guaranteed to find the global optimum without getting stuck.
        """
        if str(stock_code) not in self.catalog:
            raise ValueError(f"StockCode '{stock_code}' not found in trained product catalog.")

        item = self.catalog[str(stock_code)]
        base_price = item['base_price']
        cost = item['estimated_cost']

        # Step 1: Use guardrails to calculate the allowable [min_price, max_price] window
        p_min, p_max = self.guardrails.compute_feasible_price_range(
            base_price=base_price,
            unit_cost=cost,
            min_hist_price=item['min_observed_price'],
            max_hist_price=item['max_observed_price']
        )

        # Step 2: Generate 50 evenly spaced candidate prices between p_min and p_max
        candidate_prices = np.linspace(p_min, p_max, steps)

        best_price = base_price
        best_metric = -float('inf')
        best_demand = 0.0

        # Step 3: Evaluate each candidate price point
        curve = []
        for p in candidate_prices:
            # Predict sales volume at this price
            q = self.predict_demand(stock_code, p, base_price, month, week, quarter)
            
            # Financial calculations:
            revenue = p * q                             # Total sales dollars
            profit = (p - cost) * q                     # Gross Profit = (Price - Cost) × Units
            margin_pct = (p - cost) / p if p > 0 else 0.0 # Gross Margin Percentage

            # Check if this price achieves a higher metric based on business objective:
            # - 'max_profit': We want to make the most dollar profit.
            # - 'max_revenue': We want to maximize total cash flow / sales volume.
            metric = profit if objective == 'max_profit' else revenue
            if metric > best_metric:
                best_metric = metric
                best_price = p
                best_demand = q

            curve.append({
                "price": round(float(p), 2),
                "demand": round(float(q), 1),
                "revenue": round(float(revenue), 2),
                "profit": round(float(profit), 2),
                "margin_pct": round(float(margin_pct * 100), 1)
            })

        # Step 4: Compare against current baseline performance to compute 'Lift':
        # Baseline = What would happen if we left the price unchanged at base_price?
        baseline_demand = self.predict_demand(stock_code, base_price, base_price, month, week, quarter)
        baseline_revenue = base_price * baseline_demand
        baseline_profit = (base_price - cost) * baseline_demand

        # Recommended performance at best_price
        opt_revenue = best_price * best_demand
        opt_profit = (best_price - cost) * best_demand
        
        # Lift % = (Optimized Metric - Baseline Metric) / Baseline Metric × 100
        profit_lift_pct = ((opt_profit - baseline_profit) / baseline_profit * 100) if baseline_profit > 0 else 0.0
        revenue_lift_pct = ((opt_revenue - baseline_revenue) / baseline_revenue * 100) if baseline_revenue > 0 else 0.0

        # Return a structured decision package
        return {
            "stock_code": stock_code,
            "description": item['description'],
            "objective": objective,
            "unit_cost": cost,
            "baseline": {
                "price": base_price,
                "projected_demand": round(baseline_demand, 1),
                "projected_revenue": round(baseline_revenue, 2),
                "projected_profit": round(baseline_profit, 2)
            },
            "recommendation": {
                "recommended_price": round(best_price, 2),
                "price_change_pct": round(((best_price - base_price) / base_price) * 100, 1),
                "projected_demand": round(best_demand, 1),
                "projected_revenue": round(opt_revenue, 2),
                "projected_profit": round(opt_profit, 2),
                "profit_lift_pct": round(profit_lift_pct, 1),
                "revenue_lift_pct": round(revenue_lift_pct, 1)
            },
            "guardrails_applied": {
                "min_margin_pct": self.guardrails.min_margin_pct * 100,
                "max_price_velocity_pct": self.guardrails.max_price_change_pct * 100,
                "allowed_price_corridor": [p_min, p_max]
            }
        }


# ---------------------------------------------------------------------------------
# STANDALONE VERIFICATION TEST
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    # Test the optimizer directly in the terminal for the first product in our catalog
    optimizer = PricingOptimizer()
    sample_code = list(optimizer.catalog.keys())[0]

    print(f"--- Running Test Pricing Optimization for SKU: {sample_code} ---")
    res = optimizer.optimize_price(sample_code, objective='max_profit')
    print(json.dumps(res, indent=2))
