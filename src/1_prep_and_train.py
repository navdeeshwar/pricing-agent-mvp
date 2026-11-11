"""
===================================================================================
MODULE 1: DATA PREPARATION & MACHINE LEARNING DEMAND MODELING
File: src/1_prep_and_train.py
===================================================================================
WHAT THIS SCRIPT DOES:
1. Loads raw e-commerce transaction data (over 1 million rows of UK retail purchases).
2. Cleans the data by dropping cancelled orders, test fees, and bulk wholesale noise.
3. Groups individual shopping cart items into weekly sales per product so our model
   can observe how sales volume goes up or down when prices change (Price Elasticity).
4. Trains a LightGBM Machine Learning model to forecast weekly demand given price & season.
5. Exports the trained model and product catalog into the 'models/' folder so our 
   pricing solver and API can use them without re-reading the 100MB CSV every time.
===================================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------------------------------------------------------------------------
# STEP 1: CLEANING THE RAW TRANSACTION DATA
# ---------------------------------------------------------------------------------
def load_and_clean_data(file_path):
    """
    Cleans raw customer transaction logs.
    In real-world data, logs contain refunds, system tests, and damaged item write-offs.
    We remove these so our AI model only learns from genuine, successful retail purchases.
    """
    print("Step 1: Loading raw Online Retail II dataset...")
    
    # ISO-8859-1 encoding is standard for older UK / European retail databases 
    # to handle currency symbols like the British Pound (£).
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    initial_rows = len(df)
    print(f"-> Loaded {initial_rows:,} raw rows.")

    # 1. Filter out Cancellations & Returns:
    # In this dataset, invoice numbers starting with 'C' indicate a cancelled order or return.
    # The '~' symbol means NOT. So this keeps only invoices that DO NOT start with 'C'.
    # If we kept these, they would show up as negative demand and corrupt our training.
    df = df[~df['Invoice'].astype(str).str.startswith('C')]

    # 2. Keep only positive quantities and prices:
    # A quantity <= 0 or price <= 0 represents warehouse adjustments, broken stock write-offs,
    # or accounting errors rather than a customer willingly paying for an item.
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

    # 3. Drop non-product service charges:
    # 'POST' is postage/shipping, 'BANK CHARGES' are banking fees, 'M' is manual fee adjustments.
    # These are not physical retail products, so we should never try to predict consumer demand for them.
    non_product_codes = ['POST', 'M', 'BANK CHARGES', 'DOT', 'CRUK', 'PADS']
    df = df[~df['StockCode'].isin(non_product_codes)]

    # 4. Outlier Clipping (Handling extreme abnormal values):
    # An 'outlier' is a data point that is drastically higher or lower than the rest.
    # In this store, normal retail consumers buy 1 to 20 units. But occasionally, a commercial
    # B2B (business-to-business) buyer places a bulk order for 10,000 units in a single click.
    # If we left those giant bulk orders in, the AI would falsely believe that a tiny 5% price 
    # discount causes 10,000 units to sell! We clip at the 99th percentile (retaining the bottom 99%).
    q99_qty = df['Quantity'].quantile(0.99)
    df = df[df['Quantity'] <= q99_qty]

    print(f"-> Cleaned dataset retained {len(df):,} rows (dropped {initial_rows - len(df):,} noisy/bulk rows).")
    return df


# ---------------------------------------------------------------------------------
# STEP 2: AGGREGATING RAW TRANSACTIONS INTO WEEKLY DEMAND BUCKETS
# ---------------------------------------------------------------------------------
def aggregate_demand_features(df, top_n_products=100):
    """
    Transforms individual cart checkout lines into structured weekly sales per product.
    
    Why aggregate by week?
    Individual cart purchases happen at random seconds and are noisy. By summing sales 
    into steady weekly buckets (Monday to Sunday), we can cleanly observe the economic curve:
    'When this product was priced at £1.00 in Week 12, we sold 200 units. 
     When it was discounted to £0.85 in Week 13, we sold 260 units.'
    """
    print(f"\nStep 2: Aggregating weekly demand for top {top_n_products} products...")
    
    # Convert string dates (e.g. '2009-12-01 07:45') into proper datetime objects
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Focus on the Top 100 Products by sales volume:
    # Products that only sold 2 or 3 times in two years don't have enough history to learn a curve.
    # Focusing on the top 100 high-volume bestsellers ensures reliable statistical data.
    top_stock_codes = df.groupby('StockCode')['Quantity'].sum().nlargest(top_n_products).index
    df_top = df[df['StockCode'].isin(top_stock_codes)].copy()
    
    # Calculate revenue for each individual transaction line: Price × Quantity
    df_top['Revenue'] = df_top['Quantity'] * df_top['Price']

    # Group by Product (StockCode) and Weekly Calendar Bucket ('W-MON' = Week starting Monday):
    weekly = df_top.groupby([
        'StockCode',
        pd.Grouper(key='InvoiceDate', freq='W-MON')
    ]).agg(
        total_quantity=('Quantity', 'sum'),       # Total units sold that week
        total_revenue=('Revenue', 'sum'),         # Total money collected that week
        transaction_count=('Invoice', 'nunique'), # Number of distinct orders that week
        description=('Description', 'first')      # Keep the human-readable product title
    ).reset_index()

    # Calculate Volume-Weighted Average Price (VWAP):
    # Why? If you sell 100 units at £1.00 and 1 unit at £2.00 in the same week, a simple 
    # average would say the price was £1.50 (which is misleading).
    # Total Revenue (£102) divided by Total Quantity (101) gives the true effective price: £1.01.
    weekly['price'] = weekly['total_revenue'] / weekly['total_quantity']
    
    # Remove any empty weeks with 0 sales or 0 price
    weekly = weekly[(weekly['total_quantity'] > 0) & (weekly['price'] > 0)]

    # Feature Engineering (creating extra signals to help the AI learn patterns):
    # Calendar Seasonality: E-commerce sales spike in November/December for the holidays.
    # Giving the model month, week of year, and quarter allows it to distinguish holiday spikes 
    # from price discount spikes.
    weekly['month'] = weekly['InvoiceDate'].dt.month
    weekly['week_of_year'] = weekly['InvoiceDate'].dt.isocalendar().week.astype(int)
    weekly['quarter'] = weekly['InvoiceDate'].dt.quarter

    # Reference Price (Baseline Price):
    # Find the historical median (middle) price for each product.
    ref_prices = weekly.groupby('StockCode')['price'].median().rename('ref_price')
    weekly = weekly.merge(ref_prices, on='StockCode', how='left')
    
    # Price Ratio (Relative Discount / Premium):
    # A £0.50 price change is massive for a £1.00 item (+50%), but tiny for a £50 item (+1%).
    # Dividing the current price by the reference price (price / ref_price) gives a normalized ratio:
    # e.g., 0.90 means a 10% discount, 1.15 means a 15% price markup.
    weekly['price_ratio'] = weekly['price'] / weekly['ref_price']

    # Tell LightGBM to treat StockCode as a categorical variable:
    # Unlike old models that need One-Hot Encoding (creating 100 separate 0/1 columns),
    # LightGBM natively understands categorical IDs using fast integer bins.
    weekly['StockCode'] = weekly['StockCode'].astype('category')

    # Build a Product Catalog metadata dictionary for our downstream solver:
    # Stores each product's base price, estimated unit cost, and historical boundaries.
    catalog = {}
    grouped_desc = weekly.groupby('StockCode')
    for code, group in grouped_desc:
        median_price = float(group['price'].median())
        catalog[str(code)] = {
            "stock_code": str(code),
            "description": str(group['description'].dropna().iloc[0] if len(group['description'].dropna()) > 0 else code),
            "base_price": round(median_price, 2),
            # In this public dataset, supplier cost is not provided. 
            # We assume a realistic 60% baseline cost of goods (giving a 40% starting gross margin)
            # so our solver can calculate actual dollar profits: Profit = (Price - Cost) * Quantity.
            "estimated_cost": round(median_price * 0.60, 2),
            "min_observed_price": round(float(group['price'].min()), 2),
            "max_observed_price": round(float(group['price'].max()), 2),
            "historical_mean_weekly_qty": round(float(group['total_quantity'].mean()), 1)
        }

    print(f"-> Generated {len(weekly):,} weekly demand observation points across {len(catalog)} products.")
    return weekly, catalog


# ---------------------------------------------------------------------------------
# STEP 3: TRAINING THE LIGHTGBM MACHINE LEARNING DEMAND MODEL
# ---------------------------------------------------------------------------------
def train_demand_model(weekly_df):
    """
    Trains a Gradient Boosted Decision Tree (LightGBM) to predict weekly demand.
    
    How LightGBM works:
    It builds an ensemble of decision trees. Each tree asks a series of yes/no questions 
    (e.g., 'Is the price ratio < 0.90? Is the month == 11? Is the StockCode == 15036?').
    Each new tree fixes the prediction errors made by the previous trees.
    """
    print("\nStep 3: Training LightGBM Demand / Elasticity Model...")
    
    # Features (Inputs given to the model)
    feature_cols = ['StockCode', 'price', 'ref_price', 'price_ratio', 'month', 'week_of_year', 'quarter']
    # Target (What the model is trying to predict)
    target_col = 'total_quantity'

    # Out-of-Time Validation Split:
    # In time series (sales over time), you must NEVER randomly shuffle train and test data.
    # If you shuffle randomly, the model gets to peek into future weeks to predict the past (Data Leakage).
    # Instead, we hold out the final 12 weeks of historical data to test how well the model predicts the future.
    split_date = weekly_df['InvoiceDate'].max() - pd.Timedelta(weeks=12)
    train_df = weekly_df[weekly_df['InvoiceDate'] < split_date]
    test_df = weekly_df[weekly_df['InvoiceDate'] >= split_date]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Initialize the LightGBM Regressor:
    # - n_estimators=150: Build 150 decision trees in sequence.
    # - learning_rate=0.05: Each tree contributes 5% to the final prediction (prevents overfitting).
    # - num_leaves=31: Maximum complexity per tree (standard sweet spot for tabular data).
    # - random_state=42: Ensures reproducible, identical results every time you run it.
    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    # Train the model on the historical training data
    model.fit(X_train, y_train)

    # Evaluate on the unseen final 12 weeks of test data:
    y_pred = model.predict(X_test)
    
    # MAE (Mean Absolute Error): On average, how many units off is our weekly forecast?
    mae = mean_absolute_error(y_test, y_pred)
    # R² (R-squared / Coefficient of Determination): Measures how much of the variance in sales 
    # the model explains compared to simply guessing the average sales every week.
    r2 = r2_score(y_test, y_pred)
    print(f"-> Validation MAE: {mae:.2f} units/week | R² Score: {r2:.3f}")

    return model, feature_cols


# ---------------------------------------------------------------------------------
# STEP 4: SAVING MODEL ARTIFACTS FOR THE SOLVER & API
# ---------------------------------------------------------------------------------
def save_artifacts(model, catalog, feature_cols, output_dir="models"):
    """
    Saves our trained model and product metadata to disk so our API and optimization 
    solver can load them instantaneously without needing to re-process the raw CSV data.
    """
    print(f"\nStep 4: Saving model artifacts to '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)

    # Save the binary machine learning model weights
    joblib.dump(model, os.path.join(output_dir, "demand_model.joblib"))
    
    # Save the product catalog (costs, base prices, descriptions) as a JSON dictionary
    with open(os.path.join(output_dir, "product_catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2)
        
    # Save the exact column order expected by the model
    with open(os.path.join(output_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"-> Successfully saved:\n   - {output_dir}/demand_model.joblib\n   - {output_dir}/product_catalog.json\n   - {output_dir}/feature_cols.json")


# ---------------------------------------------------------------------------------
# MAIN EXECUTION ENTRY POINT
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    # Resolve absolute paths so this script works regardless of which directory you run it from
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "online_retail_II.csv")
    models_dir = os.path.join(project_root, "models")

    # Run the 4-step pipeline in order:
    # 1. Clean the raw data
    df_cleaned = load_and_clean_data(data_path)
    # 2. Aggregate transactions into weekly demand and build product catalog
    df_weekly, catalog = aggregate_demand_features(df_cleaned, top_n_products=100)
    # 3. Train the LightGBM demand forecasting model
    model, features = train_demand_model(df_weekly)
    # 4. Save artifacts to models/ for the solver and API
    save_artifacts(model, catalog, features, output_dir=models_dir)
