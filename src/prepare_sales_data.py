from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "retail_clean_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "retail_sales_dataset.csv"
)

print("=" * 60)
print("RETAILPULSE - SALES DATA PREPARATION")
print("=" * 60)

df = pd.read_csv(INPUT_PATH, low_memory=False)

print(f"\nRecords before preparation: {len(df):,}")

df["invoicedate"] = pd.to_datetime(
    df["invoicedate"], errors="coerce"
)

df["quantity"] = pd.to_numeric(
    df["quantity"], errors="coerce"
)

df["price"] = pd.to_numeric(
    df["price"], errors="coerce"
)

sales = df[
    (~df["is_cancellation"].astype(bool))
    & (df["quantity"] > 0)
    & (df["price"] > 0)
    & (df["invoicedate"].notna())
].copy()

sales["revenue"] = sales["quantity"] * sales["price"]

sales["year"] = sales["invoicedate"].dt.year
sales["month"] = sales["invoicedate"].dt.month
sales["year_month"] = sales["invoicedate"].dt.to_period("M").astype(str)
sales["date"] = sales["invoicedate"].dt.date.astype(str)

sales.to_csv(OUTPUT_PATH, index=False)

print("\nSALES DATA SUMMARY")
print("-" * 40)

print(f"Valid sales records: {len(sales):,}")
print(f"Unique invoices: {sales['invoice'].nunique():,}")
print(f"Unique products: {sales['stockcode'].nunique():,}")
print(f"Transactions with customer IDs: {sales['customer_id'].notna().sum():,}")
print(f"Transactions without customer IDs: {sales['customer_id'].isna().sum():,}")
print(f"Total quantity sold: {sales['quantity'].sum():,.0f}")
print(f"Total gross sales value: {sales['revenue'].sum():,.2f}")

print(f"\nSales dataset saved to:\n{OUTPUT_PATH}")
print(f"File size: {OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MB")

print("\nSales data preparation completed!")