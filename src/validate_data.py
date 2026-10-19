from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "retail_clean_dataset.csv"
)

print("=" * 60)
print("RETAILPULSE - TRANSACTION VALIDATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH, low_memory=False)

df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

cancelled = df[df["is_cancellation"] == True]

print("\n1. CANCELLATION RECORDS")
print(f"Total cancellation records: {len(cancelled):,}")
print(f"Negative quantities in cancellations: {(cancelled['quantity'] < 0).sum():,}")
print(f"Positive quantities in cancellations: {(cancelled['quantity'] > 0).sum():,}")

print("\nSample cancellation records:")
print(
    cancelled[
        ["invoice", "description", "quantity", "price", "invoicedate"]
    ].head(10).to_string(index=False)
)

negative_qty = df[df["quantity"] < 0]

print("\n2. NEGATIVE QUANTITIES")
print(f"Total records: {len(negative_qty):,}")

print(
    negative_qty[
        ["invoice", "description", "quantity", "price", "is_cancellation"]
    ].head(10).to_string(index=False)
)

invalid_prices = df[df["price"] <= 0]

print("\n3. ZERO OR NEGATIVE PRICES")
print(f"Total records: {len(invalid_prices):,}")

print(
    invalid_prices[
        ["invoice", "description", "quantity", "price"]
    ].head(10).to_string(index=False)
)

print("\n4. DATE RANGE")
print(f"Start date: {df['invoicedate'].min()}")
print(f"End date: {df['invoicedate'].max()}")

print("\n5. MISSING VALUES")
print(df.isna().sum().to_string())

print("\nValidation completed!")