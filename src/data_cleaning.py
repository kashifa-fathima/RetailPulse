from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_DIR / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "retail_clean_dataset.csv"

print("=" * 60)
print("RETAILPULSE - DATA CLEANING")
print("=" * 60)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(RAW_PATH)

print(f"\nOriginal rows: {len(df):,}")

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

duplicates = df.duplicated().sum()
df = df.drop_duplicates()

print(f"Duplicate rows removed: {duplicates:,}")

missing_descriptions = df["description"].isna().sum()
df = df.dropna(subset=["description"])

print(f"Rows removed for missing descriptions: {missing_descriptions:,}")

df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")

before = len(df)

df = df.dropna(
    subset=[
        "invoice",
        "stockcode",
        "quantity",
        "invoicedate",
        "price",
        "country",
    ]
)

print(f"Rows removed for missing essential fields: {before - len(df):,}")

df["invoice"] = df["invoice"].astype(str).str.strip()

df["is_cancellation"] = df["invoice"].str.upper().str.startswith("C")

df["revenue"] = df["quantity"] * df["price"]

df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)

print(f"Original dataset rows: {525461:,}")
print(f"Rows after cleaning: {len(df):,}")
print(f"Missing customer IDs: {df['customer_id'].isna().sum():,}")
print(f"Cancellation records: {df['is_cancellation'].sum():,}")
print(f"Negative quantity records: {(df['quantity'] < 0).sum():,}")
print(f"Zero or negative price records: {(df['price'] <= 0).sum():,}")
print(f"Remaining duplicate rows: {df.duplicated().sum():,}")

print(f"\nCleaned dataset saved to:\n{OUTPUT_PATH}")
print(f"Output file size: {OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MB")
