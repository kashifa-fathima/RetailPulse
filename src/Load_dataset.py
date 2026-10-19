from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "online_retail_II.xlsx"

print("=" * 60)
print("RETAILPULSE - DATASET INSPECTION")
print("=" * 60)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}"
    )

print(f"\nDataset found: {DATASET_PATH.name}")
print(f"File size: {DATASET_PATH.stat().st_size / (1024 * 1024):.2f} MB")

print("\nLoading dataset...")

df = pd.read_excel(DATASET_PATH)

print("\n1. DATASET SHAPE")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\n2. COLUMN NAMES")
print(df.columns.tolist())

print("\n3. FIRST FIVE ROWS")
print(df.head().to_string())

print("\n4. DATA TYPES")
print(df.dtypes)

print("\n5. MISSING VALUES")
print(df.isnull().sum())

print("\n6. DUPLICATE ROWS")
print(f"Duplicate rows: {df.duplicated().sum():,}")

print("\nDataset inspection completed!")
