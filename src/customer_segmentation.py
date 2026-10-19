from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "retail_sales_dataset.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
CHART_DIR = PROJECT_DIR / "reports" / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "customer_segments.csv"

print("=" * 60)
print("RETAILPULSE - CUSTOMER SEGMENTATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH, low_memory=False)

df["invoicedate"] = pd.to_datetime(
    df["invoicedate"], errors="coerce"
)

df["customer_id"] = pd.to_numeric(
    df["customer_id"], errors="coerce"
)

df["revenue"] = df["quantity"] * df["price"]

df = df.dropna(
    subset=["customer_id", "invoicedate", "invoice", "revenue"]
).copy()

df = df[
    ~df["invoice"].astype(str).str.strip().str.upper().str.startswith("C")
].copy()

df = df[
    (df["quantity"] > 0)
    & (df["price"] > 0)
].copy()

df["customer_id"] = df["customer_id"].astype("int64")

print(f"\nSales records used: {len(df):,}")
print(f"Unique customers: {df['customer_id'].nunique():,}")


reference_date = df["invoicedate"].max().normalize() + pd.Timedelta(days=1)

rfm = (
    df.groupby("customer_id")
    .agg(
        last_purchase=("invoicedate", "max"),
        frequency=("invoice", "nunique"),
        monetary=("revenue", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (
    reference_date - rfm["last_purchase"].dt.normalize()
).dt.days

rfm = rfm[
    (rfm["frequency"] > 0)
    & (rfm["monetary"] > 0)
].copy()

print("\nRFM FEATURES CREATED")
print(f"Customers available for clustering: {len(rfm):,}")

print("\nRFM SUMMARY")
print(
    rfm[["recency", "frequency", "monetary"]]
    .describe()
    .round(2)
    .to_string()
)


features = rfm[["recency", "frequency", "monetary"]].copy()

features["frequency"] = (
    features["frequency"].astype(float).map(lambda x: __import__("math").log1p(x))
)

features["monetary"] = (
    features["monetary"].astype(float).map(lambda x: __import__("math").log1p(x))
)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10,
)

rfm["cluster"] = model.fit_predict(scaled_features)

print("\nCUSTOMER CLUSTER COUNTS")
print(
    rfm["cluster"]
    .value_counts()
    .sort_index()
    .to_string()
)


cluster_summary = (
    rfm.groupby("cluster")
    .agg(
        customers=("customer_id", "count"),
        avg_recency_days=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        total_monetary=("monetary", "sum"),
    )
    .round(2)
)

print("\nCLUSTER SUMMARY")
print(cluster_summary.to_string())


rfm["last_purchase"] = rfm["last_purchase"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

rfm.to_csv(OUTPUT_PATH, index=False)

print(f"\nCustomer segments saved to:\n{OUTPUT_PATH}")


plt.figure(figsize=(9, 6))

plt.bar(
    cluster_summary.index.astype(str),
    cluster_summary["customers"],
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()

chart_path = CHART_DIR / "customer_segments.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print(f"Customer segment chart saved to:\n{chart_path}")

print("\nCustomer segmentation completed successfully!")