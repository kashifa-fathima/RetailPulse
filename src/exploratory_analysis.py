from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "retail_sales_dataset.csv"
)

CHART_DIR = PROJECT_DIR / "reports" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("RETAILPULSE - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_PATH, low_memory=False)

df["invoicedate"] = pd.to_datetime(df["invoicedate"])
df["revenue"] = df["quantity"] * df["price"]

print("\n1. OVERALL SALES SUMMARY")
print("-" * 40)

print(f"Sales records: {len(df):,}")
print(f"Total gross sales value: {df['revenue'].sum():,.2f}")
print(f"Total quantity sold: {df['quantity'].sum():,.0f}")
print(f"Unique invoices: {df['invoice'].nunique():,}")
print(f"Unique products: {df['stockcode'].nunique():,}")
print(f"Unique customers: {df['customer_id'].nunique():,}")
print(f"Countries: {df['country'].nunique():,}")

df["year_month"] = df["invoicedate"].dt.to_period("M").astype(str)

monthly_sales = (
    df.groupby("year_month")["revenue"]
    .sum()
    .sort_index()
)

print("\n2. MONTHLY SALES")
print(monthly_sales.to_string())

plt.figure(figsize=(12, 6))
monthly_sales.plot(marker="o")
plt.title("Monthly Gross Sales")
plt.xlabel("Month")
plt.ylabel("Gross Sales Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(CHART_DIR / "monthly_sales.png", dpi=150)
plt.close()

top_products = (
    df.groupby("description")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n3. TOP 10 PRODUCTS BY GROSS SALES VALUE")
print(top_products.to_string())

plt.figure(figsize=(12, 7))
sns.barplot(
    x=top_products.values,
    y=top_products.index
)
plt.title("Top 10 Products by Gross Sales Value")
plt.xlabel("Gross Sales Value")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig(CHART_DIR / "top_products.png", dpi=150)
plt.close()

top_countries = (
    df.groupby("country")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n4. TOP 10 COUNTRIES BY GROSS SALES VALUE")
print(top_countries.to_string())

plt.figure(figsize=(12, 7))
sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)
plt.title("Top 10 Countries by Gross Sales Value")
plt.xlabel("Gross Sales Value")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(CHART_DIR / "top_countries.png", dpi=150)
plt.close()

df["date"] = df["invoicedate"].dt.date

daily_sales = df.groupby("date")["revenue"].sum()

plt.figure(figsize=(14, 6))
daily_sales.plot()
plt.title("Daily Gross Sales Trend")
plt.xlabel("Date")
plt.ylabel("Gross Sales Value")
plt.tight_layout()
plt.savefig(CHART_DIR / "daily_sales.png", dpi=150)
plt.close()

print("\n5. CHARTS CREATED")
print(f"Chart folder: {CHART_DIR}")

for chart in sorted(CHART_DIR.glob("*.png")):
    print(f"- {chart.name}")

print("\nExploratory data analysis completed successfully!")