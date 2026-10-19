from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="RetailPulse | Customer Analytics",
    page_icon="📊",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed"

SALES_PATH = DATA_DIR / "retail_sales_dataset.csv"
SEGMENTS_PATH = DATA_DIR / "customer_segments.csv"
FORECAST_PATH = DATA_DIR / "sales_forecast.csv"

@st.cache_data
def load_data():
    sales = pd.read_csv(SALES_PATH, low_memory=False)
    segments = pd.read_csv(SEGMENTS_PATH)
    forecast = pd.read_csv(FORECAST_PATH)

    sales["invoicedate"] = pd.to_datetime(
        sales["invoicedate"], errors="coerce"
    )

    forecast["ds"] = pd.to_datetime(
        forecast["ds"], errors="coerce"
    )

    segments["customer_id"] = pd.to_numeric(
        segments["customer_id"], errors="coerce"
    )

    sales["revenue"] = (
        sales["quantity"] * sales["price"]
    )

    return sales, segments, forecast


st.title("📊 RetailPulse")
st.subheader("AI-Powered Customer Analytics & Demand Forecasting")
st.caption(
    "Explore historical retail sales, customer segments, "
    "and model-generated sales forecasts."
)

required_files = [
    SALES_PATH,
    SEGMENTS_PATH,
    FORECAST_PATH,
]

missing_files = [
    path.name for path in required_files if not path.exists()
]

if missing_files:
    st.error(
        "The following required files are missing: "
        + ", ".join(missing_files)
    )
    st.info(
        "Run the data preparation, customer segmentation, "
        "and demand forecasting scripts first."
    )
    st.stop()

sales, segments, forecast = load_data()

st.sidebar.header("Dashboard Filters")

countries = sorted(
    sales["country"].dropna().astype(str).unique().tolist()
)

selected_countries = st.sidebar.multiselect(
    "Select countries",
    options=countries,
    default=["United Kingdom"] if "United Kingdom" in countries
    else countries[:1],
)

filtered_sales = sales[
    sales["country"].astype(str).isin(selected_countries)
].copy()

st.sidebar.caption(
    "Country filters affect sales charts and sales KPIs."
)

total_sales = filtered_sales["revenue"].sum()

total_orders = filtered_sales["invoice"].nunique()

total_products = filtered_sales["stockcode"].nunique()

total_customers = filtered_sales["customer_id"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Gross Sales", f"{total_sales:,.2f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Unique Products", f"{total_products:,}")
col4.metric("Identified Customers", f"{total_customers:,}")

st.divider()

st.header("📈 Sales Performance")

if filtered_sales.empty:
    st.warning("No sales records match the selected countries.")
else:
    filtered_sales["date"] = (
        filtered_sales["invoicedate"].dt.normalize()
    )

    daily_sales = (
        filtered_sales.groupby("date", as_index=False)["revenue"]
        .sum()
    )

    daily_fig = px.line(
        daily_sales,
        x="date",
        y="revenue",
        title="Daily Gross Sales",
        labels={
            "date": "Date",
            "revenue": "Gross Sales",
        },
    )

    st.plotly_chart(daily_fig, use_container_width=True)

    filtered_sales["month"] = (
        filtered_sales["invoicedate"].dt.to_period("M").astype(str)
    )

    monthly_sales = (
        filtered_sales.groupby("month", as_index=False)["revenue"]
        .sum()
    )

    monthly_fig = px.bar(
        monthly_sales,
        x="month",
        y="revenue",
        title="Monthly Gross Sales",
        labels={
            "month": "Month",
            "revenue": "Gross Sales",
        },
    )

    st.plotly_chart(monthly_fig, use_container_width=True)


st.header("🛍️ Product and Country Insights")

left, right = st.columns(2)

with left:
    top_products = (
        filtered_sales.groupby("description", as_index=False)["revenue"]
        .sum()
        .nlargest(10, "revenue")
        .sort_values("revenue")
    )

    product_fig = px.bar(
        top_products,
        x="revenue",
        y="description",
        orientation="h",
        title="Top 10 Products by Gross Sales",
        labels={
            "revenue": "Gross Sales",
            "description": "Product",
        },
    )

    st.plotly_chart(product_fig, use_container_width=True)

with right:
    country_sales = (
        filtered_sales.groupby("country", as_index=False)["revenue"]
        .sum()
        .nlargest(10, "revenue")
        .sort_values("revenue")
    )

    country_fig = px.bar(
        country_sales,
        x="revenue",
        y="country",
        orientation="h",
        title="Top Countries by Gross Sales",
        labels={
            "revenue": "Gross Sales",
            "country": "Country",
        },
    )

    st.plotly_chart(country_fig, use_container_width=True)


st.header("👥 Customer Segmentation")

cluster_summary = (
    segments.groupby("cluster")
    .agg(
        customers=("customer_id", "count"),
        avg_recency_days=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    )
    .reset_index()
)

segment_names = {
    0: "Regular Customers",
    1: "Occasional Customers",
    2: "High-Value Loyal Customers",
    3: "At-Risk Customers",
}

cluster_summary["segment"] = (
    cluster_summary["cluster"].map(segment_names)
)

segment_fig = px.bar(
    cluster_summary,
    x="segment",
    y="customers",
    title="Customers by Segment",
    color="segment",
    labels={
        "segment": "Customer Segment",
        "customers": "Number of Customers",
    },
)

st.plotly_chart(segment_fig, use_container_width=True)

st.subheader("Customer Segment Details")

display_summary = cluster_summary[
    [
        "segment",
        "customers",
        "avg_recency_days",
        "avg_frequency",
        "avg_monetary",
    ]
].copy()

display_summary.columns = [
    "Customer Segment",
    "Customers",
    "Average Recency (Days)",
    "Average Orders",
    "Average Gross Spending",
]

st.dataframe(
    display_summary.round(2),
    use_container_width=True,
    hide_index=True,
)

st.header("🔮 Sales Demand Forecast")

historical_end = sales["invoicedate"].max().normalize()

future_forecast = forecast[
    forecast["ds"] > historical_end
].copy()

if future_forecast.empty:
    st.warning(
        "No future forecasts found. Run demand_forecasting.py again."
    )
else:
    forecast_fig = px.line(
        future_forecast,
        x="ds",
        y="predicted_sales",
        title="Forecasted Daily Gross Sales",
        labels={
            "ds": "Forecast Date",
            "predicted_sales": "Predicted Gross Sales",
        },
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True,
    )

    predicted_total = future_forecast["predicted_sales"].sum()

    st.metric(
        "Total Forecast Gross Sales",
        f"{predicted_total:,.2f}",
        help="Sum of the model's predicted daily gross sales.",
    )

    st.caption(
        "Historical data ends on December 9, 2010. "
        "Forecasts extend beyond that date and are illustrative, "
        "not current-day business predictions."
    )


st.header("🔎 Explore Sales Data")

st.dataframe(
    filtered_sales.head(100),
    use_container_width=True,
    hide_index=True,
)

csv_data = filtered_sales.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Sales Data",
    data=csv_data,
    file_name="retailpulse_filtered_sales.csv",
    mime="text/csv",
)

st.success("RetailPulse dashboard loaded successfully.")