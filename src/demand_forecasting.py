from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

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

FORECAST_PATH = OUTPUT_DIR / "sales_forecast.csv"
CHART_PATH = CHART_DIR / "sales_forecast.png"

print("=" * 60)
print("RETAILPULSE - DEMAND FORECASTING")
print("=" * 60)

df = pd.read_csv(DATA_PATH, low_memory=False)

df["invoicedate"] = pd.to_datetime(
    df["invoicedate"], errors="coerce"
)

df["quantity"] = pd.to_numeric(
    df["quantity"], errors="coerce"
)

df["price"] = pd.to_numeric(
    df["price"], errors="coerce"
)

df = df.dropna(
    subset=["invoicedate", "quantity", "price"]
).copy()

df = df[
    (df["quantity"] > 0)
    & (df["price"] > 0)
].copy()

df["revenue"] = df["quantity"] * df["price"]

print(f"\nSales records loaded: {len(df):,}")

df["date"] = df["invoicedate"].dt.normalize()

daily_sales = (
    df.groupby("date", as_index=False)["revenue"]
    .sum()
)

daily_sales.columns = ["ds", "y"]

daily_sales = (
    daily_sales.set_index("ds")
    .asfreq("D", fill_value=0)
    .rename_axis("ds")
    .reset_index()
)

daily_sales["y"] = daily_sales["y"].astype(float)

print(f"First sales date: {daily_sales['ds'].min().date()}")
print(f"Last sales date: {daily_sales['ds'].max().date()}")
print(f"Daily observations: {len(daily_sales):,}")

model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=False,
)

print("\nTraining Prophet forecasting model...")
model.fit(daily_sales)

print("Model training completed.")


future = model.make_future_dataframe(
    periods=30,
    freq="D",
)

forecast = model.predict(future)

forecast_output = forecast[
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
].copy()

forecast_output["predicted_sales"] = (
    forecast_output["yhat"].clip(lower=0)
)

forecast_output["lower_bound"] = (
    forecast_output["yhat_lower"].clip(lower=0)
)

forecast_output["upper_bound"] = (
    forecast_output["yhat_upper"].clip(lower=0)
)

forecast_output = forecast_output[
    [
        "ds",
        "predicted_sales",
        "lower_bound",
        "upper_bound",
    ]
]

forecast_output.to_csv(FORECAST_PATH, index=False)

future_forecast = forecast_output[
    forecast_output["ds"] > daily_sales["ds"].max()
].copy()

print("\nNEXT 30 DAYS - SALES FORECAST")
print(
    future_forecast.head(30).to_string(
        index=False,
        formatters={
            "predicted_sales": "{:.2f}".format,
            "lower_bound": "{:.2f}".format,
            "upper_bound": "{:.2f}".format,
        },
    )
)

print(
    "\nTotal predicted sales for the next 30 days: "
    f"{future_forecast['predicted_sales'].sum():,.2f}"
)

fig = model.plot(forecast)

fig.set_size_inches(12, 6)
fig.suptitle(
    "RetailPulse - Daily Sales Forecast",
    fontsize=15,
)

fig.tight_layout()
fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

component_fig = model.plot_components(forecast)

component_path = CHART_DIR / "forecast_components.png"

component_fig.set_size_inches(12, 8)
component_fig.tight_layout()
component_fig.savefig(
    component_path,
    dpi=150,
    bbox_inches="tight",
)
plt.close(component_fig)

print(f"\nForecast data saved to:\n{FORECAST_PATH}")
print(f"Forecast chart saved to:\n{CHART_PATH}")
print(f"Forecast components saved to:\n{component_path}")

print("\nDemand forecasting completed successfully!")