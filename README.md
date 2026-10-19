# RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform

## 1. Project Overview

RetailPulse is a retail analytics platform that analyzes historical sales data, identifies customer purchasing patterns, segments customers using machine learning, and forecasts future sales.

The project combines data cleaning, exploratory data analysis, customer segmentation, demand forecasting, and an interactive dashboard.

## 2. Project Objectives

- Analyze historical retail sales data.
- Clean and prepare transaction data.
- Identify sales trends and top-performing products.
- Analyze sales performance across countries.
- Segment customers using RFM analysis and K-Means clustering.
- Forecast daily sales using Prophet.
- Display insights through an interactive dashboard.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Prophet
- Streamlit
- Plotly
- Excel and CSV

## 4. Dataset

This project uses the Online Retail II dataset obtained through Kaggle.

The dataset includes invoice numbers, product codes, product descriptions, quantities, invoice dates, prices, customer IDs, and countries.

The analyzed historical period is December 1, 2009, to December 9, 2010.

## 5. Project Modules

### Data Cleaning
- Remove duplicate transactions.
- Handle missing values.
- Prepare numeric fields.
- Calculate transaction revenue.
- Validate the cleaned dataset.

### Exploratory Data Analysis
- Calculate gross sales.
- Analyze daily and monthly sales trends.
- Identify top products.
- Compare sales across countries.
- Generate visualizations.

### Customer Segmentation
- Calculate Recency, Frequency, and Monetary features.
- Apply K-Means clustering.
- Group customers according to purchasing behavior.
- Save customer segment results.
- Visualize customer groups.

### Demand Forecasting
- Aggregate transactions into daily sales.
- Train a Prophet forecasting model.
- Generate a 30-day sales forecast.
- Save forecast results.
- Visualize predicted sales and forecast components.

### Interactive Dashboard
- Display sales KPIs.
- Explore daily and monthly sales.
- View top products and countries.
- Analyze customer segments.
- Explore sales forecasts.
- Filter sales by country.
- Download filtered sales data.

## 6. Key Results

- Cleaned dataset: 515,668 records.
- Prepared positive-sales dataset: 504,730 records.
- Gross sales: approximately 10.27 million in the dataset's currency.
- Unique invoices: 20,951.
- Unique products: 4,251.
- Identified customers: 4,312.
- Countries represented: 40.
- Customer segments: 4.
- Forecast horizon: 30 days.

## 7. Technologies and Methodology

- Pandas for data manipulation.
- Matplotlib and Plotly for visualization.
- Scikit-learn for K-Means clustering.
- Prophet for time-series forecasting.
- Streamlit for the interactive web dashboard.

## 8. How to Run the Project

### Step 1: Activate the virtual environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
