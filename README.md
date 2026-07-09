# Sales Forecasting Dashboard

An end-to-end sales analytics project built for the XYLOFY AI internship. The repository contains a Streamlit dashboard, forecasting experiments, anomaly detection, clustering analysis, and supporting visuals generated from the `train.csv` dataset.

## Project Overview

This project turns historical retail sales data into an interactive business dashboard. It helps explore regional and category performance, compare forecasting approaches, identify anomalies, and group products into meaningful business segments.

## Key Features

- Interactive Streamlit dashboard for sales exploration
- Region and category filtering from the sidebar
- Monthly sales trend and moving average analysis
- Forecast comparison across SARIMA, Prophet, and XGBoost
- Anomaly detection using Isolation Forest
- Product clustering using K-Means and PCA
- Executive summary of business insights

## Numerical Results

The current project analysis includes the following measurable results:

- 9,800 data rows processed from `train.csv`
- 4,922 unique orders analyzed
- Total sales of $2,261,536.78
- Average order value of $459.48
- 4 regions and 3 product categories in scope
- Date range from 2015-01-03 to 2018-12-30

## Forecasting Metrics

Model comparison in the dashboard currently shows:

- SARIMA: MAE 20,580.70, RMSE 22,190.91, MAPE 21.94%
- Prophet: MAE 20,250.79, RMSE 22,318.41, MAPE 21.86%
- XGBoost: MAE 15,110.78, RMSE 19,239.17, MAPE 14.81%

XGBoost is the recommended model based on the lowest error values.

## Repository Contents

- `app.py` - Main Streamlit application
- `train.csv` - Source dataset used for analysis
- `Final_Project_Submission_XYLOFY_AI.ipynb` - Notebook version of the project
- `Charts/` - Generated charts and figures used in the submission
- `Summary.pdf` - Final project summary document
- `requirements.txt` - Python dependencies

## Requirements

Install Python 3.10+ and the packages listed in `requirements.txt`.

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

Start the app with Streamlit:

```bash
streamlit run app.py
```

The dashboard will open in your browser and load the sales data from `train.csv`.

## What the Dashboard Shows

- Total sales, total orders, and average order value
- Sales trends over time
- Category performance breakdown
- Forecasting model comparison
- Anomaly markers on the sales timeline
- Product cluster visualization
- Written business insights for decision-making

## Notes

- The project expects `train.csv` to stay in the project root.
- Charts in `Charts/` are included as submission artifacts and supporting visuals.
- The best-performing forecasting model in the current analysis is XGBoost.

## Author

Kuchuru Sai Krishna Reddy

## Acknowledgment

Prepared for the XYLOFY AI Internship Program.