# Time Series Forecasting of Hourly Energy Consumption (XGBoost)

## Project Goal
Forecast hourly energy consumption (AEP_MW) using a supervised learning approach with XGBoost, treating the time series as a regression problem and leveraging calendar/time features instead of classical ARIMA-style models.

## Dataset
- File: `AEP_hourly.csv`
- Frequency: Hourly
- Columns:
  - `Datetime` – timestamp (used as index)
  - `AEP_MW` – energy consumption in megawatts

## Methodology
1. **Preprocessing**
   - Load `AEP_hourly.csv` with `Datetime` as a parsed datetime index.
   - Train/test split at `2015-01-01`:
     - Train: data ≤ 2015-01-01
     - Test: data > 2015-01-01

2. **Feature Engineering**
   From the datetime index, create:
   - `hour`, `dayofweek`, `quarter`, `month`, `year`
   - `dayofyear`, `dayofmonth`, `weekofyear`

3. **Model**
   - Model: `xgboost.XGBRegressor(n_estimators=1000)`
   - Early stopping on test set (`early_stopping_rounds=50`)
   - Metrics computed on test set:
     - Mean Squared Error (MSE)
     - Mean Absolute Error (MAE)
     - Mean Absolute Percentage Error (MAPE)

## Results (Qualitative)
- The XGBoost model successfully captures:
  - Daily patterns (hour-of-day effects)
  - Weekly and yearly seasonality (dayofweek, dayofyear, month, year)
- Predictions closely track actual consumption on the test set, especially for:
  - Typical days and weeks (e.g., January 2015, first week of July)
- Feature importance shows:
  - `dayofyear`, `hour`, and `year` as the most influential predictors.
- Error analysis:
  - Per-day average errors are inspected to identify best and worst predicted days.
  - Plots highlight days where the model performs very well and where it struggles (e.g., unusual peaks or anomalies).

## How to Run
1. Install dependencies:
   ```bash
   pip install xgboost pandas numpy matplotlib seaborn scikit-learn
   ```
2. Open the notebook:
   ```bash
   jupyter notebook "Time Series Forecasting using XGBoost.ipynb"
   ```
3. Run all cells in order to:
   - Load and visualize data
   - Train the XGBoost model
   - Generate forecasts and plots
   - Compute error metrics and analyze best/worst days