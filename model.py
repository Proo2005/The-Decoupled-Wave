# 1. Check, Clean, and Load Dataset
import matplotlib
matplotlib.use("Agg")  # safe non-interactive backend; figure is also saved to disk

import pandas as pd
import numpy as np
import pmdarima as pm
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

project_dir = Path(__file__).resolve().parent
file_path = project_dir / "dataset" / "who_dataset.csv"
df = pd.read_csv(file_path)
df['Date_reported'] = pd.to_datetime(df['Date_reported'])
df.set_index('Date_reported', inplace=True)
df.sort_index(inplace=True)
df_region = df[df['Country'] == 'India']

# Extract target and drop missing values
target_series = df_region['New_cases'].dropna()

# WHO data occasionally contains negative values from retroactive corrections
# (e.g. a single -749 day for India). log1p() of anything < -1 is NaN, so these
# must be clipped to 0 explicitly rather than being silently dropped later,
# which would otherwise punch an undocumented hole in the daily time series.
n_negative = int((target_series < 0).sum())
if n_negative > 0:
    print(f"Clipping {n_negative} negative New_cases value(s) to 0 (data corrections).")
    target_series = target_series.clip(lower=0)

target_series = target_series.replace([np.inf, -np.inf], np.nan).dropna()

print(f"Data successfully loaded. Series shape: {target_series.shape}")

# Apply Log Transformation safely using log1p
log_target_series = np.log1p(target_series)
log_target_series = log_target_series.replace([np.inf, -np.inf], np.nan).dropna()

# 2. Linear Trend Extraction and Residual Isolation (on Log scale)
train_size = int(len(log_target_series) * 0.8)
train_series, test_series = log_target_series[:train_size], log_target_series[train_size:]

print(f"Training shape: {train_series.shape} | Testing shape: {test_series.shape}")
print("Initiating Auto-ARIMA optimization on log-transformed data...")
arima_model = pm.auto_arima(
    train_series,
    seasonal=False,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print(f"Optimal ARIMA Configuration: {arima_model.order}")
linear_predictions = arima_model.predict_in_sample()
residuals = train_series - linear_predictions
residuals = residuals.dropna()  # Ensure no NaNs from ARIMA burn-in period

print(f"Non-linear residuals successfully isolated. Shape: {residuals.shape}")

# 3. Feature Augmentation and Non-Linear Regression
def create_supervised_features(series, lag=7):
    X, y = [], []
    for i in range(len(series) - lag):
        X.append(series.iloc[i:(i + lag)].values)
        y.append(series.iloc[i + lag])
    return np.array(X), np.array(y)

temporal_lag = 7
X_train, y_train = create_supervised_features(residuals, lag=temporal_lag)

print("Initiating XGBoost non-linear regression training...")
xgb_model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    objective='reg:squarederror',
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
print("XGBoost training complete.")

# 4. Rolling One-Step-Ahead Hybrid Recombination
#
# IMPORTANT: this loop must mirror exactly what app.py does at inference time,
# otherwise the offline MAE/RMSE reported here don't reflect what the deployed
# API actually returns.
#   - The ARIMA baseline for step t must come from the model's own one-step
#     forecast (predict(n_periods=1)), not from the previous day's raw value.
#   - The XGBoost input must be a window of ARIMA *residuals*
#     (actual - ARIMA fitted/forecast), exactly like it was trained on in
#     step 3 above -- NOT a window of raw log-case values.
#   - The ARIMA model must be updated with each new true observation as we
#     roll forward, exactly like app.py's current_arima.update(...) call.
#   - No dampening factor is applied, since app.py applies none; keeping the
#     two in sync is what makes this evaluation meaningful.
print("Generating rolling out-of-sample predictions...")

log_test_full = log_target_series[train_size:]

rolling_arima = arima_model  # continues from the fit produced in step 2
residual_history = list(residuals.values)  # seed with in-sample training residuals

hybrid_preds_log = []
baseline_preds_log = []

for i in range(len(log_test_full)):
    # ARIMA one-step-ahead baseline forecast.
    # NOTE: pmdarima's predict() returns a pandas Series with a *datetime*
    # index when the model was fit on a datetime-indexed series, so naive
    # [0] positional indexing raises KeyError. np.asarray(...).ravel()[0]
    # safely extracts the scalar regardless of whether a Series or a plain
    # ndarray is returned (the return type can change after .update()).
    baseline_log = float(np.asarray(rolling_arima.predict(n_periods=1)).ravel()[0])
    baseline_preds_log.append(baseline_log)

    # XGBoost residual correction, using the most recent `temporal_lag` residuals
    if len(residual_history) >= temporal_lag:
        lag_window = np.array(residual_history[-temporal_lag:]).reshape(1, -1)
        residual_pred = float(xgb_model.predict(lag_window)[0])
    else:
        residual_pred = 0.0

    pred_log = baseline_log + residual_pred
    hybrid_preds_log.append(pred_log)

    # Reveal the true value for this step, then update ARIMA and the
    # residual history so the next iteration forecasts from real history
    true_log_val = float(log_test_full.iloc[i])
    rolling_arima.update([true_log_val])

    true_residual = true_log_val - baseline_log
    residual_history.append(true_residual)

# Inverse transform back to original case counts
final_hybrid_predictions = np.expm1(np.array(hybrid_preds_log))
final_hybrid_predictions = np.clip(final_hybrid_predictions, 0, None)
actual_values = np.expm1(log_test_full.values)

# Quantify performance
mae = mean_absolute_error(actual_values, final_hybrid_predictions)
rmse = np.sqrt(mean_squared_error(actual_values, final_hybrid_predictions))

print(f"Hybrid Architecture MAE:  {mae:.4f}")
print(f"Hybrid Architecture RMSE: {rmse:.4f}")

# 5. Empirical Visualization
plt.figure(figsize=(14, 6))
plt.plot(test_series.index, actual_values, label='Actual Historical Cases', color='blue', linewidth=2, alpha=0.7)
plt.plot(test_series.index, final_hybrid_predictions, label='Log-Hybrid ARIMA-XGBoost Forecast', color='red', linestyle='--', linewidth=2)

plt.title('Epidemiological Forecasting: Log-Transformed Hybrid Model vs. Actuals')
plt.xlabel('Date')
plt.ylabel('Daily Confirmed Cases')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_dir = Path(__file__).resolve().parent
plt.savefig(output_dir / 'forecast_plot.png', dpi=150)
print(f"Plot saved to {output_dir / 'forecast_plot.png'}")
plt.show()

# 6. Model Serialization
model_dir = Path(__file__).resolve().parent / 'model'
model_dir.mkdir(exist_ok=True)
joblib.dump(arima_model, model_dir / 'hybrid_arima_core.pkl')
joblib.dump(xgb_model, model_dir / 'hybrid_xgb_residual.pkl')
print("Models successfully exported.")
