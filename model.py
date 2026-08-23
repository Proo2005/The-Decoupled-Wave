# 1. Check, Clean, and Load Dataset
import pandas as pd
import numpy as np
import pmdarima as pm
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

file_path = "dataset/who_dataset.csv"
df = pd.read_csv(file_path)
df['Date_reported'] = pd.to_datetime(df['Date_reported'])
df.set_index('Date_reported', inplace=True)
df.sort_index(inplace=True)
df_region = df[df['Country'] == 'India']

# Extract target and drop any NaN or infinite values immediately
target_series = df_region['New_cases'].dropna()
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
residuals = residuals.dropna() # Ensure no NaNs from ARIMA burn-in period

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

# 4. Rolling One-Step-Ahead Hybrid Recombination with Dampened Residuals
print("Generating dampened rolling out-of-sample predictions...")

log_train_full = log_target_series[:train_size]
log_test_full = log_target_series[train_size:]

full_series = log_target_series.values
hybrid_preds_log = []
dampening_factor = 0.45  # Scales back over-correction during rapid surges

for i in range(len(log_test_full)):
    current_idx = train_size + i
    history = full_series[:current_idx]
    
    if len(history) >= temporal_lag:
        lag_window = history[-temporal_lag:]
        next_residual_pred = xgb_model.predict(lag_window.reshape(1, -1))[0]
    else:
        next_residual_pred = 0.0
        
    base_val = history[-1]
    # Apply dampening to prevent overshooting peaks
    pred_log = base_val + (next_residual_pred * dampening_factor)
    hybrid_preds_log.append(pred_log)

# Inverse transform back to original case counts
final_hybrid_predictions = np.expm1(np.array(hybrid_preds_log))
actual_values = np.expm1(log_test_full.values)

# Quantify performance
mae = mean_absolute_error(actual_values, final_hybrid_predictions)
rmse = np.sqrt(mean_squared_error(actual_values, final_hybrid_predictions))

print(f"Hybrid Architecture MAE:  {mae:.4f}")
print(f"Hybrid Architecture RMSE: {rmse:.4f}")

# 5. Empirical Visualization
plt.figure(figsize=(14, 6))
plt.plot(actual_values, label='Actual Historical Cases', color='blue', linewidth=2, alpha=0.7)
plt.plot(final_hybrid_predictions, label='Log-Hybrid ARIMA-XGBoost Forecast', color='red', linestyle='--', linewidth=2)

plt.title('Epidemiological Forecasting: Log-Transformed Hybrid Model vs. Actuals')
plt.xlabel('Temporal Testing Window (Days)')
plt.ylabel('Daily Confirmed Cases')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Model Serialization
joblib.dump(arima_model, 'hybrid_arima_core.pkl')
joblib.dump(xgb_model, 'hybrid_xgb_residual.pkl')
print("Models successfully exported.")