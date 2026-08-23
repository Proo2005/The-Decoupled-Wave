import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load Data and Models
file_path = "dataset/who_dataset.csv"
df = pd.read_csv(file_path)
df['Date_reported'] = pd.to_datetime(df['Date_reported'])
df.set_index('Date_reported', inplace=True)
df.sort_index(inplace=True)

target_series = df[df['Country'] == 'India']['New_cases'].dropna()
target_series = target_series.replace([np.inf, -np.inf], np.nan).dropna()
log_target_series = np.log1p(target_series)

# Replicate Train/Test Split (80% train, 20% test)
train_size = int(len(log_target_series) * 0.8)
train_series, test_series = log_target_series[:train_size], log_target_series[train_size:]

arima_model = joblib.load('model/hybrid_arima_core.pkl')
xgb_model = joblib.load('model/hybrid_xgb_residual.pkl')

# 2. Generate Train Predictions & Clean NaNs
train_preds_log = np.array(arima_model.predict_in_sample()).flatten()
min_train_len = min(len(train_series), len(train_preds_log))
train_series_aligned = train_series.iloc[-min_train_len:]
train_preds_aligned = train_preds_log[-min_train_len:]

train_residuals = train_series_aligned - train_preds_aligned
# Immediately clean any NaNs or Infs
train_residuals = pd.Series(train_residuals).replace([np.inf, -np.inf], np.nan).dropna()

# Function to create lag features safely
def create_supervised_features(series, lag=7):
    X, y = [], []
    for i in range(len(series) - lag):
        X.append(series.iloc[i:(i + lag)].values)
        y.append(series.iloc[i + lag])
    return np.array(X), np.array(y)

temporal_lag = 7
X_train, y_train = create_supervised_features(train_residuals, lag=temporal_lag)

# Final safety check for any remaining NaNs in feature matrices
mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
X_train, y_train = X_train[mask], y_train[mask]

train_pred_res = xgb_model.predict(X_train)

# 3. Generate Test Predictions (Rolling Out-of-Sample)
full_series = log_target_series.values
hybrid_preds_log = []
dampening_factor = 0.65

for i in range(len(test_series)):
    current_idx = train_size + i
    history = full_series[:current_idx]
    if len(history) >= temporal_lag:
        lag_window = history[-temporal_lag:]
        next_residual_pred = xgb_model.predict(lag_window.reshape(1, -1))[0]
    else:
        next_residual_pred = 0.0
    base_val = history[-1]
    pred_log = base_val + (next_residual_pred * dampening_factor)
    hybrid_preds_log.append(pred_log)

# Inverse transform with exact shape alignment
actual_train = np.expm1(train_series_aligned.iloc[temporal_lag:])
# Align train_preds_aligned with the cleaned training indices
valid_train_idx = temporal_lag + len(train_series_aligned) - len(train_preds_aligned)
pred_train = np.expm1(train_preds_aligned[-len(X_train):] + train_pred_res)
actual_train = actual_train.iloc[-len(X_train):]

actual_test = np.expm1(test_series.values)
pred_test = np.expm1(np.array(hybrid_preds_log))

# 4. Compute Metrics
# 4. Compute Metrics Safely by Dropping NaNs
train_df = pd.DataFrame({'actual': actual_train, 'pred': pred_train}).dropna()
test_df = pd.DataFrame({'actual': actual_test, 'pred': pred_test}).dropna()

train_rmse = np.sqrt(mean_squared_error(train_df['actual'], train_df['pred']))
test_rmse = np.sqrt(mean_squared_error(test_df['actual'], test_df['pred']))

train_mae = mean_absolute_error(train_df['actual'], train_df['pred'])
test_mae = mean_absolute_error(test_df['actual'], test_df['pred'])

print("=== OVERFITTING DIAGNOSTIC REPORT ===")
print(f"Training RMSE: {train_rmse:.4f} | Testing RMSE: {test_rmse:.4f}")
print(f"Training MAE:  {train_mae:.4f}  | Testing MAE:  {test_mae:.4f}")
if train_rmse > 0:
    print(f"Test-to-Train RMSE Ratio: {(test_rmse / train_rmse):.2f}x")
else:
    print("Test-to-Train RMSE Ratio: N/A (Train RMSE is 0)")


def calculate_smape(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    return 100 * np.mean(2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8))

train_smape = calculate_smape(train_df['actual'], train_df['pred'])
test_smape = calculate_smape(test_df['actual'], test_df['pred'])

print(f"Training sMAPE: {train_smape:.2f}%")
print(f"Testing sMAPE:  {test_smape:.2f}%")