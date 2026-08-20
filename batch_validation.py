import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Component Initialization
print("Initializing serialized hybrid architecture...")
arima_model = joblib.load('model/hybrid_arima_core.pkl')
xgb_model = joblib.load('model/hybrid_xgb_residual.pkl')

# 2. Unseen Data Ingestion
# Replace 'new_dataset.csv' with your actual target file
file_path = "new_dataset.csv"
df_new = pd.read_csv(file_path)
df_new['Date_reported'] = pd.to_datetime(df_new['Date_reported'])
df_new.set_index('Date_reported', inplace=True)
df_new.sort_index(inplace=True)

# Isolate target vector
new_target_series = df_new[df_new['Country'] == 'India']['New_cases'].dropna()
print(f"New validation sequence acquired. Shape: {new_target_series.shape}")

# 3. Supervised Matrix Construction
def create_supervised_features(series, lag=7):
    X, y = [], []
    for i in range(len(series) - lag):
        X.append(series.iloc[i:(i + lag)].values)
        y.append(series.iloc[i + lag])
    return np.array(X), np.array(y)

# 4. Hybrid Inference Execution
# Generate the baseline forecast strictly for the 20-day horizon
print("Executing deterministic linear forecasting...")
new_linear_baseline = arima_model.predict(n_periods=len(new_target_series))

# Extract raw arrays to enforce dimensional alignment
raw_actuals = new_target_series.values.flatten()
raw_baseline = np.array(new_linear_baseline).flatten()

# Isolate the non-linear variance
new_residuals_array = raw_actuals - raw_baseline
new_residuals = pd.Series(new_residuals_array)

# Construct the supervised matrix using the 7-day lag
temporal_lag = 7
X_val, y_val = create_supervised_features(new_residuals, lag=temporal_lag)

print("Executing XGBoost stochastic residual mapping...")
new_residual_predictions = xgb_model.predict(X_val)

# 5. Signal Aggregation
# Discard the initial 7 days of the baseline to align with the supervised matrix
aligned_linear = raw_baseline[temporal_lag:]
hybrid_validation_predictions = aligned_linear + new_residual_predictions

# 6. Metric Quantification
actual_validation_values = raw_actuals[temporal_lag:]
mae = mean_absolute_error(actual_validation_values, hybrid_validation_predictions)
rmse = np.sqrt(mean_squared_error(actual_validation_values, hybrid_validation_predictions))

print(f"Validation MAE:  {mae:.4f}")
print(f"Validation RMSE: {rmse:.4f}")