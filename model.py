                                                    # check dataset
import pandas as pd
file_path = "dataset/who_dataset.csv"
df = pd.read_csv(file_path)
df['Date_reported'] = pd.to_datetime(df['Date_reported'])
df.set_index('Date_reported', inplace=True)
df.sort_index(inplace=True)
df_region = df[df['Country'] == 'India']
target_series = df_region['New_cases'].dropna()

print(f"Data successfully loaded. Series shape: {target_series.shape}")



                                 # Linear Trend Extraction and Residual Isolation

import pmdarima as pm
train_size = int(len(target_series) * 0.8)
train_series, test_series = target_series[:train_size], target_series[train_size:]

print(f"Training shape: {train_series.shape} | Testing shape: {test_series.shape}")
print("Initiating Auto-ARIMA optimization (computational time required)...")
arima_model = pm.auto_arima(
    train_series,
    seasonal=False,        # Evaluated as non-seasonal for baseline optimization
    trace=True,            # Outputs the hyperparameter search matrix
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print(f"Optimal ARIMA Configuration: {arima_model.order}")
linear_predictions = arima_model.predict_in_sample()
residuals = train_series - linear_predictions

print(f"Non-linear residuals successfully isolated. Shape: {residuals.shape}")



                              #4  Feature Augmentation and Non-Linear Regression

import numpy as np
from xgboost import XGBRegressor


def create_supervised_features(series, lag=7):
    """Engineers sequential lag predictors from a univariate time-series."""
    X, y = [], []
    for i in range(len(series) - lag):
        X.append(series.iloc[i:(i + lag)].values)
        y.append(series.iloc[i + lag])
    return np.array(X), np.array(y)

# 2. Construct the supervised learning matrix
# Utilizing a 7-day lag window captures weekly reporting cyclicality
temporal_lag = 7
X_train, y_train = create_supervised_features(residuals, lag=temporal_lag)

print(f"Supervised Feature Matrix X_train shape: {X_train.shape}")
print(f"Target Vector y_train shape: {y_train.shape}")

# 3. Configure the CPU-optimized XGBoost regressor
print("Initiating XGBoost non-linear regression training...")
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    objective='reg:squarederror',
    n_jobs=-1  # Maximizes parallel CPU threading
)

# 4. Fit the algorithm exclusively on the isolated non-linear errors
xgb_model.fit(X_train, y_train)

print("XGBoost training complete.")


                          #5: Hybrid Recombination and Error Quantification

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# 1. Generate linear baseline predictions for the testing partition
print("Generating Auto-ARIMA baseline forecasts...")
test_linear_predictions = arima_model.predict(n_periods=len(test_series))

# 2. Extract raw numpy arrays to strictly bypass Pandas index misalignment
raw_test_actuals = test_series.values.flatten()
raw_test_baseline = test_linear_predictions.values.flatten()

# 3. Calculate absolute residuals and reconstruct a clean Pandas Series
test_residuals_array = raw_test_actuals - raw_test_baseline
test_residuals = pd.Series(test_residuals_array)

# 4. Construct the supervised testing matrix
# The function will now iterate over the exact dimensions required
X_test, y_test = create_supervised_features(test_residuals, lag=temporal_lag)

# 5. Generate non-linear error predictions via the trained XGBoost model
print("Executing XGBoost non-linear regression mapping...")
test_residual_predictions = xgb_model.predict(X_test)

# 6. Recombine the components to construct the ultimate hybrid forecast
# Align the linear baseline array by systematically discarding the initial 'lag' days
aligned_linear_predictions = raw_test_baseline[temporal_lag:]
final_hybrid_predictions = aligned_linear_predictions + test_residual_predictions

# 7. Quantify the final architectural performance
actual_values = raw_test_actuals[temporal_lag:]
mae = mean_absolute_error(actual_values, final_hybrid_predictions)
rmse = np.sqrt(mean_squared_error(actual_values, final_hybrid_predictions))

print(f"Hybrid Architecture MAE:  {mae:.4f}")
print(f"Hybrid Architecture RMSE: {rmse:.4f}")


                   #6: Empirical Visualization and Temporal Verification

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(actual_values, label='Actual Historical Cases', color='blue', linewidth=2, alpha=0.7)
plt.plot(final_hybrid_predictions, label='Hybrid ARIMA-XGBoost Forecast', color='red', linestyle='--', linewidth=2)

plt.title('Epidemiological Forecasting: Hybrid ARIMA-XGBoost vs. Actuals')
plt.xlabel('Temporal Testing Window (Days)')
plt.ylabel('Daily Confirmed Cases')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

                                 #7: Model Serialization and Export

import joblib

# 1. Serialize the statistical Auto-ARIMA model
print("Exporting statistical baseline...")
joblib.dump(arima_model, 'hybrid_arima_core.pkl')

# 2. Serialize the non-linear XGBoost regressor
print("Exporting non-linear regressor...")
joblib.dump(xgb_model, 'hybrid_xgb_residual.pkl')

print("Architectural components successfully serialized for deployment.")