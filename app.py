import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize the high-performance asynchronous API
app = FastAPI(title="Hybrid ARIMA-XGBoost Forecasting API")

# 2. Load the serialized mathematical components from the 'model' subdirectory
print("Loading serialized architectural weights...")
arima_model = joblib.load('model/hybrid_arima_core.pkl')
xgb_model = joblib.load('model/hybrid_xgb_residual.pkl')

# 3. Define the strict data validation schema for incoming requests
class ForecastRequest(BaseModel):
    recent_residuals: list[float]
    forecast_horizon: int = 1

# 4. Construct the prediction endpoint
@app.post("/predict")
def generate_forecast(request: ForecastRequest):
    baseline_forecast = arima_model.predict(n_periods=request.forecast_horizon)
    input_residuals = np.array(request.recent_residuals[-7:]).reshape(1, -1)
    error_forecast = xgb_model.predict(input_residuals)
    final_prediction = baseline_forecast[0] + error_forecast[0]
    
    return {
        "baseline_linear": float(baseline_forecast[0]), 
        "non_linear_adjustment": float(error_forecast[0]),
        "hybrid_forecast": float(final_prediction)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)