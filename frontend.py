import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# 1. Interface Configuration
st.set_page_config(page_title="Epidemic Forecasting Dashboard", layout="wide")
st.title("Hybrid ARIMA-XGBoost Forecasting")
st.markdown("Upload a chronological CSV dataset to dynamically generate out-of-sample predictions.")

# 2. Model Initialization
@st.cache_resource
def load_models():
    return joblib.load('model/hybrid_arima_core.pkl'), joblib.load('model/hybrid_xgb_residual.pkl')

arima_model, xgb_model = load_models()

# 3. CSV Ingestion Engine
uploaded_file = st.file_uploader("Upload Epidemiological Data (CSV)", type=['csv'])

if uploaded_file is not None:
    # Data Parsing
    df = pd.read_csv(uploaded_file)
    df['Date_reported'] = pd.to_datetime(df['Date_reported'])
    df.set_index('Date_reported', inplace=True)
    df.sort_index(inplace=True)
    
    # Isolate target sequence
    target_series = df[df['Country'] == 'India']['New_cases'].dropna()
    
    if len(target_series) < 7:
        st.error("Dataset must contain a minimum of 7 sequential observations to construct the required lag matrix.")
    else:
        st.success(f"Dataset successfully ingested. Extracted {len(target_series)} sequential records.")
        
        # 4. Hybrid Mathematical Inference
        # Generate deterministic baseline for historical length + 1 future step
        baseline_predictions = arima_model.predict(n_periods=len(target_series) + 1)
        future_baseline = float(baseline_predictions.iloc[-1] if hasattr(baseline_predictions, 'iloc') else baseline_predictions[-1])
        historical_baseline = baseline_predictions[:-1]
        
        # Calculate unmapped historical variance
        raw_actuals = target_series.values
        raw_baseline = historical_baseline.values if hasattr(historical_baseline, 'values') else historical_baseline
        historical_residuals = raw_actuals - raw_baseline
        
        # Extract the final 7-day lag window
        recent_residuals = historical_residuals[-7:]
        input_matrix = np.array(recent_residuals).reshape(1, -1)
        
        # Generate stochastic adjustment
        future_adjustment = float(xgb_model.predict(input_matrix)[0])
        
        # Synthesize final prediction
        hybrid_forecast = future_baseline + future_adjustment
        
        # 5. Dashboard Visualization
        st.subheader("Architectural Output: T+1 Forecast")
        col1, col2, col3 = st.columns(3)
        col1.metric("Linear Baseline", f"{future_baseline:.2f}")
        col2.metric("Stochastic Adjustment", f"{future_adjustment:.2f}")
        col3.metric("Hybrid Forecast", f"{hybrid_forecast:.2f}", delta="Next Day Projection")
        
        # 6. Trajectory Plotting
        st.subheader("Temporal Trajectory Analysis")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=target_series.index, y=raw_actuals, 
            mode='lines+markers', name='Historical Actuals', line=dict(color='#1f77b4')
        ))
        
        fig.add_trace(go.Scatter(
            x=target_series.index, y=raw_baseline, 
            mode='lines', name='ARIMA Baseline', line=dict(color='#ff7f0e', dash='dash')
        ))
        
        future_date = target_series.index[-1] + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[target_series.index[-1], future_date], 
            y=[raw_actuals[-1], hybrid_forecast], 
            mode='lines+markers', name='Hybrid Forecast (T+1)',
            line=dict(color='#d62728', width=3), marker=dict(size=10, symbol='star')
        ))
        
        fig.update_layout(xaxis_title="Date", yaxis_title="Daily Confirmed Cases", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)