import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from PIL import Image
from pathlib import Path

# 1. Modern Dark UI & Theme Configuration
st.set_page_config(
    page_title="HAX Epidemiological Intelligence",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Neon/Dark UI Aesthetic
st.markdown("""
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: 'Inter', sans-serif;
    }
    .hero-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-container.success {
        border-left-color: #10b981;
    }
    .metric-container.warning {
        border-left-color: #f59e0b;
    }
    h1, h2, h3 {
        color: #f9fafb !important;
        font-weight: 700;
    }
    .stFileUploader {
        background-color: #111827;
        border: 2px dashed #374151;
        border-radius: 12px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Model Initialization (Cached)
@st.cache_resource
def load_models():
    model_dir = Path(__file__).resolve().parent / "model"
    return joblib.load(model_dir / 'hybrid_arima_core.pkl'), joblib.load(model_dir / 'hybrid_xgb_residual.pkl')

arima_model, xgb_model = load_models()

# 3. Sidebar Configuration & Control Panel
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    uploaded_file = st.file_uploader("Upload Epidemiological Data (CSV)", type=['csv'])
    
    st.markdown("---")
    st.markdown("### ⚙️ Surveillance Window")
    eval_window = st.selectbox(
        "Active Evaluation Scope", 
        options=["Last 180 Days (Recommended)", "Last 365 Days", "Full Historical Timeline"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🧬 Architecture Overview")
    st.markdown("""
    * **Layer 1:** Auto-ARIMA ($\text{ARIMA}(5,1,4)$)
    * **Layer 2:** XGBoost Regressor ($k=7$ lag)
    * **Variance Stabilization:** Log-Space $\log(1+y)$
    """)
    st.markdown("---")
    st.markdown("<p style='color: #9ca3af; font-size: 0.8rem;'>HAX Engine v3.2 • CPU Optimized</p>", unsafe_allow_html=True)

# 4. Main Interface Header
st.markdown("""
    <div class="hero-card">
        <h1 style="margin: 0; font-size: 2.5rem; background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            HAX Epidemiological Intelligence
        </h1>
        <p style="color: #9ca3af; margin-top: 0.5rem; font-size: 1.1rem;">
            Decoupled Hybrid Forecasting Platform combining statistical linear extraction with non-linear residual mapping.
        </p>
    </div>
""", unsafe_allow_html=True)

# 5. Core Execution Logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date_reported'] = pd.to_datetime(df['Date_reported'])
    df.set_index('Date_reported', inplace=True)
    df.sort_index(inplace=True)
    
    full_target_series = df[df['Country'] == 'India']['New_cases'].dropna()
    full_target_series = full_target_series.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Apply evaluation window filtering
    if eval_window == "Last 180 Days (Recommended)" and len(full_target_series) > 180:
        target_series = full_target_series.tail(180)
    elif eval_window == "Last 365 Days" and len(full_target_series) > 365:
        target_series = full_target_series.tail(365)
    else:
        target_series = full_target_series

    if len(target_series) < 14:
        st.error("⚠️ Dataset must contain a minimum of 14 sequential observations to construct the required temporal lag matrix.")
    else:
        st.success(f"Dataset successfully ingested & filtered. Active surveillance records: **{len(target_series)}**")
        
        # 4. Hybrid Mathematical Inference (Dynamic Out-of-Sample Baseline Generation)
        log_target_series = np.log1p(target_series)
        log_vals = log_target_series.values
        
        # Generate dynamic baseline matching the exact length of the active target series
        n_steps = len(target_series)
        dynamic_baseline_log = arima_model.predict(n_periods=n_steps)
        if hasattr(dynamic_baseline_log, 'values'):
            raw_baseline_log = dynamic_baseline_log.values
        else:
            raw_baseline_log = np.array(dynamic_baseline_log).flatten()
            
        raw_actuals_log = log_vals
        min_len = min(len(raw_actuals_log), len(raw_baseline_log))
        
        raw_actuals_log = raw_actuals_log[-min_len:]
        raw_baseline_log = raw_baseline_log[-min_len:]
        historical_residuals_log = raw_actuals_log - raw_baseline_log
        
        # Extract final 7-day lag window from active slice
        recent_residuals = historical_residuals_log[-7:]
        input_matrix = np.array(recent_residuals).reshape(1, -1)
        
        # Generate stochastic adjustment with dampening in log space
        dampening_factor = 0.65
        future_adjustment_log = float(xgb_model.predict(input_matrix)[0]) * dampening_factor
        
        # Predict next-step baseline and combine
        next_step_baseline_log = float(arima_model.predict(n_periods=1).iloc[0])
        final_log_prediction = next_step_baseline_log + future_adjustment_log
        
        # Inverse transform cleanly back to real case counts
        final_prediction = float(np.expm1(final_log_prediction))
        final_baseline = float(np.expm1(next_step_baseline_log))
        final_adjustment = final_prediction - final_baseline
        
        final_prediction = max(0.0, final_prediction)
        final_baseline = max(0.0, final_baseline)
        
        # 6. Modern Dashboard Analytics Cards
        st.markdown("### 📊 Architectural Output: T+1 Next-Day Forecast")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="metric-container">
                    <p style="color: #9ca3af; margin: 0; font-size: 0.9rem;">Linear Baseline (T+1)</p>
                    <h2 style="margin: 0.2rem 0 0 0; color: #60a5fa;">{final_baseline:.1f}</h2>
                    <p style="color: #6a7282; font-size: 0.75rem; margin: 0;">Deterministic trend component</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class="metric-container warning">
                    <p style="color: #9ca3af; margin: 0; font-size: 0.9rem;">Stochastic Adjustment (XGB)</p>
                    <h2 style="margin: 0.2rem 0 0 0; color: #f59e0b;">{final_adjustment:+.1f}</h2>
                    <p style="color: #6a7282; font-size: 0.75rem; margin: 0;">7-day residual correction</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
                <div class="metric-container success">
                    <p style="color: #9ca3af; margin: 0; font-size: 0.9rem;">Hybrid Forecast (T+1)</p>
                    <h2 style="margin: 0.2rem 0 0 0; color: #10b981;">{final_prediction:.1f}</h2>
                    <p style="color: #10b981; font-size: 0.75rem; margin: 0;">▲ Active Next-Day Projection</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 7. Advanced Interactive Trajectory Plotting (Dynamic Window)
        st.subheader("📈 Temporal Trajectory & Outbreak Surveillance")
        active_dates = target_series.index[-min_len:]
        historical_baseline_real = np.expm1(raw_baseline_log)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=active_dates, y=target_series.values[-min_len:], 
            mode='lines', name='Historical Actuals', 
            line=dict(color='#3b82f6', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=active_dates, y=historical_baseline_real, 
            mode='lines', name='ARIMA Baseline Component', 
            line=dict(color='#f59e0b', width=1.5, dash='dot')
        ))
        
        future_date = active_dates[-1] + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[active_dates[-1], future_date], 
            y=[target_series.values[-1], final_prediction], 
            mode='lines+markers', name='Hybrid Forecast (T+1)',
            line=dict(color='#ef4444', width=3), 
            marker=dict(size=12, symbol='star', color='#ef4444')
        ))
        
        fig.update_layout(
            paper_bgcolor='#0b0f19',
            plot_bgcolor='#111827',
            font=dict(color='#f3f4f6'),
            xaxis=dict(title="Timeline Date", gridcolor='#1f2937'),
            yaxis=dict(title="Daily Confirmed Cases", gridcolor='#1f2937'),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("""
        <div style="background-color: #111827; border: 1px solid #1f2937; padding: 3rem; border-radius: 16px; text-align: center; margin-top: 2rem;">
            <h3 style="color: #9ca3af; margin-bottom: 0.5rem;">Awaiting Dataset Ingestion</h3>
            <p style="color: #6b7280; font-size: 0.95rem;">Please upload your chronological epidemiological CSV file using the control panel on the left to initialize the inference pipeline.</p>
        </div>
    """, unsafe_allow_html=True)