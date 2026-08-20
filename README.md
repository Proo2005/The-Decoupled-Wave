<img src="./logo.png" alt="App Logo" width="60" height="60" align="left" style="margin-right: 15px;"/>
<br clear="left"/>

# Hybrid ARIMA-XGBoost Epidemiological Forecasting Framework

A CPU-optimized, decoupled hybrid forecasting architecture designed to predict infectious disease transmission trajectories by combining statistical linear extraction with machine learning-driven residual mapping.

---

## 1. Core Innovation & Competitive Advantage

* **Decoupled Variance Mapping:** Unlike monolithic architectures that attempt to model all variance simultaneously, this framework isolates residuals from a statistical baseline, transforming unexplainable noise into a secondary supervised learning target.
* **Resource Constraint Optimization:** Designed specifically for CPU-bound environments, the pipeline leverages parallelized tree-building algorithms rather than heavy recurrent neural networks (RNN/LSTM), making it optimal for rapid academic deployment and resource-limited settings.
* **Temporal Sliding-Window Formulation:** The architecture dynamically engineers sequential lag predictors from one-dimensional temporal residuals, capturing reporting cyclicality without inducing data leakage.

---

## 2. Models & System Architecture

* **Auto-ARIMA Engine (`hybrid_arima_core.pkl`):** Serves as the statistical layer to extract the deterministic linear trajectory. Converges on an optimal $\text{ARIMA}(5, 1, 4)$ structure.
* **XGBoost Regressor (`hybrid_xgb_residual.pkl`):** Serves as the machine learning layer to map non-linear residual errors utilizing gradient-boosted decision trees across a 7-day temporal lag window.
* **FastAPI Microservice (`app.py`):** Provides an asynchronous backend for scalable data ingestion and serialized model execution via `joblib`.
* **Streamlit Dashboard (`frontend.py`):** Acts as an interactive frontend supporting real-time CSV ingestion, dynamic $T+1$ prediction, and interactive time-series trajectory plotting via Plotly.

---

## 3. Mathematical Methodology & Deductions

The foundational assumption defines an epidemiological time-series vector $Y_t$ as a composite of a deterministic linear component $L_t$ and a stochastic non-linear component $N_t$:

$$Y_t = L_t + N_t$$

### Phase 1: Deterministic Linear Extraction
The statistical layer utilizes an Auto-ARIMA algorithm that iterates through a hyperparameter search space minimizing the Akaike Information Criterion (AIC). Utilizing the backward shift operator $B$, the structural formulation is defined as:

$$\phi_p(B)(1-B)^d L_t = \theta_q(B)\epsilon_t$$

where $\phi_p$ represents the autoregressive parameters, $d$ dictates the degree of differencing required for stationarity, and $\theta_q$ represents the moving average components.

### Phase 2: Stochastic Residual Isolation & Supervised Mapping
Following linear extraction, the unmapped residual variance $e_t$ is isolated:

$$e_t = Y_t - \hat{L}_t$$

This one-dimensional residual vector is transformed into a two-dimensional supervised matrix using a 7-day sliding temporal lag window ($k = 7$). The XGBoost regressor maps the non-linear errors across this temporal window.

### Phase 3: Hybrid Recombination
The final integrated forecast is synthesized by combining the output of both independent layers:

$$\hat{Y}_t = \hat{L}_t + f(e_{t-1}, e_{t-2}, \dots, e_{t-k})$$

---

## 4. Empirical Prediction Results

The architecture was evaluated across out-of-sample validation sequences, yielding the following performance metrics:

* **Validation Mean Absolute Error (MAE):** $83.48$
* **Validation Root Mean Squared Error (RMSE):** $85.34$

The tight convergence between MAE and RMSE indicates robust generalization capabilities, demonstrating that the gradient boosting layer effectively mitigates extreme prediction errors and captures stochastic variance without overfitting.

---
<img src="./images/Epidemilogical forecasting  Hybrid ARIMA_XGBoost vs Actuals.png"  align="center" />
<br clear="left"/>

## 5. Local Installation & Execution

### Prerequisites
* Python 3.10+
* Git

### Initialization
If building from sratch , run :
```bash
python model.py
```
(This will create 2 pkl files)

### Step 1: Clone Repository & Install Dependencies
```bash
git clone https://github.com/Proo2005/The-Decoupled-Wave
cd disease_detecton_model
pip install -r requirements.txt
```

### Step 2: Initialize FastAPI backend
```bash
python app.py
```
(the server will initialize at   https://127.0.0.1:8000)

### Step 3: Launch Streamlit Dashboard
Open a secondary terminal window and execute:
```bash
streamlit run frontend.py
```
(The web interface will launch at   http://localhost:8501)

## Basic Configuration (dockerfile)
```bash
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501
```

## 6 Service Orchestration (docker-compose.yml)

```bash
services:
  backend:
    build: .
    command: uvicorn app:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    volumes:
      - .:/app

  frontend:
    build: .
    command: streamlit run frontend.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    volumes:
      - .:/app
    depends_on:
      - backend
```

## Execution Protocol
```bash
docker-compose up --build
```
* FastAPI Documentation : http://localhost:8000/docs
* Streamlit Interface: http://localhost:8501


## 7. Directory Structure
```bash
disease_detecton_model/
│
├── model/
│   ├── hybrid_arima_core.pkl      # Serialized statistical weights
│   └── hybrid_xgb_residual.pkl    # Serialized gradient-boosted weights
│
├── app.py                         # FastAPI backend microservice
├── frontend.py                    # Streamlit interactive dashboard
├── requirements.txt               # Project dependency manifest
├── Dockerfile                     # Containerization blueprint
└── docker-compose.yml             # Multi-service orchestration config
```


## Support

For support, email prochak1922@gmail.com or connect via [linkedin](https://www.linkedin.com/in/prodipta-chakraborty-5484b722a/).


## Links

- [@World Health Organization](https://data.who.int/dashboards/covid19/data)

- [@IEEE Xplore](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=epidemic%20forecasting&highlight=true&returnFacets=ALL&returnType=SEARCH&matchPubs=true&ranges=2022_2026_Year)