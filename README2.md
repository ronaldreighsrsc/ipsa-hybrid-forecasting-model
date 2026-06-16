# IPSA Hybrid Forecasting Model

This repository contains the codebase architecture for a master's/undergraduate thesis on predictive modeling and backtesting of the IPSA stock index.

The project compares classical statistical models (ARIMAX), Machine Learning algorithms (Random Forest, XGBoost), Deep Learning architectures (LSTM, BiLSTM), and sequential Hybrid Models. It evaluates their directional predictive capabilities under realistic financial constraints.

---

## Project Architecture

The code is structured following modularity and single-responsibility principles (SOLID), dividing the workflow into three sequential blocks:

```text
ipsa-hybrid-forecasting-model/
 |-- data/
 |   |-- raw/                # Original unprocessed CSVs and Excel files.
 |   |-- processed/          # Consolidated and stationary dataset (ipsa_master_processed.csv).
 |-- src/
 |   |-- data_engineering/   # Data extraction modules (scrape_ipsa.py, auto_updater.py)
 |   |-- preprocessing/      # Block 2: ETL, Technical Features, Volatility (EGARCH), and FFD.
 |   |-- models/             # Predictive engine classes and Data Leakage prevention logic.
 |   |-- evaluation/         # Block 4: Correlation Analysis and Triple Barrier Backtesting.
 |   |-- dashboard/          # Interactive Streamlit Dashboard (Data Visualization).
 |   |-- main_updater.py       # Block 1 Orchestrator (Raw Data extraction)
 |   |-- main_preprocessing.py # Block 2 Orchestrator (Math and FFD)
 |   |-- main_ablation.py      # Orchestrator (Ablation tournament)
 |   |-- main_evaluation.py    # Block 4 Orchestrator (Financial Results)
 |-- fastapii.py                 # RESTful API (FastAPI)
 |-- Dockerfile                  # Production Container
 |-- .dockerignore               # Docker exclusion rules
 |-- .github/
 |-- requirements.txt      # Project dependencies
 |-- README.md
```

---

## Installation and Setup

1. **Clone the repository** and open the terminal in the project root.
2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution Workflow (Pipeline)

To reproduce the thesis experiments or update the database, execute the following scripts in order:

### Block 1: Master Raw Data Updater
Updates the entire raw database (scrapes IPSA via Selenium, downloads APIs for S&P500/Copper/TPM/FXI/Yield). It does NOT perform feature engineering or FFD.
```bash
python src/main_updater.py
```
*Note: A Google Chrome window will briefly appear to bypass Cloudflare protection for IPSA data.*

*(Updates raw CSVs in `data/raw/`)*

### Block 2: Data Preprocessing & Consolidation
Handles data ingestion from the `data/raw/` directory, performs null imputation, calculates technical indicators (MACD, RSI, ATR), models conditional volatility using EGARCH, and applies **Fractional Differencing (FFD)** to achieve stationarity while preserving long-term memory.
```bash
python src/main_preprocessing.py
```
*(Generates: `data/processed/ipsa_master_processed.csv`)*
*(Generates: `data/processed/ipsa_master_processed.csv`)*

### Block 2: Ablation Study (Training)
Runs the model tournament. It utilizes **Purged K-Fold Cross-Validation with Embargo (López de Prado)** to prevent temporal data leakage and optimizes hyperparameters. Then, it performs a step-by-step *Walk-Forward* prediction on the out-of-sample test set.
*(You can configure which models to evaluate by editing the `MODELOS_A_CORRER` variable inside the script).*
```bash
python src/main_ablation.py
```
*(Generates: `.npy` files containing the predicted probabilities in `src/evaluation/results/` and a `tabla_ablacion_global.csv` file)*

### Block 3: Financial Evaluation and Backtesting
Checks for multicollinearity using a Pearson correlation matrix and evaluates the economic performance of the models using the **Triple Barrier** method. It simulates dynamic financial trading accounting for conditional volatility, Take Profit, Stop Loss, time limits, and **institutional friction** (Slippage + Broker Commissions).
```bash
python src/main_evaluation.py
```
*(Generates: An institutional ASCII summary table in the console and an Equity Curve vs. Benchmark plot).*

### Interactive Dashboard (Visualization)
Launches a **Streamlit** web application that serves as an interactive showcase for the project results. It allows non-technical stakeholders to explore the processed data (candlestick charts, EGARCH volatility, FFD), compare model performance from the ablation study, and review the correlation analysis — all through an intuitive point-and-click interface.
```bash
streamlit run src/dashboard/app1.py
```
*(Opens a local web server at `http://localhost:8501` to view IPSA performance and variables).*

### Block 4: RESTful API Deployment (MLOps Model Registry)
Exposes the models via a **FastAPI** server with a dynamic Model Registry architecture.
- **`POST /promote`**: Evaluates the results from the ablation study and hot-swaps the model in production based on the chosen metric (e.g., `{"metric": "alpha"}` or `{"metric": "sharpe"}`).
- **`POST /predict`**: Accepts up to 60-day historical window matrices and dynamically trims the input variables to perfectly match the active model's architecture.

```bash
uvicorn fastapii:app --reload
```
*(Swagger UI documentation is available at `http://127.0.0.1:8000/docs`).*

---

## 🐳 Docker Deployment (MLOps)

The API is containerized using **Docker** to provide a seamless, reproducible production environment.

1. **Build the image:**
   ```bash
   docker build -t ipsa-api .
   ```
2. **Run the container:**
   ```bash
   docker run -p 8000:8000 ipsa-api
   ```
*(The API will be exposed and accessible at `http://localhost:8000/docs`).*

---

## Implemented Models

1. **ARIMAX:** Univariate and multivariate statistical model with AIC-based order search.
2. **Random Forest:** Ensemble learning for directional classification.
3. **XGBoost:** Extreme Gradient Boosting.
4. **LSTM:** Long Short-Term Memory recurrent neural network.
5. **BiLSTM:** Bidirectional neural network.
6. **ARIMA-LSTM (Residual Hybrid):** Non-linear error fitting over stochastic regression.
7. **LSTM-RF (Extractor Hybrid):** 3D latent feature extraction coupled with a random forest classifier.
