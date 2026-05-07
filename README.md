# Multi-Asset Trading Strategy & Stock Predictor 📈

This project is a comprehensive simulation and backtesting environment designed to evaluate multiple trading strategies across a diverse portfolio of 20 global assets. It compares traditional technical analysis, machine learning (LightGBM & Logistic Regression), and deep learning (LSTM) models against a classic "Buy and Hold" benchmark.

## 🚀 Key Features
- **Diverse Asset Coverage**: Supports 20 tickers including:
  - **US Tech Giants**: AAPL, MSFT, NVDA, TSLA, etc.
  - **Indian Blue-chips**: RELIANCE.NS, TCS.NS, HDFCBANK.NS.
  - **Crypto**: BTC, ETH, SOL.
  - **Commodities**: Gold (GC=F).
- **Multi-Strategy Approach**:
  - **Technical**: Ensemble of RSI, Moving Average Crossovers, Bollinger Bands, and OBV.
  - **Machine Learning**: LightGBM and Logistic Regression with periodic retraining and Optuna hyperparameter optimization.
  - **Deep Learning**: LSTM (Long Short-Term Memory) neural networks built with PyTorch to capture time-series dependencies.
- **Robust Backtesting**:
  - Parallelized simulation for efficiency.
  - Integrated transaction costs and slippage modeling.
  - Automatic currency conversion (USD/INR) for a unified USD-denominated portfolio.
  - Dynamic position sizing using ATR (Average True Range).

## 🛠️ Project Structure
The project is organized into a sequential pipeline of Jupyter Notebooks:

1.  **`0_data_and_features.ipynb`**: Fetches historical data via `yfinance` and engineers technical features (Volatility, Momentum, Volume indicators).
2.  **`1_technical_strategy.ipynb`**: Executes the indicator-based trading logic and logs performance.
3.  **`2_ml_strategy.ipynb`**: Trains and backtests LightGBM and Logistic Regression models using a rolling-window approach.
4.  **`3_dl_strategy.ipynb`**: Implements the LSTM architecture for time-series forecasting and trading.
5.  **`4_comparison.ipynb`**: Aggregates all results, generates equity curves, and provides a final performance leaderboard.

## 📊 Summary of Results (Sample Period: 2019-2024)

| Strategy | Avg. Total Return | Best Performer |
| :--- | :--- | :--- |
| **Buy & Hold** | 82.35% | NVDA (642%) |
| **Technical** | 18.63% | SOL-USD (93.2%) |
| **LSTM (DL)** | 17.76% | BHARTIARTL.NS (46.7%) |
| **LightGBM (ML)** | 6.66% | NVDA (48.8%) |

*Note: While Buy & Hold outperformed in absolute returns during this bull-heavy period (driven by NVDA's surge), the model-based strategies often showed lower drawdowns and more controlled risk profiles.*

## ⚙️ Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Dhruvjn007/algorithmic-trading-simulation/stock-predictor.git
    cd stock-predictor/stock_predictor
    ```

2.  **Install Dependencies**:
    *It is recommended to install PyTorch with CUDA support first if using a GPU.*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Pipeline**:
    Open the notebooks in order (0 to 4) and run all cells to reproduce the results.

## 🧰 Tech Stack
- **Languages**: Python
- **Data**: Pandas, NumPy, yfinance, PyArrow
- **ML/DL**: PyTorch, LightGBM, Scikit-learn, Optuna
- **Visualization**: Matplotlib, Seaborn

---
*Disclaimer: This project is for educational and research purposes only. Trading stocks and crypto involve significant risk.*
