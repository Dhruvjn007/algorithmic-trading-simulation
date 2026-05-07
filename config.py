"""
Configuration for Multi-Asset Trading Strategy Simulation.
All constants and parameters in one place — edit here, affects all notebooks.
"""

import os

# ============================================================
# Asset Lists
# ============================================================
US_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'JNJ', 'V']
INDIAN_STOCKS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS']
CRYPTO = ['BTC-USD', 'ETH-USD', 'SOL-USD']
GOLD = ['GC=F']
EXCHANGE_RATE_TICKER = 'USDINR=X'

ALL_TICKERS = US_STOCKS + INDIAN_STOCKS + CRYPTO + GOLD

# ============================================================
# Date Range
# ============================================================
START_DATE = '2019-01-01'
END_DATE = '2024-12-31'

# ============================================================
# Capital (all in USD)
# ============================================================
TOTAL_CAPITAL = 100_000
CAPITAL_PER_ASSET = TOTAL_CAPITAL / len(ALL_TICKERS)  # $5,000

# ============================================================
# Transaction Costs (one-way, as decimal fraction)
# ============================================================
TRANSACTION_COSTS = {
    'us_stock': 0.001,       # 0.1%
    'indian_stock': 0.0015,  # 0.15%
    'crypto': 0.0025,        # 0.25%
    'gold': 0.001,           # 0.1%
}

# ============================================================
# Position Sizing
# ============================================================
RISK_PER_TRADE = 0.02   # risk 2% of capital per trade
ATR_MULTIPLIER = 2       # stop-loss distance = 2 * ATR(14)

# ============================================================
# Technical Indicator Parameters
# ============================================================
EMA_SHORT = 20
EMA_LONG = 50
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
OBV_EMA_PERIOD = 20

# ============================================================
# Technical Strategy Signal Weights & Thresholds
# ============================================================
INDICATOR_WEIGHTS = {
    'ema_crossover': 2.0,
    'rsi': 1.5,
    'bollinger': 1.0,
    'obv': 1.0,
}
BUY_THRESHOLD = 3.0
SELL_THRESHOLD = -3.0

# ============================================================
# ML Parameters
# ============================================================
ML_TRAIN_WINDOW = 750       # trading days for training (~3 years)
ML_RETRAIN_FREQ = 60        # retrain every 60 trading days (~1 quarter)
SAMPLE_WEIGHT_HALFLIFE = 250  # exponential decay half-life in days

# ============================================================
# DL Parameters
# ============================================================
DL_TRAIN_WINDOW = 500       # trading days for training (~2 years)
DL_RETRAIN_FREQ = 60        # retrain every 60 trading days
DL_SEQ_LENGTH = 60          # LSTM input sequence length
LSTM_HIDDEN_1 = 128         # first LSTM layer hidden size
LSTM_HIDDEN_2 = 64          # second LSTM layer hidden size
LSTM_DROPOUT = 0.3
LSTM_LR = 0.001
LSTM_MAX_EPOCHS = 50
LSTM_PATIENCE = 10          # early stopping patience
LSTM_LR_PATIENCE = 5        # ReduceLROnPlateau patience
LSTM_LR_FACTOR = 0.5        # LR reduction factor
LSTM_BATCH_SIZE = 64
MIN_VRAM_MB = 4096          # minimum free VRAM to use GPU (else CPU fallback)

# ============================================================
# Optuna
# ============================================================
OPTUNA_TRIALS = 20

# ============================================================
# Target Label Definition
# ============================================================
FORWARD_RETURN_DAYS = 5      # predict 5-day forward return direction
BUY_RETURN_THRESHOLD = 0.01  # forward return > +1% → BUY (label=2)
SELL_RETURN_THRESHOLD = -0.01  # forward return < -1% → SELL (label=0)
# HOLD = label 1 (between thresholds)

# ============================================================
# ML Feature Columns (used by ML and DL notebooks)
# ============================================================
FEATURE_COLS = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'ema_20_ratio', 'ema_50_ratio',
    'rsi_14',
    'bb_upper_ratio', 'bb_lower_ratio', 'bb_width',
    'atr_14_norm',
    'obv_ema_ratio',
    'volume_ratio',
    'high_low_ratio', 'close_open_ratio',
]

# ============================================================
# File Paths
# ============================================================
DATA_DIR = 'data'
RAW_DIR = os.path.join(DATA_DIR, 'raw')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'
LOGS_DIR = 'logs'
USDINR_PATH = os.path.join(DATA_DIR, 'usdinr.parquet')

# ============================================================
# Data Fetch Settings
# ============================================================
FETCH_RETRIES = 3
FETCH_RETRY_DELAY = 5  # seconds between retries
