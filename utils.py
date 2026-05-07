"""
Shared utility functions for Multi-Asset Trading Strategy Simulation.
Imported by all strategy notebooks for backtesting, metrics, and currency conversion.
"""

import numpy as np
import pandas as pd
import logging
import os
from config import (
    US_STOCKS, INDIAN_STOCKS, CRYPTO, GOLD,
    TRANSACTION_COSTS, RISK_PER_TRADE, ATR_MULTIPLIER,
    CAPITAL_PER_ASSET, LOGS_DIR, USDINR_PATH,
    DATA_DIR, RAW_DIR, FEATURES_DIR, RESULTS_DIR, PLOTS_DIR,
)


# ============================================================
# Setup
# ============================================================

def setup_logging(log_name='simulation'):
    """Configure logging to file and console."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f'{log_name}.log')

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # avoid adding duplicate handlers on re-import
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(fmt)
        console_handler.setFormatter(fmt)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def create_directories():
    """Create all required directories if they don't exist."""
    for d in [DATA_DIR, RAW_DIR, FEATURES_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)


# ============================================================
# Asset Classification
# ============================================================

def get_asset_class(ticker):
    """Return the asset class string for a given ticker."""
    if ticker in US_STOCKS:
        return 'us_stock'
    elif ticker in INDIAN_STOCKS:
        return 'indian_stock'
    elif ticker in CRYPTO:
        return 'crypto'
    elif ticker in GOLD:
        return 'gold'
    return 'unknown'


def get_transaction_cost(ticker):
    """Return one-way transaction cost rate for a ticker."""
    return TRANSACTION_COSTS.get(get_asset_class(ticker), 0.001)


def is_indian_stock(ticker):
    """Check if ticker is an Indian stock (needs INR→USD conversion)."""
    return ticker in INDIAN_STOCKS


# ============================================================
# Currency Conversion
# ============================================================

def load_exchange_rates():
    """Load forward-filled USDINR exchange rates from Parquet."""
    if not os.path.exists(USDINR_PATH):
        raise FileNotFoundError(f"Exchange rate file not found at {USDINR_PATH}. Run notebook 0 first.")
    rates = pd.read_parquet(USDINR_PATH)
    return rates


def convert_inr_to_usd(price_inr, date, exchange_rates):
    """
    Convert INR price to USD using historical exchange rate.
    exchange_rates: DataFrame with DatetimeIndex and 'rate' column.
    """
    # find the closest available date (should always match after forward-fill)
    if date in exchange_rates.index:
        rate = exchange_rates.loc[date, 'rate']
    else:
        # fallback: get the most recent rate before this date
        mask = exchange_rates.index <= date
        if mask.any():
            rate = exchange_rates.loc[mask].iloc[-1]['rate']
        else:
            rate = exchange_rates.iloc[0]['rate']
    return price_inr / rate


def get_price_in_usd(price, ticker, date, exchange_rates):
    """
    Get price in USD. Handles Indian stock conversion transparently.
    For non-Indian assets, returns price unchanged.
    """
    if is_indian_stock(ticker):
        return convert_inr_to_usd(price, date, exchange_rates)
    return price


# ============================================================
# Position Sizing (ATR-based)
# ============================================================

def calculate_position_size(capital, price_usd, atr_usd):
    """
    Calculate number of shares/units to buy based on ATR-based risk sizing.

    Logic:
    - risk_amount = capital * RISK_PER_TRADE (2%)
    - stop_loss_distance = ATR * ATR_MULTIPLIER
    - shares = risk_amount / stop_loss_distance
    - but never more than what capital can afford
    """
    if atr_usd <= 0 or price_usd <= 0:
        return 0

    risk_amount = capital * RISK_PER_TRADE
    stop_loss_distance = atr_usd * ATR_MULTIPLIER

    # shares based on risk
    shares_by_risk = risk_amount / stop_loss_distance

    # max shares we can afford
    shares_by_capital = capital / price_usd

    # take the smaller, floor to whole shares
    shares = int(min(shares_by_risk, shares_by_capital))
    return max(shares, 0)


# ============================================================
# Backtesting Engine
# ============================================================

def simulate_trading(prices_df, signals, ticker, initial_capital=None, exchange_rates=None, start_date=None):
    """
    Simulate trading for a single asset using signals.

    Args:
        prices_df: DataFrame with columns [Open, High, Low, Close, atr_14], DatetimeIndex
        signals: Series with BUY/SELL/HOLD values, DatetimeIndex
        ticker: string ticker symbol
        initial_capital: starting capital in USD (default: CAPITAL_PER_ASSET)
        exchange_rates: DataFrame with 'rate' column for INR conversion
        start_date: first date to start executing trades (common backtest start)

    Returns:
        portfolio_history: DataFrame with daily portfolio state
        order_book: DataFrame with executed trades only
    """
    if initial_capital is None:
        initial_capital = CAPITAL_PER_ASSET

    cost_rate = get_transaction_cost(ticker)
    needs_conversion = is_indian_stock(ticker)

    # align signals with price data
    common_dates = prices_df.index.intersection(signals.index)
    if start_date is not None:
        common_dates = common_dates[common_dates >= pd.Timestamp(start_date)]

    if len(common_dates) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # state variables
    cash = initial_capital
    position = 0  # shares held
    history = []
    trades = []

    for date in common_dates:
        signal = signals.loc[date]
        close_price = prices_df.loc[date, 'Close']
        atr = prices_df.loc[date, 'atr_14'] if 'atr_14' in prices_df.columns else close_price * 0.02

        # convert prices to USD if needed
        if needs_conversion and exchange_rates is not None:
            close_usd = get_price_in_usd(close_price, ticker, date, exchange_rates)
            atr_usd = get_price_in_usd(atr, ticker, date, exchange_rates)
        else:
            close_usd = close_price
            atr_usd = atr

        # execute signal
        if signal == 'BUY' and position == 0:
            shares_to_buy = calculate_position_size(cash, close_usd, atr_usd)
            if shares_to_buy > 0:
                trade_cost = shares_to_buy * close_usd
                fee = trade_cost * cost_rate
                cash -= (trade_cost + fee)
                position = shares_to_buy

                trades.append({
                    'timestamp': date,
                    'ticker': ticker,
                    'signal': 'BUY',
                    'price': close_usd,
                    'price_local': close_price,
                    'exchange_rate': close_price / close_usd if close_usd > 0 else 1.0,
                    'shares': shares_to_buy,
                    'cost': trade_cost,
                    'transaction_fee': fee,
                    'capital_after': cash,
                    'position': position,
                })

        elif signal == 'SELL' and position > 0:
            trade_revenue = position * close_usd
            fee = trade_revenue * cost_rate
            cash += (trade_revenue - fee)
            sold_shares = position
            position = 0

            trades.append({
                'timestamp': date,
                'ticker': ticker,
                'signal': 'SELL',
                'price': close_usd,
                'price_local': close_price,
                'exchange_rate': close_price / close_usd if close_usd > 0 else 1.0,
                'shares': sold_shares,
                'cost': trade_revenue,
                'transaction_fee': fee,
                'capital_after': cash,
                'position': position,
            })

        # record daily portfolio state
        holdings_value = position * close_usd
        portfolio_value = cash + holdings_value

        history.append({
            'Date': date,
            'ticker': ticker,
            'signal': signal,
            'price': close_usd,
            'position': position,
            'cash': cash,
            'holdings_value': holdings_value,
            'portfolio_value': portfolio_value,
        })

    portfolio_history = pd.DataFrame(history)
    if len(portfolio_history) > 0:
        portfolio_history.set_index('Date', inplace=True)
        portfolio_history['daily_return'] = portfolio_history['portfolio_value'].pct_change().fillna(0)

    order_book = pd.DataFrame(trades)
    return portfolio_history, order_book


# ============================================================
# Performance Metrics
# ============================================================

def compute_metrics(portfolio_history):
    """
    Compute key performance metrics from portfolio history.

    Returns dict with:
        total_return: percentage total return
        sharpe_ratio: annualized Sharpe ratio (assuming 252 trading days)
        max_drawdown: maximum drawdown as a positive fraction
        num_trades: total number of BUY+SELL trades
        win_rate: fraction of profitable trades (based on portfolio value at sell)
    """
    if portfolio_history is None or len(portfolio_history) == 0:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
        }

    pv = portfolio_history['portfolio_value']
    initial = pv.iloc[0]
    final = pv.iloc[-1]

    # total return
    total_return = (final - initial) / initial

    # sharpe ratio (annualized)
    daily_returns = portfolio_history['daily_return']
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # maximum drawdown
    cumulative_max = pv.cummax()
    drawdown = (cumulative_max - pv) / cumulative_max
    max_dd = drawdown.max()

    # trade stats
    if 'signal' in portfolio_history.columns:
        trade_signals = portfolio_history['signal']
        num_buys = (trade_signals == 'BUY').sum()
        num_sells = (trade_signals == 'SELL').sum()
    else:
        num_buys = 0
        num_sells = 0

    return {
        'total_return': round(total_return, 4),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(max_dd, 4),
        'num_trades': int(num_buys + num_sells),
    }


def compute_buy_and_hold(prices_df, ticker, initial_capital=None, exchange_rates=None, start_date=None):
    """
    Compute buy-and-hold baseline: buy on first day, hold until last day.
    Returns portfolio_history DataFrame.
    """
    if initial_capital is None:
        initial_capital = CAPITAL_PER_ASSET

    needs_conversion = is_indian_stock(ticker)
    df = prices_df.copy()

    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]

    if len(df) == 0:
        return pd.DataFrame()

    # buy on first day
    first_close = df['Close'].iloc[0]
    if needs_conversion and exchange_rates is not None:
        first_close_usd = get_price_in_usd(first_close, ticker, df.index[0], exchange_rates)
    else:
        first_close_usd = first_close

    shares = int(initial_capital / first_close_usd)
    remaining_cash = initial_capital - shares * first_close_usd

    history = []
    first_date = df.index[0]
    for date in df.index:
        close = df.loc[date, 'Close']
        if needs_conversion and exchange_rates is not None:
            close_usd = get_price_in_usd(close, ticker, date, exchange_rates)
        else:
            close_usd = close

        pv = remaining_cash + shares * close_usd
        history.append({
            'Date': date,
            'ticker': ticker,
            'signal': 'BUY' if date == first_date else 'HOLD',
            'portfolio_value': pv,
        })

    bh = pd.DataFrame(history).set_index('Date')
    bh['daily_return'] = bh['portfolio_value'].pct_change().fillna(0)
    return bh


# ============================================================
# Result I/O
# ============================================================

def save_results(portfolio_histories, order_books, strategy_name):
    """
    Save backtest results for all assets to Parquet files.

    Args:
        portfolio_histories: dict[ticker → DataFrame]
        order_books: dict[ticker → DataFrame]
        strategy_name: string like 'technical', 'ml_lgbm', 'dl_lstm'
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # combine all assets into single DataFrames
    all_portfolios = []
    all_orders = []

    for ticker, ph in portfolio_histories.items():
        if ph is not None and len(ph) > 0:
            all_portfolios.append(ph)

    for ticker, ob in order_books.items():
        if ob is not None and len(ob) > 0:
            all_orders.append(ob)

    if all_portfolios:
        combined_ph = pd.concat(all_portfolios)
        combined_ph.to_parquet(os.path.join(RESULTS_DIR, f'{strategy_name}_results.parquet'))

    if all_orders:
        combined_ob = pd.concat(all_orders, ignore_index=True)
        combined_ob.to_parquet(os.path.join(RESULTS_DIR, f'{strategy_name}_orderbook.parquet'))


def load_results(strategy_name):
    """Load saved results for a strategy."""
    ph_path = os.path.join(RESULTS_DIR, f'{strategy_name}_results.parquet')
    ob_path = os.path.join(RESULTS_DIR, f'{strategy_name}_orderbook.parquet')

    portfolio_history = pd.read_parquet(ph_path) if os.path.exists(ph_path) else pd.DataFrame()
    order_book = pd.read_parquet(ob_path) if os.path.exists(ob_path) else pd.DataFrame()
    return portfolio_history, order_book


# ============================================================
# Common Backtest Start Date
# ============================================================

def get_common_start_date(feature_dfs):
    """
    Determine the common backtest start date based on ML training window.
    This is the earliest date at which ML can start making predictions.
    All strategies start trading from this date for fair comparison.

    Args:
        feature_dfs: dict[ticker → DataFrame] with DatetimeIndex

    Returns:
        pd.Timestamp: the common start date
    """
    from config import ML_TRAIN_WINDOW

    # find the latest "first usable date" across all assets
    # each asset needs ML_TRAIN_WINDOW days of data before the start date
    start_dates = []
    for ticker, df in feature_dfs.items():
        valid_dates = df.dropna(subset=['target']).index
        if len(valid_dates) > ML_TRAIN_WINDOW:
            start_dates.append(valid_dates[ML_TRAIN_WINDOW])

    if not start_dates:
        raise ValueError("Not enough data for any asset to start backtesting.")

    # use the latest start date so all assets can participate
    return max(start_dates)
