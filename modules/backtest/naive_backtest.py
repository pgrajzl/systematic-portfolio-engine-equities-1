"""
naive_backtest.py

Frictionless historical simulation: compounds a portfolio's daily
returns into an equity curve, and computes standard performance
metrics. No transaction costs, no turnover penalty -- this is the
simple baseline. See cost_aware_backtest.py for the version that
accounts for trading costs.
"""

import pandas as pd
import numpy as np

from modules.allocation.simple_allocation import compute_strategy_returns


def run_backtest(weights, close, initial_capital=1.0):
    """
    Runs a frictionless backtest: computes the portfolio's daily
    return series from its position weights, then compounds it into
    an equity curve starting from initial_capital.

    weights: DataFrame of portfolio weights, shaped dates x tickers
    close: DataFrame of daily close prices, same tickers
    initial_capital: starting portfolio value (default 1.0, so the
        equity curve reads as a growth multiple)

    Returns a DataFrame with daily_return and equity_curve columns.
    """
    daily_returns = compute_strategy_returns(weights, close)
    daily_returns = daily_returns.dropna()

    equity_curve = initial_capital * (1 + daily_returns).cumprod()

    return pd.DataFrame({
        "daily_return": daily_returns,
        "equity_curve": equity_curve,
    })


def compute_performance_metrics(backtest_result, periods_per_year=252):
    """
    Standard performance metrics from a backtest result: total return,
    annualized return, annualized volatility, Sharpe ratio (assuming
    a 0% risk-free rate for simplicity), and max drawdown.
    """
    returns = backtest_result["daily_return"]
    equity = backtest_result["equity_curve"]

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_periods = len(returns)
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    annualized_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annualized_return / annualized_vol if annualized_vol != 0 else float("nan")

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
    }


def compute_turnover(weights):
    """
    Computes daily turnover: the sum of absolute changes in each
    stock's weight from one day to the next. Not used to adjust
    returns in this naive version -- just measured and reported, so
    it's available as an input for the cost-aware version later.
    """
    return weights.diff().abs().sum(axis=1)