"""
cost_aware_backtest.py

Same as naive_backtest.py, but subtracts a transaction cost from each
day's return, proportional to that day's turnover. This is the more
realistic version -- see naive_backtest.py for the frictionless
baseline this builds on.

Cost model is intentionally simple: a fixed cost per unit of turnover
(e.g. 0.0005 = 5 basis points), applied uniformly regardless of which
stock is being traded. A more refined version could vary cost by
liquidity, stock, or trade size, but that's a later refinement.
"""

import pandas as pd
import numpy as np

from modules.backtest.naive_backtest import compute_turnover, compute_performance_metrics
from modules.allocation.simple_allocation import compute_strategy_returns


def run_cost_aware_backtest(weights, close, cost_per_unit_turnover=0.0005, initial_capital=1.0):
    """
    Runs a backtest that subtracts transaction costs from daily
    returns, proportional to turnover.

    cost_per_unit_turnover: cost as a fraction, applied per unit of
        turnover (default 0.0005 = 5 basis points -- a standard,
        moderate assumption for a liquid large-cap universe)

    Returns a DataFrame with daily_return, turnover, cost, net_return,
    and equity_curve columns.
    """
    gross_returns = compute_strategy_returns(weights, close)
    turnover = compute_turnover(weights)

    # Align both series on the same dates before combining
    combined = pd.DataFrame({"daily_return": gross_returns, "turnover": turnover}).dropna()

    combined["cost"] = combined["turnover"] * cost_per_unit_turnover
    combined["net_return"] = combined["daily_return"] - combined["cost"]

    combined["equity_curve"] = initial_capital * (1 + combined["net_return"]).cumprod()

    return combined


def compare_naive_vs_cost_aware(weights, close, cost_per_unit_turnover=0.0005):
    from modules.backtest.naive_backtest import run_backtest

    naive_result = run_backtest(weights, close)
    naive_metrics = compute_performance_metrics(naive_result)

    cost_aware_result = run_cost_aware_backtest(weights, close, cost_per_unit_turnover)

    # Build a clean two-column frame for metrics calculation, avoiding
    # the duplicate "daily_return" column name collision that occurs
    # if net_return is renamed directly on cost_aware_result (which
    # already has its own "daily_return" column for the pre-cost
    # gross return)
    cost_aware_for_metrics = pd.DataFrame({
        "daily_return": cost_aware_result["net_return"],
        "equity_curve": cost_aware_result["equity_curve"],
    })
    cost_aware_metrics = compute_performance_metrics(cost_aware_for_metrics)

    total_cost_drag = naive_metrics["Total Return"] - cost_aware_metrics["Total Return"]

    comparison = pd.DataFrame({
        "Naive (No Costs)": naive_metrics,
        "Cost-Aware": cost_aware_metrics,
    })

    return comparison, total_cost_drag