"""
portfolio_dashboard.py

Naive monitoring dashboard: takes a portfolio's weights, backtest
result, and exposure history (already computed elsewhere in the
pipeline), and presents them together as one view -- equity curve,
drawdown, rolling volatility, factor exposure over time, and
turnover. This module does not compute anything new; it only
organizes and displays what backtest/, risk/, and allocation/ already
produce, so any strategy or combined portfolio that flows through the
pipeline can be monitored the same way.

Intentionally simple for now -- a starting point, likely to change as
the pipeline grows.
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def compute_rolling_drawdown(equity_curve):
    """Rolling drawdown from the running peak, as a percent."""
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max


def compute_rolling_volatility(daily_returns, window=20, periods_per_year=252):
    """Rolling annualized volatility, over a trailing window."""
    return daily_returns.rolling(window).std() * (periods_per_year ** 0.5)


def build_monitoring_summary(backtest_result, weights, betas=None, universe=None):
    """
    Assembles a single summary DataFrame combining equity curve,
    drawdown, rolling volatility, and turnover -- one row per date,
    everything a monitoring dashboard needs in one place.

    backtest_result: DataFrame with daily_return and equity_curve
        columns, from naive_backtest.py or cost_aware_backtest.py
    weights: the portfolio's position weights, same dates
    betas, universe: optional, if provided also includes beta and
        sector exposure over time
    """
    from modules.backtest.naive_backtest import compute_turnover

    summary = pd.DataFrame(index=backtest_result.index)
    summary["equity_curve"] = backtest_result["equity_curve"]
    summary["daily_return"] = backtest_result["daily_return"]
    summary["drawdown"] = compute_rolling_drawdown(summary["equity_curve"])
    summary["rolling_vol"] = compute_rolling_volatility(summary["daily_return"])
    summary["turnover"] = compute_turnover(weights).reindex(summary.index)

    if betas is not None:
        from modules.risk.factor_exposure import compute_portfolio_factor_exposure
        summary["beta_exposure"] = compute_portfolio_factor_exposure(weights, betas).reindex(summary.index)

    if universe is not None:
        from modules.risk.factor_exposure import compute_sector_exposure
        sector_exposure = compute_sector_exposure(weights, universe)
        summary["max_abs_sector_exposure"] = sector_exposure.abs().max(axis=1).reindex(summary.index)

    return summary


def plot_monitoring_dashboard(summary):
    """
    A single, multi-panel plot: equity curve, drawdown, rolling
    volatility, and turnover, stacked so they can all be scanned at
    once. Beta and sector exposure are only plotted if present in
    the summary.
    """
    has_beta = "beta_exposure" in summary.columns
    has_sector = "max_abs_sector_exposure" in summary.columns
    n_panels = 4 + int(has_beta) + int(has_sector)

    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 2.5 * n_panels), sharex=True)

    axes[0].plot(summary.index, summary["equity_curve"], color="black", linewidth=1.2)
    axes[0].set_title("Equity Curve")
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(summary.index, summary["drawdown"], 0, color="red", alpha=0.4)
    axes[1].set_title("Drawdown")
    axes[1].grid(alpha=0.3)

    axes[2].plot(summary.index, summary["rolling_vol"], color="darkblue", linewidth=1.2)
    axes[2].set_title("Rolling 20-Day Annualized Volatility")
    axes[2].grid(alpha=0.3)

    axes[3].plot(summary.index, summary["turnover"], color="darkorange", linewidth=1)
    axes[3].set_title("Daily Turnover")
    axes[3].grid(alpha=0.3)

    next_idx = 4
    if has_beta:
        axes[next_idx].plot(summary.index, summary["beta_exposure"], color="purple", linewidth=1)
        axes[next_idx].axhline(0, color="grey", linestyle="--", linewidth=0.8)
        axes[next_idx].set_title("Beta Exposure")
        axes[next_idx].grid(alpha=0.3)
        next_idx += 1

    if has_sector:
        axes[next_idx].plot(summary.index, summary["max_abs_sector_exposure"], color="teal", linewidth=1)
        axes[next_idx].set_title("Max Absolute Sector Exposure")
        axes[next_idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()