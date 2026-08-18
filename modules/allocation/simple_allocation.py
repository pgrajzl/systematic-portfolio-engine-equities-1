"""
simple_allocation.py

Combines multiple already-optimized strategies into one final
portfolio. Each input strategy is assumed to already be fully
constructed, risk-neutralized, and optimized on its own -- this
module only decides how much capital each one gets, then sums them
into a single combined weight vector.

Two simple methods, no optimizer:
- Equal-weight: fixed split across strategies, regardless of behavior.
- Risk-parity: sized inversely to each strategy's own recent
  realized volatility, so a more volatile strategy gets less capital.

A full allocation optimizer (accounting for correlation BETWEEN
strategies, not just each one's own volatility) is a natural future
upgrade, mirroring how construction/ and optimization/ relate.
"""

import pandas as pd


def compute_strategy_returns(weights, close):
    """
    Computes a strategy's own realized daily return, given its
    position weights and price data: the weighted sum of each held
    stock's daily return.
    """
    returns = close.pct_change()
    common_tickers = weights.columns.intersection(returns.columns)
    strategy_returns = (weights[common_tickers].shift(1) * returns[common_tickers]).sum(axis=1)
    return strategy_returns


def equal_weight_allocation(strategy_weights_dict):
    """
    Combines strategies with a fixed, equal capital split.

    strategy_weights_dict: dict of {strategy_name: weights_df}, each
        already optimized on its own

    Returns a single combined weights DataFrame.
    """
    n_strategies = len(strategy_weights_dict)
    allocation = 1.0 / n_strategies

    combined = None
    for name, weights in strategy_weights_dict.items():
        scaled = weights * allocation
        combined = scaled if combined is None else combined.add(scaled, fill_value=0.0)

    return combined


def risk_parity_allocation(strategy_weights_dict, close, vol_window=60):
    """
    Combines strategies with capital sized inversely to each
    strategy's own trailing realized volatility, so no single
    strategy dominates the combined book's risk purely because it
    happens to be more volatile.

    vol_window: number of trailing days used to estimate each
        strategy's realized volatility
    """
    strategy_returns = {
        name: compute_strategy_returns(weights, close)
        for name, weights in strategy_weights_dict.items()
    }

    strategy_vols = {
        name: returns.rolling(vol_window).std()
        for name, returns in strategy_returns.items()
    }

    vol_df = pd.DataFrame(strategy_vols)
    inv_vol = 1.0 / vol_df
    allocation_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)  # normalize to sum to 1 each date

    combined = None
    for name, weights in strategy_weights_dict.items():
        # Align the per-date allocation weight to each strategy's own position weights
        strategy_allocation = allocation_weights[name].reindex(weights.index)
        scaled = weights.mul(strategy_allocation, axis=0)
        combined = scaled if combined is None else combined.add(scaled, fill_value=0.0)

    return combined