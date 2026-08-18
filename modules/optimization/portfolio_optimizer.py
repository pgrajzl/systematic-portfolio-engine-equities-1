"""
portfolio_optimizer.py

The core of the pipeline: takes a signal's raw target weights, its
relevant factor exposures (beta, sector -- extensible to more later),
and the hard limits from limits/limits_config.py, and solves for the
weight vector CLOSEST to the raw targets that satisfies every
constraint simultaneously.

This replaces the naive, sequential fixes in risk/neutralization.py --
instead of applying corrections one at a time (which can undo each
other), this solves for all constraints jointly using convex
optimization.

Designed to be extensible: any new factor exposure (Fama-French,
custom factors, etc.) is just one more linear constraint added to the
same solve -- no change to the core optimization logic required as
the number of signals, factors, or limits grows.
"""

import cvxpy as cp
import numpy as np
import pandas as pd

from modules.limits import limits_config as limits


def optimize_portfolio(target_weights, betas, universe,
                         max_position_weight=limits.MAX_POSITION_WEIGHT,
                         target_gross_exposure=limits.TARGET_GROSS_EXPOSURE,
                         max_net_exposure=limits.MAX_NET_EXPOSURE,
                         max_beta_exposure=limits.MAX_BETA_EXPOSURE,
                         max_sector_exposure=limits.MAX_SECTOR_EXPOSURE):
    """
    Solves for the optimal weight vector on a SINGLE date, given raw
    target weights and the constraints to satisfy. Called once per
    rebalance date by optimize_portfolio_over_time below.

    target_weights: Series, raw signal-driven weights for one date
        (index = tickers)
    betas: Series, each stock's beta loading for that date
    universe: DataFrame with columns Symbol, Sector

    Returns a Series of optimized weights, same tickers as the input.
    """
    tickers = target_weights.index.intersection(betas.dropna().index)
    if len(tickers) < limits.MIN_POSITIONS_PER_SIDE * 2:
        return target_weights

    target = target_weights[tickers].copy()

    # Stocks with no valid beta get dropped above, which shrinks gross
    # exposure below target before optimization even starts (e.g. a
    # recently-listed stock with no 252-day rolling window yet).
    # Rescale the remaining valid positions back up to the original
    # gross exposure so that dropped stocks' capital gets redeployed
    # among the stocks we CAN measure and constrain, rather than
    # silently sitting uninvested.
    original_gross = target.abs().sum()
    if original_gross > 0:
        target = target * (target_gross_exposure / original_gross)

    target_values = target.values
    beta_vec = betas[tickers].values

    sector_map = dict(zip(universe["Symbol"], universe["Sector"]))
    sectors = sorted(set(sector_map.get(t) for t in tickers if sector_map.get(t) is not None))

    n = len(tickers)
    w = cp.Variable(n)

    objective = cp.Minimize(cp.sum_squares(w - target_values))

    constraints = [
        w <= max_position_weight,
        w >= -max_position_weight,
        cp.sum(cp.abs(w)) <= target_gross_exposure,
        cp.abs(cp.sum(w)) <= max_net_exposure,
        cp.abs(w @ beta_vec) <= max_beta_exposure,
    ]

    for sector in sectors:
        sector_mask = np.array([1.0 if sector_map.get(t) == sector else 0.0 for t in tickers])
        constraints.append(cp.abs(w @ sector_mask) <= max_sector_exposure)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in ("optimal", "optimal_inaccurate"):
        print(f"Warning: optimizer did not converge (status: {problem.status}). Returning raw targets.")
        return target_weights

    return pd.Series(w.value, index=tickers)


def optimize_portfolio_over_time(target_weights_df, betas_df, universe, **limit_overrides):
    """
    Runs optimize_portfolio for every date in target_weights_df,
    producing a fully optimized weights DataFrame over time.

    **limit_overrides lets any of the limit parameters be overridden
    for a specific run, without editing limits_config.py directly.
    """
    optimized = pd.DataFrame(index=target_weights_df.index, columns=target_weights_df.columns, dtype=float)

    for date in target_weights_df.index:
        row = target_weights_df.loc[date]
        active = row[row != 0]
        if active.empty or date not in betas_df.index:
            continue

        result = optimize_portfolio(active, betas_df.loc[date], universe, **limit_overrides)
        optimized.loc[date, result.index] = result.values

    return optimized.fillna(0.0)