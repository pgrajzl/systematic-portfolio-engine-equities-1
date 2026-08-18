"""
factor_exposure.py

Measures a portfolio's exposure to systematic factors -- currently
market beta, with sector exposure also covered. Built to be
extensible: the core exposure-measurement logic is factor-agnostic,
so adding new factors later (Fama-French value/size/quality, a
momentum factor, etc.) means computing a new factor return series and
reusing the same measurement functions below, not rewriting this file.

This module only MEASURES exposure -- it does not adjust or correct
anything. See neutralization.py for the (currently simple, naive)
correction step. A proper constrained optimizer, which will use the
exposures measured here as constraints, belongs in optimization/ and
is not built yet.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Beta estimation
# ---------------------------------------------------------------------

def compute_rolling_beta(close, market_ticker="SPY", window=252):
    """
    Computes each stock's rolling beta to a market proxy, using price
    history already loaded (no external beta source -- fully
    self-contained and reproducible from data we already have).

    Beta here is the standard regression slope: how much a stock's
    daily return tends to move for a 1-unit move in the market's
    daily return, estimated via rolling covariance / variance
    (mathematically equivalent to a rolling single-variable OLS beta,
    much faster to compute than looping a regression per stock).

    close: DataFrame of daily close prices, shaped dates x tickers.
        Must include market_ticker as one of the columns.
    market_ticker: which column in `close` to use as the market proxy
        (default SPY, a standard broad-market ETF)
    window: rolling window length in trading days (default 252, ~1 year)

    Returns a DataFrame of the same shape as `close` (minus the market
    column itself), with each stock's rolling beta at every date.
    Early dates (before a full window of history exists) are NaN.
    """
    if market_ticker not in close.columns:
        raise ValueError(
            f"'{market_ticker}' not found in close price columns -- "
            f"beta calculation requires the market proxy's own price "
            f"history to be present in the same DataFrame."
        )

    returns = close.pct_change()
    market_returns = returns[market_ticker]

    stock_returns = returns.drop(columns=[market_ticker])

    betas = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype=float)

    for ticker in stock_returns.columns:
        rolling_cov = stock_returns[ticker].rolling(window).cov(market_returns)
        rolling_var = market_returns.rolling(window).var()
        betas[ticker] = rolling_cov / rolling_var

    return betas


# ---------------------------------------------------------------------
# Generic factor exposure measurement (factor-agnostic -- reusable for
# beta, and later for any other factor loading, e.g. Fama-French)
# ---------------------------------------------------------------------

def compute_portfolio_factor_exposure(weights, factor_loadings):
    """
    Computes a portfolio's NET exposure to a given factor, on every
    date: the weighted sum of each held stock's factor loading,
    weighted by its position size.

    This is factor-agnostic by design -- pass in beta loadings to get
    beta exposure, or (later) Fama-French factor loadings to get
    exposure to that factor. Same formula either way:

        portfolio_exposure = sum(weight_i * factor_loading_i)

    weights: DataFrame of portfolio weights, shaped dates x tickers
        (positive = long, negative = short, from construction/)
    factor_loadings: DataFrame of the same shape, containing each
        stock's loading on the factor in question (e.g. rolling beta
        from compute_rolling_beta above)

    Returns a Series indexed by date: the portfolio's net exposure to
    this factor on each date. A value near 0 means the portfolio is
    roughly neutral to this factor; a large positive/negative value
    means a real, unintended directional bet on it.
    """
    # Align columns in case weights and factor_loadings don't share
    # the exact same ticker set on a given date (e.g. a stock missing
    # from one but not the other)
    common_tickers = weights.columns.intersection(factor_loadings.columns)

    exposure = (weights[common_tickers] * factor_loadings[common_tickers]).sum(axis=1)
    return exposure


# ---------------------------------------------------------------------
# Sector exposure measurement
# ---------------------------------------------------------------------

def compute_sector_exposure(weights, universe):
    """
    Computes a portfolio's NET exposure to each sector, on every date:
    the sum of position weights for all stocks in that sector.

    weights: DataFrame of portfolio weights, shaped dates x tickers
    universe: DataFrame with columns Symbol, Sector (from
        signals/fetch.py's get_sp500_universe)

    Returns a DataFrame shaped dates x sectors, where each cell is
    the portfolio's net weight in that sector on that date. A sector
    with a large positive or negative value means the portfolio is
    making an unintended directional bet on that sector, rather than
    pure stock-vs-stock bets within it.
    """
    sector_map = dict(zip(universe["Symbol"], universe["Sector"]))
    sectors = sorted(universe["Sector"].unique())

    sector_exposure = pd.DataFrame(index=weights.index, columns=sectors, dtype=float)

    for sector in sectors:
        sector_tickers = [t for t, s in sector_map.items() if s == sector and t in weights.columns]
        if not sector_tickers:
            continue
        sector_exposure[sector] = weights[sector_tickers].sum(axis=1)

    return sector_exposure