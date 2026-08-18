"""
mean_reversion.py

Mean reversion signal: ranks stocks by the NEGATIVE of their recent
trailing return. Stocks that have fallen recently score higher
(expected to bounce); stocks that have risen recently score lower.
See notebooks/signal_library.ipynb for the full writeup (rationale,
definition, known weaknesses).
"""

import pandas as pd


def compute_mean_reversion(close, lookback=5):
    """
    Computes trailing mean reversion for every stock.

    close: DataFrame of daily close prices, shaped dates x tickers
    lookback: number of trading days to look back (default 5,
        roughly 1 trading week)

    Returns a DataFrame of the same shape as `close`.
    """
    return -close.pct_change(lookback)