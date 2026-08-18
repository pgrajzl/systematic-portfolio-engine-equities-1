"""
momentum.py

Momentum signal: ranks stocks by trailing price return over a
lookback window. Higher score = stronger recent uptrend, more
attractive under this signal. See notebooks/signal_library.ipynb for
the full writeup (rationale, definition, known weaknesses).
"""

import pandas as pd


def compute_momentum(close, lookback=63):
    """
    Computes trailing momentum for every stock.

    close: DataFrame of daily close prices, shaped dates x tickers
    lookback: number of trading days to look back (default 63,
        roughly 3 months)

    Returns a DataFrame of the same shape as `close`.
    """
    return close.pct_change(lookback)