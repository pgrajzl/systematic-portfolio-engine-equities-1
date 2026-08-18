"""
zscore.py

Cross-sectional standardization: converts any signal (raw units) into
a comparable z-score, on every date, across all stocks in that day's
universe. This is the standard "contract" every signal passes through
before position construction -- it lets completely different signals
(a percentage return, a yield ratio, anything else) be compared or
combined on equal footing.
"""

import pandas as pd


def zscore_signal(signal_df):
    """
    Z-scores a signal cross-sectionally: for each date, subtract that
    date's mean and divide by that date's standard deviation, across
    all tickers with a valid value that day.

    signal_df: DataFrame of raw signal values, shaped dates x tickers

    Returns a DataFrame of the same shape, with each date's row
    z-scored independently.
    """
    mean = signal_df.mean(axis=1)
    std = signal_df.std(axis=1)
    return signal_df.sub(mean, axis=0).div(std, axis=0)


def winsorize_signal(signal_df, lower=-3.0, upper=3.0):
    """
    Clips signal values to a fixed range, applied AFTER z-scoring.
    Prevents one extreme outlier on a given date from dominating that
    date's cross-sectional ranking.

    lower, upper: clip bounds, in standard deviations (defaults +/-3,
        a standard convention)
    """
    return signal_df.clip(lower=lower, upper=upper)


def standardize_signal(signal_df, winsorize=True, lower=-3.0, upper=3.0):
    """
    Full standardization pipeline: z-score, then optionally winsorize.
    This is the function every signal should be run through before
    moving downstream to position construction.
    """
    zscored = zscore_signal(signal_df)
    if winsorize:
        zscored = winsorize_signal(zscored, lower=lower, upper=upper)
    return zscored