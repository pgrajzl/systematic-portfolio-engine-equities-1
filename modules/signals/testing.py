"""
testing.py

Standard signal evaluation for the signals module. Every signal in
this folder gets tested the same way: compute forward returns over a
chosen horizon, then compute the Information Coefficient (IC) --
the cross-sectional rank correlation between the signal's scores and
what actually happened next.

This is the standard, first-pass check for whether a signal shows any
real predictive power before it's ever allowed downstream into
portfolio construction.
"""

import pandas as pd
import warnings
from scipy.stats import spearmanr, ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)


def compute_forward_returns(close, horizon):
    """
    Forward return over `horizon` days, aligned so the value on date T
    represents the return from T to T+horizon -- i.e. what you'd earn
    holding a position entered on date T.
    """
    return close.pct_change(horizon).shift(-horizon)


def compute_ic_series(signal_df, forward_returns_df):
    """
    For each date, computes the Spearman rank correlation between the
    signal's cross-sectional scores and forward returns across all
    tickers. Returns a Series of IC values indexed by date.
    """
    ic_values = {}
    for date in signal_df.index:
        sig_row = signal_df.loc[date].dropna()
        ret_row = forward_returns_df.loc[date].dropna()

        common = sig_row.index.intersection(ret_row.index)
        if len(common) < 5:
            continue

        ic, _ = spearmanr(sig_row[common], ret_row[common])
        ic_values[date] = ic

    return pd.Series(ic_values)


def summarize_ic(ic_series):
    """Standard IC summary stats: mean, std, information ratio, hit rate."""
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    ir = mean_ic / std_ic if std_ic != 0 else float("nan")
    pct_positive = (ic_series > 0).mean()

    return {
        "Mean IC": mean_ic,
        "IC Std": std_ic,
        "Information Ratio": ir,
        "% Positive IC": pct_positive,
        "N Observations": len(ic_series),
    }


def evaluate_signal(signal_df, close, horizon):
    """
    One-call convenience function: computes forward returns, IC
    series, and summary stats for any signal, given a chosen horizon.
    """
    fwd_returns = compute_forward_returns(close, horizon)
    ic_series = compute_ic_series(signal_df, fwd_returns)
    summary = summarize_ic(ic_series)
    return ic_series, summary