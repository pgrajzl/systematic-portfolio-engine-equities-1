"""
simple_construction.py

Simple, rule-based position construction: converts a standardized
signal into portfolio weights, with no optimizer involved. Two
approaches:

- Quantile-based: rank into long/short buckets, equal-weighted within
  each bucket.
- Score-proportional: weight directly by the standardized score
  itself, so higher conviction gets a bigger position.

Both produce a weights DataFrame (dates x tickers) where positive =
long, negative = short, and the weights on any given date are scaled
to sum to a target gross exposure.
"""

import pandas as pd


def quantile_weights(standardized_signal, long_pct=0.2, short_pct=0.2, gross_exposure=1.0):
    """
    Ranks stocks into long/short buckets by their standardized score
    on each date. Top long_pct get equal-weighted long positions;
    bottom short_pct get equal-weighted short positions.

    long_pct, short_pct: fraction of the cross-section to include on
        each side (e.g. 0.2 = top/bottom 20%)
    gross_exposure: total gross exposure the resulting weights sum to
        (long side and short side each get half of this)
    """
    weights = pd.DataFrame(index=standardized_signal.index, columns=standardized_signal.columns, dtype=float)

    for date in standardized_signal.index:
        row = standardized_signal.loc[date].dropna()
        if len(row) < 10:  # need a reasonable cross-section to rank meaningfully
            continue

        n_long = max(1, int(len(row) * long_pct))
        n_short = max(1, int(len(row) * short_pct))

        ranked = row.sort_values(ascending=False)
        long_tickers = ranked.index[:n_long]
        short_tickers = ranked.index[-n_short:]

        long_weight = (gross_exposure / 2) / n_long
        short_weight = (gross_exposure / 2) / n_short

        weights.loc[date, long_tickers] = long_weight
        weights.loc[date, short_tickers] = -short_weight

    return weights.fillna(0.0)


def score_proportional_weights(standardized_signal, gross_exposure=1.0):
    """
    Weights each stock directly proportional to its standardized
    score on each date, scaled so the long side and short side each
    sum to half of gross_exposure.
    """
    weights = pd.DataFrame(index=standardized_signal.index, columns=standardized_signal.columns, dtype=float)

    for date in standardized_signal.index:
        row = standardized_signal.loc[date].dropna()
        if len(row) < 10:
            continue

        long_scores = row[row > 0]
        short_scores = row[row < 0]

        long_total = long_scores.sum()
        short_total = short_scores.abs().sum()

        if long_total > 0:
            weights.loc[date, long_scores.index] = (long_scores / long_total) * (gross_exposure / 2)
        if short_total > 0:
            weights.loc[date, short_scores.index] = (short_scores / short_total) * (gross_exposure / 2)

    return weights.fillna(0.0)