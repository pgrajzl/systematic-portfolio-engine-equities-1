"""
neutralization.py

DELIBERATELY SIMPLE / NOT ROBUST. This module makes naive, direct
adjustments to cancel out unwanted factor exposure -- it's a
placeholder to prove the pipeline works end-to-end, not the real
solution. A proper constrained optimizer (which jointly handles
beta neutrality, sector neutrality, position limits, and turnover
all at once, using the exposures measured in factor_exposure.py as
constraints) belongs in optimization/ and will replace this module's
role once built. Treat everything here as a rough first pass.
"""

import pandas as pd


def sector_neutralize(weights, universe):
    """
    Naively cancels out sector exposure: for each sector on each date,
    subtracts that sector's average weight from every stock in it, so
    the sector's NET weight becomes ~0. Preserves relative
    stock-vs-stock bets within each sector, removes the sector-level
    directional tilt.

    This is a simple demeaning operation, not an optimizer -- it does
    not try to preserve gross exposure, position limits, or anything
    else while doing this adjustment.
    """
    sector_map = dict(zip(universe["Symbol"], universe["Sector"]))
    neutralized = weights.copy()

    for date in weights.index:
        row = weights.loc[date]
        active = row[row != 0]
        if active.empty:
            continue

        for sector in universe["Sector"].unique():
            sector_tickers = [t for t in active.index if sector_map.get(t) == sector]
            if len(sector_tickers) < 2:
                continue
            sector_mean = row[sector_tickers].mean()
            neutralized.loc[date, sector_tickers] = row[sector_tickers] - sector_mean

    return neutralized


def beta_neutralize(weights, betas):
    """
    Naively cancels out beta exposure: scales the short side up or
    down (uniformly) so the short side's weighted-average beta matches
    the long side's, making the portfolio's net beta exposure ~0.

    This is a crude uniform rescaling, not an optimizer -- it doesn't
    try to preserve individual position sizes intelligently, gross
    exposure targets, or anything else while doing this adjustment.
    """
    neutralized = weights.copy()

    for date in weights.index:
        row = weights.loc[date]
        if date not in betas.index:
            continue

        common = row.index.intersection(betas.columns)
        row = row[common]
        beta_row = betas.loc[date, common]

        long_mask = row > 0
        short_mask = row < 0

        if long_mask.sum() == 0 or short_mask.sum() == 0:
            continue

        long_beta_exposure = (row[long_mask] * beta_row[long_mask]).sum()
        short_beta_exposure = (row[short_mask] * beta_row[short_mask]).sum()

        if short_beta_exposure == 0:
            continue

        # Scale the short side so its beta exposure offsets the long
        # side's exactly, leaving net beta exposure at ~0
        scale_factor = -long_beta_exposure / short_beta_exposure
        neutralized.loc[date, row[short_mask].index] = row[short_mask] * scale_factor

    return neutralized