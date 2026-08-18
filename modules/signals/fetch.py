"""
fetch.py

Standard data-fetching utilities for the signals module. Every signal
in this folder pulls from these same functions, so all signals work
off identical, consistently-fetched data.

Currently covers: S&P 500 universe (tickers + GICS sector), and daily
close/volume price data via yfinance. Will expand to cover other data
sources (financial statements, macro data, etc.) as new signals need them.
"""

import io
import time
import pandas as pd
import requests
import yfinance as yf

START_DATE = "2019-01-01"
END_DATE = "2026-08-05"


def get_sp500_universe():
    """
    Scrapes the current S&P 500 constituent list from Wikipedia,
    including GICS sector classification.

    Returns a DataFrame with columns: Symbol, Sector.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research script)"}

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    sp500_table = tables[0]

    universe = sp500_table[["Symbol", "GICS Sector"]].copy()
    universe.columns = ["Symbol", "Sector"]
    universe["Symbol"] = universe["Symbol"].str.replace(".", "-", regex=False)
    return universe


def fetch_price_data(tickers, start=START_DATE, end=END_DATE, batch_size=40, pause=2):
    """
    Downloads daily close price and volume data for a list of tickers,
    in batches to avoid overloading yfinance in a single call.

    Returns (close_df, volume_df), both shaped dates x tickers.
    """
    all_close, all_volume = [], []
    n_batches = -(-len(tickers) // batch_size)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Fetching batch {batch_num} / {n_batches} ({len(batch)} tickers)...")

        try:
            data = yf.download(batch, start=start, end=end, auto_adjust=True)
            all_close.append(data["Close"])
            all_volume.append(data["Volume"])
        except Exception as e:
            print(f"Batch starting at index {i} failed: {e}")

        time.sleep(pause)

    close = pd.concat(all_close, axis=1)
    volume = pd.concat(all_volume, axis=1)

    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]

    return close, volume

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def save_price_data(universe, close, volume):
    """Saves universe, close, and volume to the data/ folder as CSVs."""
    DATA_DIR.mkdir(exist_ok=True)
    universe.to_csv(DATA_DIR / "sp500_universe.csv", index=False)
    close.to_csv(DATA_DIR / "close.csv")
    volume.to_csv(DATA_DIR / "volume.csv")
    print(f"Saved universe, close, and volume to {DATA_DIR}/")


def load_price_data():
    """Loads previously-saved universe, close, and volume from the data/ folder."""
    universe = pd.read_csv(DATA_DIR / "sp500_universe.csv")
    close = pd.read_csv(DATA_DIR / "close.csv", index_col=0, parse_dates=True)
    volume = pd.read_csv(DATA_DIR / "volume.csv", index_col=0, parse_dates=True)
    return universe, close, volume

def fill_short_gaps(close, market_ticker="SPY", max_gap_days=3, beta_window=60):
    """
    Fills short gaps (1 to max_gap_days consecutive missing days) in
    price data by estimating the missing price from the market's
    return over the gap, scaled by the stock's own trailing beta
    (estimated from the days right before the gap). This lets the
    price move a plausible amount during the gap, rather than sitting
    flat the way a plain forward fill would, which would otherwise
    create artificial zero-return days and understate variance/beta.

    Gaps longer than max_gap_days are left as NaN, since a longer
    absence is more likely a real event than a short data provider
    hiccup, and shouldn't be silently patched over.

    close: DataFrame of daily close prices, shaped dates x tickers.
        Must include market_ticker as one of the columns.
    beta_window: number of trading days before a gap used to estimate
        a short-term beta for filling that specific gap
    """
    if market_ticker not in close.columns:
        raise ValueError(f"'{market_ticker}' not found in close price columns.")

    filled = close.copy()
    market_returns = close[market_ticker].pct_change()

    for ticker in close.columns:
        if ticker == market_ticker:
            continue

        series = close[ticker]
        is_missing = series.isna()
        if not is_missing.any():
            continue

        # Identify runs of consecutive missing days
        gap_id = (is_missing != is_missing.shift()).cumsum()
        gap_groups = series[is_missing].groupby(gap_id[is_missing])

        for _, gap_dates in gap_groups.groups.items():
            gap_dates = list(gap_dates)
            if len(gap_dates) > max_gap_days:
                continue  # leave longer gaps as NaN

            gap_start_idx = close.index.get_loc(gap_dates[0])
            if gap_start_idx == 0:
                continue  # can't fill a gap with no prior price

            last_good_date = close.index[gap_start_idx - 1]
            last_good_price = series.loc[last_good_date]
            if pd.isna(last_good_price):
                continue

            # Estimate this stock's short-term beta from the window
            # right before the gap
            pre_gap_stock_returns = series.loc[:last_good_date].pct_change().tail(beta_window)
            pre_gap_market_returns = market_returns.loc[:last_good_date].tail(beta_window)
            cov = pre_gap_stock_returns.cov(pre_gap_market_returns)
            var = pre_gap_market_returns.var()
            local_beta = cov / var if var and not pd.isna(var) else 1.0

            # Walk forward through the gap, applying each day's SPY
            # return scaled by the local beta to the running price
            running_price = last_good_price
            for gap_date in gap_dates:
                market_return = market_returns.loc[gap_date]
                if pd.isna(market_return):
                    break  # can't estimate without a market return that day
                estimated_return = local_beta * market_return
                running_price = running_price * (1 + estimated_return)
                filled.loc[gap_date, ticker] = running_price

    return filled