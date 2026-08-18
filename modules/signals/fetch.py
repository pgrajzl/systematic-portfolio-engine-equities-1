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