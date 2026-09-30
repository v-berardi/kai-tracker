"""Downloads stock prices with yfinance.

Note: yfinance gives delayed data, not real-time. That is fine here
because we only look at daily prices.
"""

import logging
import time

import pandas as pd
import yfinance as yf

from src.config import (
    BENCHMARKS,
    HTTP_BACKOFF_BASE,
    HTTP_MAX_RETRIES,
    PRICE_HISTORY_PERIOD,
    PRICE_INTERVAL,
    TICKERS,
)
from src.storage.db import upsert_prices

logger = logging.getLogger(__name__)


def fetch_prices(
    tickers: list[str] | None = None,
    period: str = PRICE_HISTORY_PERIOD,
    interval: str = PRICE_INTERVAL,
) -> pd.DataFrame:
    """Download price history and return one tidy DataFrame.

    Output columns: ticker, date, open, high, low, close, volume.
    Retries a few times before giving up (the API sometimes fails or
    rate-limits us). Raises RuntimeError if nothing could be downloaded.
    """
    tickers = tickers or list(TICKERS.keys())
    last_err: Exception | None = None

    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw is None or raw.empty:
                raise ValueError("yfinance returned an empty DataFrame")
            return _to_tidy(raw, tickers)
        except Exception as err:  # network error, bad data, rate limit...
            last_err = err
            wait = HTTP_BACKOFF_BASE ** attempt
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.0fs.",
                attempt, HTTP_MAX_RETRIES, err, wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"Failed to download prices: {last_err}")


def _to_tidy(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Turn yfinance's multi-index table into one simple, tidy table."""
    frames = []
    for t in tickers:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            logger.warning("Ticker missing from response: %s", t)
            continue
        sub = sub.dropna(subset=["Close"])
        if sub.empty:
            logger.warning("No usable data for %s", t)
            continue
        tidy = pd.DataFrame(
            {
                "ticker": t,
                "date": sub.index.strftime("%Y-%m-%d"),
                "open": sub["Open"].values,
                "high": sub["High"].values,
                "low": sub["Low"].values,
                "close": sub["Close"].values,
                "volume": sub["Volume"].values,
            }
        )
        frames.append(tidy)

    if not frames:
        raise RuntimeError("No ticker returned any data")
    return pd.concat(frames, ignore_index=True)


def run(tickers: list[str] | None = None) -> int:
    """Full pipeline: download prices, then save them. Returns rows written.

    By default it also downloads the market indexes (BENCHMARKS in the
    config), because the market model of the event study needs them.
    """
    if tickers is None:
        tickers = list(TICKERS.keys()) + sorted(set(BENCHMARKS.values()))
    df = fetch_prices(tickers)
    n = upsert_prices(df)
    logger.info("Prices stored: %d rows (%d tickers)", n, df["ticker"].nunique())
    return n
