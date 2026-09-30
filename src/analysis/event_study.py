"""Event study: checks if stock returns move after a sentiment spike.

This is a simple version of a method called an "event study", used in
finance to check if something (here, a news sentiment spike) is
followed by unusual stock returns. Steps:

  1. EVENTS: days where the average FinBERT sentiment score is very high
     or very low. A 5-day cooldown avoids counting the same news spike
     more than once.

  2. ESTIMATION WINDOW [-120, -21] (about 100 trading days before the
     event): this is used to compute what a "normal" daily return looks
     like for this stock, using the simplest method there is:
       normal_return = average of daily returns over that period
     A more advanced method exists (comparing the stock to a market
     index instead of just its own average), but the simple average is
     enough to answer the question here. The window stops 21 days
     before the event (not right up to day -1) so that any price
     run-up just before the news doesn't leak into the "normal"
     baseline and make the event look smaller than it is.

  3. EVENT WINDOW [-5, +5] (11 trading days around the event):
     Abnormal Return (AR) = actual return - normal return, for each day.
     Cumulative Abnormal Return (CAR) = sum of the AR values, one number
     per event.
     +/-5 trading days is about a week on each side: long enough to see
     if the market keeps reacting after the news, short enough that a
     different, unrelated news story is unlikely to land in the same
     window and confuse the result.

  4. RESULT:
     CAAR = average CAR across all events.
     A t-test checks if CAAR is really different from 0, or if it could
     just be random noise.

What we expect: the t-test probably will NOT be significant, because a
daily news sentiment score is not a proven trading signal by itself.
That is a normal and honest result, not a bug.
"""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import TICKER_TOPICS
from src.storage.db import daily_sentiment, get_connection

logger = logging.getLogger(__name__)


class StudyResult(NamedTuple):
    ticker: str
    direction: str          # 'positive', 'negative', or 'all'
    n_events: int
    caar: float             # Cumulative Average Abnormal Return
    t_stat: float
    p_value: float
    caar_by_day: list[float]   # CAAR added up day by day over the event window
    relative_days: list[int]   # [-5, -4, ..., 0, ..., +5]
    events_df: pd.DataFrame    # details for each event


# -- Step 1: find events --

def detect_events(
    sentiment: pd.DataFrame,
    threshold: float = 0.25,
    min_articles: int = 2,
    cooldown_days: int = 5,
) -> pd.DataFrame:
    """Return the days with an extreme sentiment score.

    Parameters:
    - threshold: a day counts as an event if |avg_score| >= threshold.
      0.25 was picked by looking at the score distribution: most days
      sit close to 0, so 0.25 keeps only the clearer spikes without
      cutting the sample down to almost nothing.
    - min_articles: skips days with too few articles (the average would
      be noisy with only 1 article)
    - cooldown_days: two events must be at least this many days apart,
      so one ongoing news story does not get counted as several separate
      events just because it stayed in the headlines for a few days
    """
    df = sentiment[sentiment["n_articles"] >= min_articles].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "avg_score", "n_articles", "direction"])

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["avg_score"].abs() >= threshold].sort_values("date")

    events: list[dict] = []
    last_date: pd.Timestamp | None = None

    for _, row in df.iterrows():
        if last_date is None or (row["date"] - last_date).days >= cooldown_days:
            events.append({
                "date": row["date"],
                "avg_score": float(row["avg_score"]),
                "n_articles": int(row["n_articles"]),
                "direction": "positive" if row["avg_score"] > 0 else "negative",
            })
            last_date = row["date"]

    return pd.DataFrame(events)


# -- Step 2: abnormal returns --

def _load_prices(ticker: str) -> pd.DataFrame:
    """Load prices for a ticker and compute daily returns."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date",
            conn, params=(ticker,),
        )
    if df.empty:
        raise ValueError(f"No prices in the database for {ticker}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    return df.dropna(subset=["ret"]).reset_index(drop=True)


def compute_cars(
    ticker: str,
    events: pd.DataFrame,
    estimation_window: tuple[int, int] = (-120, -21),
    event_window: tuple[int, int] = (-5, 5),
) -> pd.DataFrame:
    """Compute the CAR (Cumulative Abnormal Return) for each event.

    Returns a DataFrame with: event_date, direction, avg_score,
    normal_return, car, ar_series (the AR value for each day).

    The windows count trading days, not calendar days. This matters: if
    an event happens on a Friday, the next day in the window is the
    following Monday, not the actual Saturday.
    """
    prices = _load_prices(ticker)
    trading_dates = prices["date"].values          # numpy datetime64
    returns = prices["ret"].values

    ev_len = event_window[1] - event_window[0] + 1  # number of days in the window
    results = []

    for _, event in events.iterrows():
        event_dt = np.datetime64(event["date"], "ns")

        # index of the closest trading day on/after the event date
        idx = int(np.searchsorted(trading_dates, event_dt))
        if idx >= len(trading_dates):
            logger.debug("Event %s out of range, skipped", event["date"].date())
            continue

        est_s = idx + estimation_window[0]
        est_e = idx + estimation_window[1]
        ev_s  = idx + event_window[0]
        ev_e  = idx + event_window[1]

        if est_s < 0 or ev_e >= len(trading_dates):
            logger.debug("Not enough data for event %s, skipped", event["date"].date())
            continue

        normal_ret = returns[est_s : est_e + 1].mean()
        ev_returns = returns[ev_s : ev_e + 1]

        if len(ev_returns) < ev_len:
            continue

        ar = ev_returns - normal_ret          # Abnormal Return, per day
        car = float(ar.sum())                 # Cumulative Abnormal Return

        results.append({
            "event_date": event["date"],
            "direction": event["direction"],
            "avg_score": event["avg_score"],
            "normal_return": float(normal_ret),
            "car": car,
            "ar_series": ar.tolist(),
        })

    return pd.DataFrame(results)


# -- Step 3: combine results and run the significance test --

def aggregate(
    cars: pd.DataFrame,
    event_window: tuple[int, int] = (-5, 5),
    direction: str = "all",
    ticker: str = "",
) -> StudyResult | None:
    """Compute CAAR, t-stat, and p-value across all events.

    We test H0: CAAR = 0 (sentiment is not linked to abnormal returns).
    If p < 0.05, the result is called significant. Based on the
    limitations of this method, we expect it usually will NOT be.
    """
    sub = cars if direction == "all" else cars[cars["direction"] == direction]
    n = len(sub)
    if n < 2:
        logger.warning("Too few events (%d) to run a statistical test.", n)
        return None

    car_values = sub["car"].values
    caar = float(car_values.mean())
    se = float(car_values.std(ddof=1) / np.sqrt(n))
    t_stat = caar / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))

    ar_matrix = np.array(sub["ar_series"].tolist())   # shape: (n_events, days_in_window)
    caar_cumulative = ar_matrix.mean(axis=0).cumsum().tolist()

    logger.info(
        "[%s | %s] n=%d  CAAR=%.4f  t=%.2f  p=%.3f",
        ticker, direction, n, caar, t_stat, p_value,
    )
    return StudyResult(
        ticker=ticker,
        direction=direction,
        n_events=n,
        caar=caar,
        t_stat=t_stat,
        p_value=p_value,
        caar_by_day=caar_cumulative,
        relative_days=list(range(event_window[0], event_window[1] + 1)),
        events_df=sub.copy(),
    )


# -- Full pipeline --

def run(
    ticker: str,
    query_tags: list[str] | None = None,
    threshold: float = 0.25,
    min_articles: int = 2,
    estimation_window: tuple[int, int] = (-120, -21),
    event_window: tuple[int, int] = (-5, 5),
) -> dict[str, StudyResult | None]:
    """Run the full event study for one ticker.

    query_tags: the news topics to use. If None, I take the topics of
    this ticker in config.TICKER_TOPICS, so the events come from news
    about this company and not about another one.

    Returns a dict {direction: StudyResult} with keys
    'positive', 'negative', 'all'.
    """
    if query_tags is None:
        query_tags = TICKER_TOPICS.get(ticker)
    logger.info("News topics used for %s: %s", ticker, query_tags or "all")

    sentiment = daily_sentiment(query_tags, ticker=ticker)
    if sentiment.empty:
        logger.warning("No sentiment data in the database (topics=%s).", query_tags)
        return {}

    events = detect_events(sentiment, threshold=threshold, min_articles=min_articles)
    if events.empty:
        logger.warning("No events detected with threshold=%.2f.", threshold)
        return {}

    logger.info("%d events detected for %s", len(events), ticker)

    cars = compute_cars(ticker, events, estimation_window, event_window)
    if cars.empty:
        logger.warning("Could not compute any CAR, not enough data.")
        return {}

    return {
        direction: aggregate(cars, event_window, direction, ticker)
        for direction in ("all", "positive", "negative")
    }
