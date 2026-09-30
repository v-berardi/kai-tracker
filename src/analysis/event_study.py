"""Event study: checks if stock returns move after a sentiment spike.

This is a simple version of a method called an "event study", used in
finance to check if something (here, a news sentiment spike) is
followed by unusual stock returns. Steps:

  1. EVENTS: days where the average FinBERT sentiment score is very high
     or very low. A 5-day cooldown avoids counting the same news spike
     more than once.

  2. ESTIMATION WINDOW [-120, -21] (about 100 trading days before the
     event): this is used to compute what a "normal" daily return looks
     like for this stock. There are two models:
       - market model (default): I fit a linear regression
           stock_return = alpha + beta * index_return
         on the estimation window. Then the normal return of a day in
         the event window is alpha + beta * (index return of that day).
         So if the whole market goes up, it is not counted as abnormal.
       - constant mean (older, simpler): normal_return = average of the
         daily returns over the estimation window. Its problem: if the
         stock was going down during this period, any rebound later looks
         "abnormal", even with no news. I keep it only to compare.
     The window stops 21 days before the event (not right up to day -1)
     so that any price run-up just before the news doesn't leak into
     the "normal" baseline and make the event look smaller than it is.

  3. EVENT WINDOW [-5, +5] (11 trading days around the event):
     Abnormal Return (AR) = actual return - normal return, for each day.
     Cumulative Abnormal Return (CAR) = sum of the AR values, one number
     per event.
     +/-5 trading days is about a week on each side: long enough to see
     if the market keeps reacting after the news, short enough that a
     different, unrelated news story is unlikely to land in the same
     window and confuse the result.
     The events must be independent: if the window of an event overlaps
     the previous one, I skip it, and the days inside event windows are
     removed from the estimation window of the other events (see
     compute_cars).

  4. RESULT:
     CAAR = average CAR across all events.
     A t-test checks if CAAR is really different from 0, or if it could
     just be random noise.
     I also split the CAR in two parts, with one t-test for each:
       - before the event: days [-5, -1]
       - from the event day: days [0, +5]
     Only the second part tells if returns move AFTER the news. The first
     part shows if the price was already moving before: headlines often
     talk about a move that already happened ("Nvidia shares jump"), so
     a big "before" CAR means the news follows the price and not the
     opposite.

What we expect: the t-test probably will NOT be significant, because a
daily news sentiment score is not a proven trading signal by itself.
That is a normal and honest result, not a bug.
"""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import BENCHMARKS, TICKER_TOPICS
from src.storage.db import daily_sentiment, get_connection

logger = logging.getLogger(__name__)


class StudyResult(NamedTuple):
    ticker: str
    direction: str          # 'positive', 'negative', or 'all'
    n_events: int
    caar: float             # Cumulative Average Abnormal Return, full window
    t_stat: float
    p_value: float
    caar_pre: float         # same but only the days before the event [-5, -1]
    t_pre: float
    p_pre: float
    caar_post: float        # same but from the event day [0, +5]
    t_post: float
    p_post: float
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


def _load_returns(ticker: str, benchmark: str | None = None) -> pd.DataFrame:
    """Daily returns of the stock ("ret") and of its index ("ret_m").

    Without benchmark, only "ret" (for the constant mean model).
    With a benchmark, I first keep only the days where both have a price,
    and I compute the returns after that. If I compute them before, a day
    missing for one of the two would give returns over different periods.
    """
    stock = _load_prices(ticker)
    if benchmark is None:
        return stock[["date", "ret"]]
    try:
        market = _load_prices(benchmark)
    except ValueError:
        raise ValueError(
            f"No prices for the index {benchmark}. Run "
            f"scripts/run_ingestion.py again to download it."
        ) from None
    df = stock[["date", "close"]].merge(
        market[["date", "close"]], on="date", suffixes=("", "_m")
    )
    df["ret"] = df["close"].pct_change()
    df["ret_m"] = df["close_m"].pct_change()
    return df.dropna(subset=["ret", "ret_m"]).reset_index(drop=True)


def compute_cars(
    ticker: str,
    events: pd.DataFrame,
    estimation_window: tuple[int, int] = (-120, -21),
    event_window: tuple[int, int] = (-5, 5),
    min_estimation_days: int = 30,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """Compute the CAR (Cumulative Abnormal Return) for each event.

    benchmark: ticker of the market index. If given, I use the market
    model, if None the constant mean model.

    Returns a DataFrame with: event_date, direction, avg_score,
    normal_return (average normal return per day in the event window),
    beta (market model only, NaN otherwise), car (full window),
    car_pre (days before the event), car_post (event day and after),
    ar_series (the AR value for each day).

    The windows count trading days, not calendar days. This matters: if
    an event happens on a Friday, the next day in the window is the
    following Monday, not the actual Saturday.

    Two rules so that the events stay independent:
    - No overlap: if an event starts less than one event window after the
      last kept event, I skip it. If not, the same days are counted in
      two CARs, the CARs are not independent anymore and the t-test looks
      more significant than it really is.
    - Clean baseline: I remove the days inside any event window from the
      estimation window of the other events, so the reaction to one news
      does not change the "normal" return of another event.
      min_estimation_days = minimum number of clean days I need
      (30 days is a usual minimum to estimate a mean).
    """
    prices = _load_returns(ticker, benchmark)
    trading_dates = prices["date"].values          # numpy datetime64
    returns = prices["ret"].values
    market_returns = prices["ret_m"].values if benchmark else None
    n_days = len(trading_dates)

    ev_len = event_window[1] - event_window[0] + 1  # number of days in the window
    n_pre = max(-event_window[0], 0)                # days before day 0 in the window

    # 1. find the trading day of each event
    positions: list[tuple[int, pd.Series]] = []
    for _, event in events.sort_values("date").iterrows():
        event_dt = np.datetime64(event["date"], "ns")
        # index of the closest trading day on/after the event date
        idx = int(np.searchsorted(trading_dates, event_dt))
        if idx >= n_days:
            logger.debug("Event %s out of range, skipped", event["date"].date())
            continue
        positions.append((idx, event))

    # 2. mark all the days inside an event window (for the clean baseline)
    in_event_window = np.zeros(n_days, dtype=bool)
    for idx, _ in positions:
        start = max(idx + event_window[0], 0)
        in_event_window[start : idx + event_window[1] + 1] = True

    share = in_event_window.mean()
    if share > 0.5:
        logger.warning(
            "%.0f%% of the trading days are in an event window, the events "
            "are not rare so it is hard to find a clean baseline. Try a "
            "higher threshold.", 100 * share,
        )

    # 3. CAR of each event with enough data and no overlap
    results = []
    last_kept_idx: int | None = None
    skipped = {"not enough data": 0, "overlap": 0, "baseline too short": 0}

    for idx, event in positions:
        est_s = idx + estimation_window[0]
        est_e = idx + estimation_window[1]
        ev_s  = idx + event_window[0]
        ev_e  = idx + event_window[1]

        if est_s < 0 or ev_e >= n_days:
            logger.debug("Not enough data for event %s, skipped", event["date"].date())
            skipped["not enough data"] += 1
            continue

        if last_kept_idx is not None and idx - last_kept_idx < ev_len:
            logger.debug("Event %s overlaps the previous event window, skipped",
                         event["date"].date())
            skipped["overlap"] += 1
            continue

        clean_days = ~in_event_window[est_s : est_e + 1]
        clean = returns[est_s : est_e + 1][clean_days]
        if len(clean) < min_estimation_days:
            logger.debug("Event %s: only %d clean estimation days, skipped",
                         event["date"].date(), len(clean))
            skipped["baseline too short"] += 1
            continue

        ev_returns = returns[ev_s : ev_e + 1]

        if benchmark:
            # market model: linear regression of the stock on the index
            # (np.polyfit with degree 1 = ordinary least squares)
            clean_m = market_returns[est_s : est_e + 1][clean_days]
            beta, alpha = np.polyfit(clean_m, clean, 1)
            normal_ret = alpha + beta * market_returns[ev_s : ev_e + 1]
        else:
            # constant mean: the same normal return for every day
            beta = np.nan
            normal_ret = np.full(ev_len, clean.mean())

        ar = ev_returns - normal_ret          # Abnormal Return, per day
        car = float(ar.sum())                 # Cumulative Abnormal Return

        results.append({
            "event_date": event["date"],
            "direction": event["direction"],
            "avg_score": event["avg_score"],
            "normal_return": float(normal_ret.mean()),
            "beta": float(beta),
            "car": car,
            "car_pre": float(ar[:n_pre].sum()),
            "car_post": float(ar[n_pre:].sum()),
            "ar_series": ar.tolist(),
        })
        last_kept_idx = idx

    # + the events after the last price date (they are not in `positions`)
    skipped["not enough data"] += len(events) - len(positions)
    logger.info("Events kept: %d of %d (skipped: %s)", len(results), len(events),
                ", ".join(f"{k} {v}" for k, v in skipped.items()))
    return pd.DataFrame(results)


# -- Step 3: combine results and run the significance test --

def _t_test(values: np.ndarray) -> tuple[float, float, float]:
    """One sample t-test, H0: mean = 0. Returns (mean, t_stat, p_value).

    t = mean / (std / sqrt(n)), with n - 1 degrees of freedom.
    If all the values are equal, std = 0 and we can't compute t, so I
    return t = 0 to not divide by zero.
    """
    n = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    t_stat = mean / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))
    return mean, t_stat, p_value


def aggregate(
    cars: pd.DataFrame,
    event_window: tuple[int, int] = (-5, 5),
    direction: str = "all",
    ticker: str = "",
) -> StudyResult | None:
    """Compute CAAR, t-stat, and p-value across all events.

    We test H0: CAAR = 0 (sentiment is not linked to abnormal returns)
    three times: on the full window, before the event and after it.
    If p < 0.05, the result is called significant. Based on the
    limitations of this method, we expect it usually will NOT be.
    """
    sub = cars if direction == "all" else cars[cars["direction"] == direction]
    n = len(sub)
    if n < 2:
        logger.warning("Too few events (%d) to run a statistical test.", n)
        return None

    caar, t_stat, p_value = _t_test(sub["car"].values)
    caar_pre, t_pre, p_pre = _t_test(sub["car_pre"].values)
    caar_post, t_post, p_post = _t_test(sub["car_post"].values)

    ar_matrix = np.array(sub["ar_series"].tolist())   # shape: (n_events, days_in_window)
    caar_cumulative = ar_matrix.mean(axis=0).cumsum().tolist()

    logger.info(
        "[%s | %s] n=%d  CAAR=%.4f (before %.4f, after %.4f)  p_after=%.3f",
        ticker, direction, n, caar, caar_pre, caar_post, p_post,
    )
    return StudyResult(
        ticker=ticker,
        direction=direction,
        n_events=n,
        caar=caar,
        t_stat=t_stat,
        p_value=p_value,
        caar_pre=caar_pre,
        t_pre=t_pre,
        p_pre=p_pre,
        caar_post=caar_post,
        t_post=t_post,
        p_post=p_post,
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
    model: str = "market",
) -> dict[str, StudyResult | None]:
    """Run the full event study for one ticker.

    query_tags: the news topics to use. If None, I take the topics of
    this ticker in config.TICKER_TOPICS, so the events come from news
    about this company and not about another one.
    model: "market" (compare to the index in config.BENCHMARKS) or
    "constant" (average return of the stock).

    Returns a dict {direction: StudyResult} with keys
    'positive', 'negative', 'all'.
    """
    if model not in ("market", "constant"):
        raise ValueError(f"Unknown model: {model} (use 'market' or 'constant')")
    benchmark = BENCHMARKS.get(ticker) if model == "market" else None
    if model == "market" and benchmark is None:
        raise ValueError(f"No index for {ticker} in BENCHMARKS (config.py)")
    logger.info("Model: %s%s", model, f" (index: {benchmark})" if benchmark else "")

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

    cars = compute_cars(ticker, events, estimation_window, event_window,
                        benchmark=benchmark)
    if cars.empty:
        logger.warning("Could not compute any CAR, not enough data.")
        return {}

    return {
        direction: aggregate(cars, event_window, direction, ticker)
        for direction in ("all", "positive", "negative")
    }
