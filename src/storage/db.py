"""Functions to save data to and read data from the SQLite database.

Two things to know about the schema:
- Every row has `ingested_at_utc`, the last time the row was written.
  For news it is when I collected the article (news rows are never
  updated). For prices it is the last update, because prices are
  rewritten at each run (yfinance changes the old adjusted prices after
  a split or a dividend). I only use this column to check the data, the
  event study does not use it.
- `sentiment_score` starts as NULL when a news row is added. It gets
  filled in later by a separate script, so collecting news never has to
  wait for the NLP model to load.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd

from src.config import DB_PATH, DATA_DIR, MARKET_HOURS

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker          TEXT NOT NULL,
    date            TEXT NOT NULL,          -- YYYY-MM-DD (trading day)
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL NOT NULL,
    volume          INTEGER,
    ingested_at_utc TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS news (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query_tag       TEXT NOT NULL,          -- topic of the RSS query
    title           TEXT NOT NULL,
    source          TEXT,
    url             TEXT NOT NULL UNIQUE,   -- dedup key: titles can repeat or
                                             -- get reworded, the URL doesn't
    published_utc   TEXT,
    ingested_at_utc TEXT NOT NULL,
    sentiment_score REAL,                   -- filled in later by FinBERT
    sentiment_label TEXT
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news (published_utc);
CREATE INDEX IF NOT EXISTS idx_news_tag ON news (query_tag);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Open a database connection. Commits on success, rolls back on error."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(_SCHEMA)
    logger.info("Database ready: %s", DB_PATH)


def upsert_prices(df: pd.DataFrame) -> int:
    """Insert new price rows, or update them if they already exist.

    Expects a DataFrame with columns:
    ticker, date, open, high, low, close, volume.
    Returns the number of rows processed.
    """
    if df.empty:
        return 0
    now = utc_now_iso()
    rows = [
        (
            r.ticker, r.date, r.open, r.high, r.low, r.close,
            int(r.volume) if pd.notna(r.volume) else None, now,
        )
        for r in df.itertuples(index=False)
    ]
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO prices
               (ticker, date, open, high, low, close, volume, ingested_at_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker, date) DO UPDATE SET
                 open=excluded.open, high=excluded.high, low=excluded.low,
                 close=excluded.close, volume=excluded.volume,
                 ingested_at_utc=excluded.ingested_at_utc""",
            rows,
        )
    return len(rows)


def fetch_unscored_news(limit: int | None = None) -> list[tuple[int, str]]:
    """Return (id, title) for every article that has no sentiment score yet.

    This is what makes scoring incremental: running the script again only
    scores new articles, it never redoes old ones.
    """
    sql = "SELECT id, title FROM news WHERE sentiment_score IS NULL ORDER BY id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with get_connection() as conn:
        return conn.execute(sql).fetchall()


def update_sentiment(rows: list[tuple[float, str, int]]) -> int:
    """Save scores. rows = [(score, label, news_id), ...]."""
    if not rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            "UPDATE news SET sentiment_score = ?, sentiment_label = ? WHERE id = ?",
            rows,
        )
    return len(rows)


def daily_sentiment(
    query_tags: list[str] | str | None = None,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Average sentiment per day, based on the PUBLICATION time of the
    articles (not the time I collected them).

    query_tags: one topic, a list of topics, or None for all the topics.
    ticker: if given, each article goes to the first trading day where
      the stock can react to it:
        - I convert the publication time to the local time of the exchange
        - if the article comes out after the close, it counts for the next day
      Weekends and holidays are handled later in the event study (a day
      without trading goes to the next trading day).
      If ticker is None, I just use the UTC date.

    Returns: date (YYYY-MM-DD), avg_score, n_articles.
    """
    if isinstance(query_tags, str):
        query_tags = [query_tags]
    where = "WHERE sentiment_score IS NOT NULL AND published_utc IS NOT NULL"
    params: tuple = ()
    if query_tags:
        # one "?" for each topic, so the values are still sent as
        # parameters (no SQL injection)
        placeholders = ", ".join("?" for _ in query_tags)
        where += f" AND query_tag IN ({placeholders})"
        params = tuple(query_tags)
    sql = f"SELECT published_utc, sentiment_score FROM news {where}"
    with get_connection() as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        return pd.DataFrame(columns=["date", "avg_score", "n_articles"])

    published = pd.to_datetime(df["published_utc"], utc=True, format="ISO8601")

    market = MARKET_HOURS.get(ticker) if ticker else None
    if ticker and market is None:
        logger.warning("No market hours for %s, using UTC dates.", ticker)

    if market:
        tz_name, close_time = market
        local = published.dt.tz_convert(tz_name)
        day = local.dt.normalize().dt.tz_localize(None)   # local date, without the time zone
        after_close = local.dt.strftime("%H:%M") >= close_time
        day = day + pd.to_timedelta(after_close.astype(int), unit="D")
    else:
        day = published.dt.tz_localize(None).dt.normalize()

    df["date"] = day.dt.strftime("%Y-%m-%d")
    return (
        df.groupby("date")
        .agg(avg_score=("sentiment_score", "mean"),
             n_articles=("sentiment_score", "size"))
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )


def insert_news(items: list[dict]) -> int:
    """Insert articles. Duplicates (same URL) are skipped.

    Returns the number of articles actually inserted.
    """
    if not items:
        return 0
    now = utc_now_iso()
    inserted = 0
    with get_connection() as conn:
        for it in items:
            cur = conn.execute(
                """INSERT OR IGNORE INTO news
                   (query_tag, title, source, url, published_utc, ingested_at_utc)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    it["query_tag"], it["title"], it.get("source"),
                    it["url"], it.get("published_utc"), now,
                ),
            )
            inserted += cur.rowcount
    return inserted
