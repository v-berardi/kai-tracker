"""Tests for the database functions (src/storage/db.py) and the config.

Each test uses a new empty SQLite file in a temporary folder, so my
real database in data/ is not touched.
"""

import pytest

from src import config
from src.storage import db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Make the database functions use a temporary file."""
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()


def _add_scored_news(items: list[tuple[str, str, float]]) -> None:
    """items = [(query_tag, published_utc, score), ...]"""
    db.insert_news([
        {"query_tag": tag, "title": f"title {i}", "url": f"https://x/{i}",
         "published_utc": published}
        for i, (tag, published, _score) in enumerate(items)
    ])
    unscored = db.fetch_unscored_news()
    db.update_sentiment([
        (score, "positive", news_id)
        for (news_id, _title), (_tag, _pub, score) in zip(unscored, items)
    ])


def test_insert_news_skips_duplicate_urls(temp_db):
    item = {"query_tag": "a", "title": "t", "url": "https://x/1"}
    assert db.insert_news([item]) == 1
    assert db.insert_news([item]) == 0  # same URL: ignored


def test_daily_sentiment_filters_by_topics(temp_db):
    _add_scored_news([
        ("nvidia_supply", "2026-01-05T10:00:00+00:00", 0.5),
        ("samsung_hbm",   "2026-01-05T11:00:00+00:00", -0.5),
    ])

    only_nvidia = db.daily_sentiment(["nvidia_supply"])
    assert only_nvidia["avg_score"].tolist() == [pytest.approx(0.5)]

    everything = db.daily_sentiment()
    assert everything["avg_score"].tolist() == [pytest.approx(0.0)]
    assert everything["n_articles"].tolist() == [2]


def test_every_ticker_has_valid_topics():
    for ticker in config.TICKERS:
        topics = config.TICKER_TOPICS.get(ticker)
        assert topics, f"{ticker} has no news topics"
        for tag in topics:
            assert tag in config.NEWS_QUERIES, f"unknown topic {tag}"
