"""Tests for the event study (src/analysis/event_study.py).

These tests use small, made-up sentiment and price series instead of
the real database. This way we can check the math (event detection,
CAR, CAAR, t-test) without needing real data or a network connection.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import event_study as es


# --- detect_events ---

def test_detect_events_filters_by_threshold_and_min_articles():
    sentiment = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "avg_score": [0.30, 0.10, -0.40],
        "n_articles": [5, 5, 1],  # day 3 has too few articles to count
    })

    events = es.detect_events(sentiment, threshold=0.25, min_articles=2)

    # only day 1 passes both the score threshold and the article count filter
    assert len(events) == 1
    assert events.iloc[0]["direction"] == "positive"


def test_detect_events_applies_cooldown():
    # two spikes only 3 days apart: with a 5-day cooldown, only the first counts
    sentiment = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=6, freq="D").astype(str),
        "avg_score": [0.30, 0.0, 0.0, 0.35, 0.0, 0.0],
        "n_articles": [5, 5, 5, 5, 5, 5],
    })

    events = es.detect_events(sentiment, threshold=0.25, min_articles=2, cooldown_days=5)

    assert len(events) == 1
    assert events.iloc[0]["date"] == pd.Timestamp("2026-01-01")


def test_detect_events_empty_input_returns_empty_frame():
    empty = pd.DataFrame(columns=["date", "avg_score", "n_articles"])
    events = es.detect_events(empty)
    assert events.empty


# --- compute_cars ---

def _fake_prices(returns: list[float]) -> pd.DataFrame:
    """Build a made-up price/return table, shaped like _load_prices()'s output."""
    dates = pd.date_range("2025-01-01", periods=len(returns), freq="D")
    return pd.DataFrame({"date": dates, "close": np.nan, "ret": returns})


def test_compute_cars_matches_manual_calculation(monkeypatch):
    # 200 flat days (return = 0) except a few known values around the event
    n = 200
    returns = [0.0] * n
    event_idx = 150
    bump = {-2: 0.01, 0: 0.02, 3: -0.005}  # offsets relative to the event day
    for offset, value in bump.items():
        returns[event_idx + offset] = value

    fake = _fake_prices(returns)
    monkeypatch.setattr(es, "_load_prices", lambda ticker: fake)

    events = pd.DataFrame({
        "date": [fake["date"].iloc[event_idx]],
        "avg_score": [0.5],
        "n_articles": [3],
        "direction": ["positive"],
    })

    cars = es.compute_cars("FAKE", events, estimation_window=(-120, -21), event_window=(-5, 5))

    assert len(cars) == 1
    # the estimation window is all zeros, so normal_return = 0
    # and car should just be the sum of the event window returns
    expected_car = sum(bump.values())
    assert cars.iloc[0]["car"] == pytest.approx(expected_car)
    assert cars.iloc[0]["normal_return"] == pytest.approx(0.0)


def test_compute_cars_skips_events_without_enough_history():
    # event too close to the start: no room for the estimation window
    fake = _fake_prices([0.0] * 50)
    events = pd.DataFrame({
        "date": [fake["date"].iloc[5]],
        "avg_score": [0.5],
        "n_articles": [3],
        "direction": ["positive"],
    })

    import src.analysis.event_study as es_module
    original = es_module._load_prices
    es_module._load_prices = lambda ticker: fake
    try:
        cars = es.compute_cars("FAKE", events)
    finally:
        es_module._load_prices = original

    assert cars.empty


def _events_at(fake: pd.DataFrame, indexes: list[int]) -> pd.DataFrame:
    """Build positive events on the given trading day indexes."""
    return pd.DataFrame({
        "date": [fake["date"].iloc[i] for i in indexes],
        "avg_score": [0.5] * len(indexes),
        "n_articles": [3] * len(indexes),
        "direction": ["positive"] * len(indexes),
    })


def test_compute_cars_skips_overlapping_events(monkeypatch):
    fake = _fake_prices([0.0] * 300)
    monkeypatch.setattr(es, "_load_prices", lambda ticker: fake)

    # 150 and 155 are only 5 trading days apart, so their windows overlap.
    # 170 is 20 days after 150, no overlap, so it is kept.
    events = _events_at(fake, [150, 155, 170])
    cars = es.compute_cars("FAKE", events)

    kept = cars["event_date"].tolist()
    assert kept == [fake["date"].iloc[150], fake["date"].iloc[170]]


def test_compute_cars_leaves_other_event_windows_out_of_the_baseline(monkeypatch):
    returns = [0.0] * 300
    returns[130] = 0.5          # big move on the day of the first event
    fake = _fake_prices(returns)
    monkeypatch.setattr(es, "_load_prices", lambda ticker: fake)

    # the estimation window of the 2nd event (day 180) is days 60-159,
    # and the window of the 1st event (days 125-135) is inside it.
    events = _events_at(fake, [130, 180])
    cars = es.compute_cars("FAKE", events)

    assert len(cars) == 2
    assert cars.iloc[0]["car"] == pytest.approx(0.5)
    # before the fix, the 0.5 move was going into the 2nd "normal" return
    assert cars.iloc[1]["normal_return"] == pytest.approx(0.0)


# --- aggregate ---

def test_aggregate_computes_caar_and_flags_significance():
    # three events with the exact same AR pattern for every day
    ar_series = [0.01] * 11  # 11 days in the [-5, +5] window
    cars = pd.DataFrame({
        "event_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
        "direction": ["positive", "positive", "positive"],
        "avg_score": [0.5, 0.6, 0.55],
        "normal_return": [0.0, 0.0, 0.0],
        "car": [0.11, 0.11, 0.11],
        "ar_series": [ar_series, ar_series, ar_series],
    })

    result = es.aggregate(cars, event_window=(-5, 5), direction="all", ticker="FAKE")

    assert result is not None
    assert result.n_events == 3
    assert result.caar == pytest.approx(0.11)
    # all three CAR values are identical, so there is no variance
    # aggregate() handles this by returning t_stat = 0.0 instead of dividing by zero
    assert result.t_stat == 0.0


def test_aggregate_returns_none_with_too_few_events():
    cars = pd.DataFrame({
        "event_date": pd.to_datetime(["2026-01-01"]),
        "direction": ["positive"],
        "avg_score": [0.5],
        "normal_return": [0.0],
        "car": [0.05],
        "ar_series": [[0.01] * 11],
    })

    assert es.aggregate(cars, direction="all", ticker="FAKE") is None
