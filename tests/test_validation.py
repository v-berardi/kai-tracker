"""Tests for the FinBERT validation (src/nlp/validation.py)."""

import pandas as pd
import pytest

from src.nlp import validation
from src.storage import db


def test_metrics_on_a_small_example_computed_by_hand():
    y_true = ["positive", "positive", "negative", "neutral", "neutral", "neutral"]
    y_pred = ["positive", "neutral", "negative", "neutral", "neutral", "positive"]

    m = validation.metrics(pd.Series(y_true), pd.Series(y_pred))

    assert m["accuracy"] == pytest.approx(4 / 6)
    # chance agreement = (2*2 + 1*1 + 3*3) / 36 = 14/36
    # kappa = (24/36 - 14/36) / (1 - 14/36) = 10/22
    assert m["kappa"] == pytest.approx(5 / 11)
    assert m["majority_label"] == "neutral"
    assert m["majority_accuracy"] == pytest.approx(0.5)
    assert m["per_class"]["positive"]["precision"] == pytest.approx(0.5)
    assert m["per_class"]["neutral"]["recall"] == pytest.approx(2 / 3)
    assert m["confusion"].loc["positive", "neutral"] == 1   # true pos, FinBERT neutral


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    db.insert_news([
        {"query_tag": "a", "title": f"News {i} - Reuters", "source": "Reuters",
         "url": f"https://x/{i}"}
        for i in range(20)
    ])
    ids = [row[0] for row in db.fetch_unscored_news()]
    db.update_sentiment([(0.9, "positive", i) for i in ids])


def test_sample_is_blind_and_reproducible(temp_db, tmp_path):
    out1, out2 = tmp_path / "s1.csv", tmp_path / "s2.csv"
    assert validation.make_sample(out1, n=5, seed=1) == 5
    validation.make_sample(out2, n=5, seed=1)

    df = pd.read_csv(out1, encoding="utf-8-sig")
    # no FinBERT output in the file, so the labels are not influenced
    assert list(df.columns) == ["id", "query_tag", "headline", "my_label"]
    assert out1.read_text(encoding="utf-8-sig") == out2.read_text(encoding="utf-8-sig")


def test_evaluate_and_bad_labels(temp_db, tmp_path):
    path = tmp_path / "labels.csv"
    validation.make_sample(path, n=4, seed=1)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["my_label"] = ["positive", "Positive ", "negative", ""]   # last one not labeled yet
    df.to_csv(path, index=False)

    res = validation.evaluate(path)
    assert res["n"] == 3                               # the empty label is skipped
    assert res["accuracy"] == pytest.approx(2 / 3)     # FinBERT says positive for all

    df["my_label"] = ["positive", "good", "negative", ""]
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="good"):
        validation.evaluate(path)
