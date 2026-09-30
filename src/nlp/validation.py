"""Check FinBERT against my own labels on a random sample of headlines.

FinBERT was trained on financial news (Financial PhraseBank), not on
chip industry headlines from Google News. So before trusting the event
study, I check how often FinBERT agrees with a human (me) on my data.

How it works:
  1. make_sample(): take N random scored headlines and write them to a
     CSV with an empty "my_label" column. The CSV does NOT contain
     FinBERT's answer, so I can't be influenced by it when I label.
  2. I fill "my_label" by hand: positive / negative / neutral.
  3. evaluate(): compare my labels with FinBERT's labels (accuracy,
     confusion matrix, precision / recall per class, Cohen's kappa).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.storage.db import get_connection

logger = logging.getLogger(__name__)

LABELS = ["positive", "negative", "neutral"]


def make_sample(out_path: Path, n: int = 100, seed: int = 42) -> int:
    """Write n random scored headlines to out_path. Returns the number written.

    The seed makes the sample the same every time (reproducible).
    """
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT id, query_tag, title FROM news "
            "WHERE sentiment_label IS NOT NULL",
            conn,
        )
    if df.empty:
        raise ValueError("No scored articles. Run scripts/run_sentiment.py first.")

    sample = df.sample(n=min(n, len(df)), random_state=seed).sort_values("id")
    # the exact text that FinBERT scored, so I label the same thing
    sample["headline"] = sample["title"]
    sample["my_label"] = ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so that Excel shows the accents correctly
    sample[["id", "query_tag", "headline", "my_label"]].to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )
    return len(sample)


def read_labels(path: Path) -> pd.DataFrame:
    """Read the labeled CSV and check the labels. Returns id, my_label."""
    df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
    df["my_label"] = df["my_label"].fillna("").astype(str).str.strip().str.lower()
    df = df[df["my_label"] != ""]                      # rows not labeled yet
    bad = df[~df["my_label"].isin(LABELS)]
    if not bad.empty:
        raise ValueError(
            f"Unknown labels {sorted(bad['my_label'].unique())} for ids "
            f"{bad['id'].tolist()[:10]}. Use: {', '.join(LABELS)}."
        )
    return df[["id", "my_label"]]


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Accuracy, confusion matrix, precision / recall / F1 per class,
    Cohen's kappa, and the accuracy of always answering the most frequent
    class (to know if the model does better than a trivial guess).
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    n = len(y_true)

    # rows = my label (truth), columns = FinBERT
    confusion = pd.crosstab(y_true, y_pred).reindex(
        index=LABELS, columns=LABELS, fill_value=0
    )
    accuracy = float((y_true == y_pred).mean())

    per_class = {}
    for label in LABELS:
        tp = confusion.loc[label, label]
        predicted = confusion[label].sum()
        actual = confusion.loc[label].sum()
        precision = tp / predicted if predicted else np.nan
        recall = tp / actual if actual else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if predicted and actual and (precision + recall) else np.nan)
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": f1, "support": int(actual)}

    # Cohen's kappa: agreement corrected for the agreement expected by chance
    p_chance = float(sum(
        (confusion.loc[label].sum() / n) * (confusion[label].sum() / n) for label in LABELS
    ))
    kappa = (accuracy - p_chance) / (1 - p_chance) if p_chance < 1 else np.nan

    majority = y_true.value_counts().idxmax()
    return {
        "n": n,
        "accuracy": accuracy,
        "kappa": kappa,
        "majority_label": majority,
        "majority_accuracy": float((y_true == majority).mean()),
        "confusion": confusion,
        "per_class": per_class,
    }


def evaluate(path: Path) -> dict:
    """Compare my labels in path with the FinBERT labels in the database."""
    labels = read_labels(path)
    if labels.empty:
        raise ValueError(f"No labels found in {path}. Fill the my_label column first.")
    with get_connection() as conn:
        finbert = pd.read_sql_query(
            "SELECT id, sentiment_label AS finbert_label FROM news "
            "WHERE sentiment_label IS NOT NULL", conn,
        )
    df = labels.merge(finbert, on="id", how="inner")
    missing = len(labels) - len(df)
    if missing:
        logger.warning("%d labeled ids are not scored in the database, skipped.", missing)
    return metrics(df["my_label"], df["finbert_label"])
