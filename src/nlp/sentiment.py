"""Scores news headlines with FinBERT.

FinBERT is a BERT model fine-tuned on financial text, so it should
understand finance-specific phrasing better than a generic sentiment
model (for example, "cuts guidance" is negative, "beats estimates" is
positive).

score = P(positive) - P(negative), a number between -1 and +1. A
continuous score is easier to average per day than a discrete label.

The model is loaded lazily: just importing this file does not download
anything. The ~440MB model is only loaded the first time scoring runs.
"""

import logging

from src.config import FINBERT_MODEL, NLP_BATCH_SIZE, NLP_MAX_LENGTH
from src.storage.db import fetch_unscored_news, update_sentiment

logger = logging.getLogger(__name__)

# Order of the classes in ProsusAI/finbert's output
_LABELS = ("positive", "negative", "neutral")

_tokenizer = None
_model = None


def _load_model():
    """Load the tokenizer and the model once, and keep them in memory."""
    global _tokenizer, _model
    if _model is not None:
        return
    # imported here, not at the top of the file: transformers pulls in
    # torch, and both are heavy, so we only pay that cost if scoring runs
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading %s (first call may download it)...", FINBERT_MODEL)
    _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    _model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    _model.eval()  # inference mode: turns off dropout, etc.
    logger.info("Model loaded.")


def score_texts(texts: list[str]) -> list[tuple[float, str]]:
    """Score a list of texts. Returns [(score, label), ...].

    score = P(positive) - P(negative), between -1 and +1.
    label = the most likely class ('positive' / 'negative' / 'neutral').
    """
    if not texts:
        return []
    _load_model()
    import torch

    results: list[tuple[float, str]] = []
    for i in range(0, len(texts), NLP_BATCH_SIZE):
        batch = texts[i : i + NLP_BATCH_SIZE]
        inputs = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=NLP_MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():  # no training here, so we don't need gradients
            logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)  # (batch, 3)

        for row in probs:
            p_pos, p_neg, _p_neu = row.tolist()
            score = p_pos - p_neg
            label = _LABELS[int(row.argmax())]
            results.append((round(score, 4), label))

        logger.info("Batch %d-%d scored (%d texts)",
                    i, i + len(batch) - 1, len(batch))
    return results


def run(limit: int | None = None) -> int:
    """Read unscored articles, score them, and save the results.

    Only articles with sentiment_score = NULL are picked up, so running
    this again only costs the new articles. Returns how many were scored.
    """
    pending = fetch_unscored_news(limit)
    if not pending:
        logger.info("Nothing to score, everything is up to date.")
        return 0

    ids = [row[0] for row in pending]
    titles = [row[1] for row in pending]
    logger.info("To score: %d articles", len(titles))

    scored = score_texts(titles)
    rows = [(score, label, news_id)
            for (score, label), news_id in zip(scored, ids)]
    n = update_sentiment(rows)
    logger.info("Scores written: %d", n)
    return n
