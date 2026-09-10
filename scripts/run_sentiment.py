"""Entry point: scores every article that hasn't been analyzed yet.

Usage:
    python scripts/run_sentiment.py
    python scripts/run_sentiment.py --limit 50   # quick test
"""

import argparse
import logging
import sys
from pathlib import Path

# Make the src package importable when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlp import sentiment  # noqa: E402
from src.storage.db import daily_sentiment, init_db  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_sentiment")


def main() -> int:
    parser = argparse.ArgumentParser(description="K-AI Tracker - sentiment scoring")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score N articles (quick test)")
    args = parser.parse_args()

    init_db()
    try:
        n = sentiment.run(limit=args.limit)
        logger.info("Done: %d articles scored", n)
    except Exception as err:
        logger.error("Scoring failed: %s", err)
        return 1

    df = daily_sentiment()
    if not df.empty:
        logger.info("Daily sentiment (last 5 days):\n%s",
                    df.tail().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
