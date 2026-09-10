"""Entry point: runs full ingestion (prices + news).

Usage:
    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --skip-prices
    python scripts/run_ingestion.py --skip-news
"""

import argparse
import logging
import sys
from pathlib import Path

# Make the src package importable when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import market, news  # noqa: E402
from src.storage.db import init_db      # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_ingestion")


def main() -> int:
    parser = argparse.ArgumentParser(description="K-AI Tracker - ingestion")
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    args = parser.parse_args()

    init_db()
    exit_code = 0

    if not args.skip_prices:
        try:
            n = market.run()
            logger.info("Market OK: %d price rows", n)
        except Exception as err:
            logger.error("Market ingestion failed: %s", err)
            exit_code = 1

    if not args.skip_news:
        try:
            n = news.run()
            logger.info("News OK: %d new articles", n)
        except Exception as err:
            logger.error("News ingestion failed: %s", err)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
