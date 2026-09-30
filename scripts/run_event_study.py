"""Runs the event study from the command line.

Usage:
    python scripts/run_event_study.py
    python scripts/run_event_study.py --ticker NVDA --threshold 0.3
    python scripts/run_event_study.py --ticker TSM --query-tag tsmc_cowos
    python scripts/run_event_study.py --ticker NVDA --all-topics

By default it only uses the news topics of the ticker
(see TICKER_TOPICS in src/config.py).

Prints results to the terminal. Charts live in the dashboard.
"""

import argparse
import logging
import sys
from pathlib import Path

# Make the src package importable when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import NEWS_QUERIES, TICKERS  # noqa: E402
from src.analysis.event_study import run      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA", choices=list(TICKERS.keys()))
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Minimum FinBERT score to declare an event")
    parser.add_argument("--query-tag", action="append",
                        choices=list(NEWS_QUERIES.keys()),
                        help="News topic to use (you can give it more than once)")
    parser.add_argument("--all-topics", action="store_true",
                        help="Use all the news topics, not only the ones of the ticker")
    args = parser.parse_args()

    if args.all_topics:
        query_tags = list(NEWS_QUERIES.keys())
    else:
        query_tags = args.query_tag  # None -> topics of the ticker

    print(f"\n{'='*60}")
    print(f"Event study - {TICKERS[args.ticker]} ({args.ticker})")
    print(f"Sentiment threshold: |score| >= {args.threshold}")
    print(f"News topics: {', '.join(query_tags) if query_tags else 'linked to the ticker'}")
    print(f"{'='*60}\n")

    results = run(
        ticker=args.ticker,
        query_tags=query_tags,
        threshold=args.threshold,
    )

    if not results:
        print("No results. Check that the database has sentiment scores.")
        return

    for direction, res in results.items():
        if res is None:
            print(f"[{direction:8s}] Not enough events for a test.\n")
            continue

        sig = "SIGNIFICANT" if res.p_value < 0.05 else "not significant"
        print(f"[{direction:8s}]  n={res.n_events:3d}  "
              f"CAAR={res.caar:+.4f}  "
              f"t={res.t_stat:+.2f}  "
              f"p={res.p_value:.3f}  -> {sig}")

    print()
    if "all" in results and results["all"] is not None:
        df = results["all"].events_df[
            ["event_date", "direction", "avg_score", "normal_return", "car"]
        ].copy()
        df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d")
        df["avg_score"] = df["avg_score"].map("{:+.3f}".format)
        df["normal_return"] = df["normal_return"].map("{:+.4f}".format)
        df["car"] = df["car"].map("{:+.4f}".format)
        print(df.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
