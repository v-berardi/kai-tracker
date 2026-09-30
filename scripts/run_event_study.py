"""Runs the event study from the command line.

Usage:
    python scripts/run_event_study.py
    python scripts/run_event_study.py --ticker NVDA --threshold 0.3
    python scripts/run_event_study.py --ticker TSM --query-tag tsmc_cowos
    python scripts/run_event_study.py --ticker NVDA --all-topics
    python scripts/run_event_study.py --ticker NVDA --model constant
    python scripts/run_event_study.py --all-tickers

By default it only uses the news topics of the ticker
(see TICKER_TOPICS in src/config.py) and the market model
(see BENCHMARKS in src/config.py).

--all-tickers runs the study on the 4 tickers and prints one summary
table. I use it to report ALL the results together, and not only the
ticker that gives the best p-value (that would be p-hacking).

Prints results to the terminal. Charts live in the dashboard.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Make the src package importable when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BENCHMARKS, NEWS_QUERIES, TICKERS  # noqa: E402
from src.analysis.event_study import run                  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DIRECTIONS = ("all", "positive", "negative")


def line(label: str, caar: float, t: float, p: float) -> str:
    sig = "SIGNIFICANT" if p < 0.05 else "not significant"
    return (f"    {label:15s} CAAR={caar:+.4f}  t={t:+.2f}  "
            f"p={p:.3f}  -> {sig}")


def run_one(args, query_tags: list[str] | None) -> None:
    """Detailed results for one ticker."""
    benchmark = BENCHMARKS.get(args.ticker) if args.model == "market" else None
    print(f"\n{'='*60}")
    print(f"Event study - {TICKERS[args.ticker]} ({args.ticker})")
    print(f"Model: {args.model}" + (f" (index: {benchmark})" if benchmark else ""))
    print(f"Sentiment threshold: |score| >= {args.threshold}")
    print(f"News topics: {', '.join(query_tags) if query_tags else 'linked to the ticker'}")
    print(f"{'='*60}\n")

    results = run(
        ticker=args.ticker,
        query_tags=query_tags,
        threshold=args.threshold,
        model=args.model,
    )

    if not results:
        print("No results. Check that the database has sentiment scores.")
        return

    for direction, res in results.items():
        if res is None:
            print(f"[{direction:8s}] Less than 5 events, no test.\n")
            continue
        print(f"[{direction:8s}]  n={res.n_events}")
        # "after" is the real question: do returns move AFTER the news?
        print(line("after  [0,+5]", res.caar_post, res.t_post, res.p_post))
        print(line("before [-5,-1]", res.caar_pre, res.t_pre, res.p_pre))
        print(line("full   [-5,+5]", res.caar, res.t_stat, res.p_value))
        print()

    if "all" in results and results["all"] is not None:
        cols = ["event_date", "direction", "avg_score", "normal_return",
                "car_pre", "car_post", "car"]
        if args.model == "market":
            cols.insert(3, "beta")
        df = results["all"].events_df[cols].copy()
        df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d")
        df["avg_score"] = df["avg_score"].map("{:+.3f}".format)
        if "beta" in df:
            df["beta"] = df["beta"].map("{:.2f}".format)
        for col in ["normal_return", "car_pre", "car_post", "car"]:
            df[col] = df[col].map("{:+.4f}".format)
        print(df.to_string(index=False))
    print()


def run_all(args, all_topics: bool) -> None:
    """One summary table for all the tickers."""
    rows = []
    for ticker in TICKERS:
        query_tags = list(NEWS_QUERIES.keys()) if all_topics else None
        try:
            results = run(ticker=ticker, query_tags=query_tags,
                          threshold=args.threshold, model=args.model)
        except ValueError as err:   # for example no prices for the index
            print(f"{ticker}: skipped ({err})")
            continue
        for direction in DIRECTIONS:
            res = results.get(direction)
            if res is None:
                rows.append({"ticker": ticker, "group": direction, "n": "<5"})
                continue
            rows.append({
                "ticker": ticker, "group": direction, "n": res.n_events,
                "CAAR after": res.caar_post, "p after": res.p_post,
                "CAAR before": res.caar_pre, "p before": res.p_pre,
                "CAAR full": res.caar, "p full": res.p_value,
            })

    print(f"\n{'='*78}")
    print(f"Event study - all tickers | model: {args.model} | "
          f"threshold: {args.threshold} | topics: {'all' if all_topics else 'per ticker'}")
    print(f"{'='*78}")
    if not rows:
        print("No results.")
        return

    table = pd.DataFrame(rows)
    shown = table.copy()
    for col in ["CAAR after", "CAAR before", "CAAR full"]:
        if col in shown:
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:+.4f}")
    for col in ["p after", "p before", "p full"]:
        if col in shown:
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    print(shown.to_string(index=False))

    # Multiple tests: with many tests, some p < 0.05 happen by chance.
    # Bonferroni correction: divide 0.05 by the number of tests.
    p_cols = [c for c in ["p after", "p before", "p full"] if c in table]
    n_tests = int(table[p_cols].notna().sum().sum()) if p_cols else 0
    if n_tests == 0:
        return
    limit = 0.05 / n_tests
    n_raw = int((table[p_cols] < 0.05).sum().sum())
    n_bonf = int((table[p_cols] < limit).sum().sum())
    print(f"\n{n_tests} tests in total. p < 0.05: {n_raw} "
          f"(about {0.05 * n_tests:.1f} expected by chance only).")
    print(f"With the Bonferroni correction (p < {limit:.4f}): {n_bonf}.")
    print("Groups with less than 5 events have no test (n = <5).")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA", choices=list(TICKERS.keys()))
    parser.add_argument("--all-tickers", action="store_true",
                        help="Run the 4 tickers and print one summary table")
    parser.add_argument("--model", default="market", choices=["market", "constant"],
                        help="How to compute the normal return (default: market)")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Minimum FinBERT score to declare an event")
    parser.add_argument("--query-tag", action="append",
                        choices=list(NEWS_QUERIES.keys()),
                        help="News topic to use (you can give it more than once)")
    parser.add_argument("--all-topics", action="store_true",
                        help="Use all the news topics, not only the ones of the ticker")
    args = parser.parse_args()

    if args.all_tickers:
        if args.query_tag:
            parser.error("--query-tag can't be used with --all-tickers")
        run_all(args, all_topics=args.all_topics)
        return

    if args.all_topics:
        query_tags = list(NEWS_QUERIES.keys())
    else:
        query_tags = args.query_tag  # None -> topics of the ticker
    run_one(args, query_tags)


if __name__ == "__main__":
    main()
