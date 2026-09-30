"""Check FinBERT against reference labels (see src/nlp/validation.py).

Usage:
    # 1. make a random sample of headlines to label (I used --n 50)
    python scripts/validate_finbert.py sample --n 50

    # 2. fill the llm_label column of validation/headlines_to_label.csv
    #    with positive / negative / neutral, without looking at the
    #    FinBERT scores (in this project an LLM proposed the labels)

    # 3. compare the labels with FinBERT
    python scripts/validate_finbert.py evaluate
"""

import argparse
import logging
import sys
from pathlib import Path

# Make the src package importable when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT                              # noqa: E402
from src.nlp.validation import LABELS, evaluate, make_sample     # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DEFAULT_FILE = PROJECT_ROOT / "validation" / "headlines_to_label.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check FinBERT against reference labels")
    parser.add_argument("step", choices=["sample", "evaluate"])
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--n", type=int, default=100, help="Sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the sample file if it already exists")
    args = parser.parse_args()

    if args.step == "sample":
        if args.file.exists() and not args.force:
            print(f"{args.file} already exists (maybe with your labels). "
                  f"Use --force to overwrite it.")
            return 1
        n = make_sample(args.file, n=args.n, seed=args.seed)
        print(f"{n} headlines written to {args.file}")
        print(f"Fill the llm_label column with: {', '.join(LABELS)}")
        return 0

    res = evaluate(args.file)
    print(f"\nFinBERT vs the labels, on {res['n']} headlines\n")
    print(f"Accuracy:       {res['accuracy']:.1%}")
    print(f"Always '{res['majority_label']}': {res['majority_accuracy']:.1%}"
          f"  (trivial baseline: FinBERT must do better)")
    print(f"Cohen's kappa:  {res['kappa']:.2f}"
          f"  (0 = chance level, 1 = perfect agreement)\n")
    print("Confusion matrix (rows = llm_label, columns = FinBERT):")
    print(res["confusion"].to_string())
    print("\nPer class:")
    for label, m in res["per_class"].items():
        print(f"  {label:9s} precision={m['precision']:.2f}  recall={m['recall']:.2f}"
              f"  f1={m['f1']:.2f}  (n={m['support']})")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
