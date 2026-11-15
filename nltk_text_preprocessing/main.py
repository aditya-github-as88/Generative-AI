"""Single entry point: run the full text-preprocessing tutorial.

Usage:
    python main.py                       # run every stage
    python main.py --step demo           # run one stage only
    python main.py --sample 2000         # use the first 2,000 reviews
    python main.py --show-plots          # also open the chart window

Stages (in order):
    1. setup       - download missing NLTK resources
    2. load        - read data/raw/all-data.csv
    3. eda         - summary, class counts, bar chart
    4. demo        - each NLTK technique on the notebook's example sentences
    5. preprocess  - clean every review and save to data/processed/

This file only orchestrates. The logic lives in ``src/`` and ``utils/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from config import settings
from src.data_loader import get_review, load_dataset
from src.eda import describe_dataset, plot_sentiment_distribution, sentiment_counts
from src.pipeline import preprocess_reviews, run_demos, save_processed
from utils.logger import get_logger
from utils.setup_resources import ensure_nltk_resources

logger = get_logger("main")

STEPS = ("all", "load", "eda", "demo", "preprocess")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line arguments. Every argument has a default."""
    parser = argparse.ArgumentParser(
        description="Text preprocessing of financial news sentences with NLTK."
    )
    parser.add_argument(
        "--step", choices=STEPS, default="all",
        help="Run one stage only (default: all).",
    )
    parser.add_argument(
        "--data", type=Path, default=settings.DATASET_PATH,
        help=f"Path to the dataset CSV (default: {settings.DATASET_PATH}).",
    )
    parser.add_argument(
        "--sample", type=int, default=settings.SAMPLE_SIZE,
        help="Use only the first N reviews (default: all rows, or SAMPLE_SIZE in .env).",
    )
    parser.add_argument(
        "--show-plots", action="store_true", default=settings.SHOW_PLOTS,
        help="Open the chart in a window as well as saving it.",
    )
    return parser.parse_args(argv)


def _print_header(number: int, title: str) -> None:
    print(f"\n{'=' * 70}\n[{number}] {title}\n{'=' * 70}")


def stage_load(data_path: Path, sample: int | None) -> pd.DataFrame:
    _print_header(2, "Load the dataset")
    df = load_dataset(data_path, sample_size=sample)
    print(df.shape)
    print(df.head(settings.PREVIEW_ROWS))
    print(f"\nReview #{settings.SAMPLE_REVIEW_INDEX}:")
    print(get_review(df, settings.SAMPLE_REVIEW_INDEX))
    return df


def stage_eda(df: pd.DataFrame, show_plots: bool) -> Path:
    _print_header(3, "Exploratory Data Analysis")
    print(describe_dataset(df))
    counts = sentiment_counts(df)
    print()
    print(counts)
    print(counts.values)
    return plot_sentiment_distribution(df, show=show_plots)


def stage_demo() -> dict[str, object]:
    _print_header(4, "NLTK technique demos (notebook examples)")
    results = run_demos()
    for name, value in results.items():
        print(f"\n--- {name} ---")
        print(value)
    return results


def stage_preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    _print_header(5, "Preprocess every review")
    if len(df) > 5000:
        logger.info(
            "Processing %d reviews. The stemming step can take 2-3 minutes.", len(df)
        )
    cleaned = preprocess_reviews(df)
    print(cleaned.head())
    path = save_processed(cleaned)
    return cleaned, path


def main(argv: list[str] | None = None) -> int:
    """Run the selected stages and print a numbered summary.

    Returns:
        Process exit code: 0 on success, 1 on a known error.
    """
    args = parse_args(argv)
    summary: list[str] = []

    try:
        _print_header(1, "Set up NLTK resources")
        missing = ensure_nltk_resources()
        if missing:
            print("Missing NLTK resources: " + ", ".join(missing))
            print("Action: connect to the internet and run: python -m utils.setup_resources")
            return 1
        summary.append("setup: all NLTK resources are ready")

        df: pd.DataFrame | None = None
        if args.step in ("all", "load", "eda", "preprocess"):
            df = stage_load(args.data, args.sample)
            summary.append(f"load: read {len(df)} reviews from {args.data.name}")

        if args.step in ("all", "eda") and df is not None:
            plot_path = stage_eda(df, args.show_plots)
            summary.append(f"eda: class counts printed, chart saved to {plot_path}")

        if args.step in ("all", "demo"):
            stage_demo()
            summary.append("demo: ran 10 technique examples from the notebook")

        if args.step in ("all", "preprocess") and df is not None:
            cleaned, out_path = stage_preprocess(df)
            summary.append(f"preprocess: cleaned {len(cleaned)} reviews, saved to {out_path}")

    except (FileNotFoundError, ValueError) as error:
        print(f"\nERROR: {error}")
        return 1

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for number, line in enumerate(summary, start=1):
        print(f"{number}. {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
