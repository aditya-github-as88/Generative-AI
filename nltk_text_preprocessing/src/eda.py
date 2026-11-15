"""Exploratory Data Analysis (EDA) of the reviews.

Purpose:
    Summarize the dataset and check the class balance of the ``sentiment``
    column, with a bar chart.

Main functions:
    - ``describe_dataset()``
    - ``sentiment_counts()``
    - ``plot_sentiment_distribution()``

Notebook cells covered:
    "Exploratory Data Analysis" (``df.describe()``,
    ``df['sentiment'].value_counts()``, ``sns.barplot(...)``).

Called by:
    ``main.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from config.settings import LABEL_COLUMN, SENTIMENT_PLOT_PATH, SHOW_PLOTS
from utils.logger import get_logger

logger = get_logger(__name__)


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return the summary statistics of the DataFrame.

    Args:
        df: The reviews DataFrame.

    Returns:
        The output of ``df.describe()``: count, unique, top, freq per column.

    Why: see knowledge.md, section "2. Exploratory Data Analysis".
    """
    return df.describe()


def sentiment_counts(df: pd.DataFrame) -> pd.Series:
    """Count the reviews per sentiment class.

    Args:
        df: The reviews DataFrame.

    Returns:
        A Series: index = sentiment label, value = number of reviews.

    Why: see knowledge.md, section "2. Exploratory Data Analysis".
    """
    return df[LABEL_COLUMN].value_counts()


def plot_sentiment_distribution(
    df: pd.DataFrame,
    save_path: Path = SENTIMENT_PLOT_PATH,
    show: bool = SHOW_PLOTS,
) -> Path:
    """Draw a bar chart of the review count per sentiment class and save it.

    Args:
        df: The reviews DataFrame.
        save_path: Where to save the PNG file.
        show: If True, also open the chart in a window.

    Returns:
        The path of the saved PNG file.

    Why: see knowledge.md, section "2. Exploratory Data Analysis".
    """
    if not show:
        matplotlib.use("Agg")  # no window needed; works on servers and in tests
    import matplotlib.pyplot as plt
    import seaborn as sns

    counts = sentiment_counts(df)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    # Same call as the notebook, drawn on an explicit Axes.
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=ax)
    ax.set_title("Sentiment distribution")
    ax.set_xlabel(LABEL_COLUMN)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    logger.info("Saved sentiment bar chart to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    from src.data_loader import load_dataset

    data = load_dataset()
    print(describe_dataset(data))
    print(sentiment_counts(data))
    plot_sentiment_distribution(data)
