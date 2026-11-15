"""Load the financial news sentiment dataset.

Purpose:
    Read ``all-data.csv`` into a pandas DataFrame and warn the user when
    the file is placeholder data.

Main functions:
    - ``is_placeholder_data()``
    - ``load_dataset()``
    - ``get_review()``

Notebook cells covered:
    "Load the dataset" (``pd.read_csv``, ``df.shape``, ``df.head(10)``) and
    "Let us view one of the reviews" (``df["review"][5]``).

Called by:
    ``main.py`` and ``src.pipeline``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import (
    CSV_COLUMN_ORDER,
    DATASET_ENCODING,
    DATASET_HAS_HEADER,
    DATASET_PATH,
    LABEL_COLUMN,
    PLACEHOLDER_MARKER_PATH,
    TEXT_COLUMN,
    validate_settings,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def is_placeholder_data(marker_path: Path = PLACEHOLDER_MARKER_PATH) -> bool:
    """Return True if the placeholder marker file exists next to the dataset.

    Args:
        marker_path: Path of the ``.PLACEHOLDER.md`` marker file.

    Returns:
        True when the dataset in ``data/raw`` is placeholder data.

    Why: see knowledge.md, section "1. Loading the Dataset".
    """
    return marker_path.exists()


def load_dataset(
    path: Path = DATASET_PATH,
    sample_size: int | None = None,
    marker_path: Path = PLACEHOLDER_MARKER_PATH,
) -> pd.DataFrame:
    """Read the dataset CSV file into a DataFrame.

    Args:
        path: Path to the CSV file with ``review`` and ``sentiment`` columns.
        sample_size: If set, keep only the first ``sample_size`` rows.
            ``None`` keeps all rows (the notebook behavior).
        marker_path: Path of the placeholder marker file.

    Returns:
        A DataFrame with the columns ``review`` and ``sentiment``.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If a required column is missing.

    Why: see knowledge.md, section "1. Loading the Dataset".
    """
    validate_settings(path)
    if DATASET_HAS_HEADER:
        df = pd.read_csv(path, encoding=DATASET_ENCODING)
    else:
        # No header row: name the columns ourselves (label first, then text).
        df = pd.read_csv(
            path, header=None, names=list(CSV_COLUMN_ORDER), encoding=DATASET_ENCODING
        )
    # Keep the usual "review, sentiment" column order for previews/output.
    df = df[[TEXT_COLUMN, LABEL_COLUMN]]

    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} must have the columns '{TEXT_COLUMN}' and "
            f"'{LABEL_COLUMN}'. Missing: {sorted(missing)}"
        )

    if is_placeholder_data(marker_path):
        logger.warning(
            "PLACEHOLDER DATA: '%s' contains %d made-up reviews. "
            "Replace it with the real dataset file, then delete '%s'. "
            "See data/README.md.",
            path.name,
            len(df),
            marker_path.name,
        )

    if sample_size is not None:
        df = df.head(sample_size).reset_index(drop=True)
        logger.info("Using the first %d rows (sample_size is set).", len(df))

    logger.info("Loaded dataset with shape %s.", df.shape)  # notebook: print(df.shape)
    return df


def get_review(df: pd.DataFrame, index: int) -> str:
    """Return the full text of one review.

    Args:
        df: The reviews DataFrame.
        index: Row index of the review.

    Returns:
        The review text. Returns an empty string if the index does not exist.

    Why: see knowledge.md, section "1. Loading the Dataset".
    """
    if index not in df.index:
        logger.warning("Row %d does not exist (dataset has %d rows).", index, len(df))
        return ""
    return str(df[TEXT_COLUMN][index])


if __name__ == "__main__":
    data = load_dataset()
    print(data.head(10))
