"""Preprocessing pipeline and the notebook's small demos.

Purpose:
    1. Run each NLTK technique on the small example sentences from the
       notebook (``run_demos``).
    2. Apply the four cleaning steps to every review, in the notebook order
       (``preprocess_reviews``):
       denoise -> remove special characters -> stem -> remove stop words.
    3. Save the cleaned reviews to ``data/processed``.

Main functions:
    - ``run_demos()``
    - ``preprocess_reviews()``
    - ``save_processed()``

Called by:
    ``main.py``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from config import settings
from src.normalization import lemmatize_word, simple_stemmer
from src.stopword_removal import (
    get_stopword_list,
    get_updated_stopword_list,
    remove_stopwords,
)
from src.text_cleaning import (
    denoise_text,
    remove_between_square_brackets,
    remove_special_characters,
    strip_html,
)
from src.tokenization import pos_tag_tokens, tokenize
from utils.logger import get_logger

logger = get_logger(__name__)

# The order of the steps is the order of the notebook. Do not change it
# without reading knowledge.md, section "10. The Full Preprocessing Pipeline".
PREPROCESSING_STEPS: list[tuple[str, Callable[[str], str]]] = [
    ("denoise_text", denoise_text),
    ("remove_special_characters", remove_special_characters),
    ("simple_stemmer", simple_stemmer),
    ("remove_stopwords", remove_stopwords),
]


def run_demos() -> dict[str, object]:
    """Run every technique on the notebook's example inputs.

    Returns:
        A dict: demo name -> output. The values match the notebook outputs.

    Why: see knowledge.md, sections 3 to 9 (one demo per technique).
    """
    tokens = tokenize(settings.DEMO_TOKENIZE_TEXT)
    stopword_list = get_stopword_list()
    return {
        "strip_html": strip_html(settings.DEMO_HTML),
        "remove_between_square_brackets": remove_between_square_brackets(
            settings.DEMO_BRACKETS
        ),
        "remove_special_characters": remove_special_characters(
            settings.DEMO_SPECIAL_CHARS
        ),
        "lemmatize_word": lemmatize_word(
            settings.DEMO_LEMMA_WORD, pos=settings.DEMO_LEMMA_POS
        ),
        "simple_stemmer": simple_stemmer(settings.DEMO_STEM_TEXT),
        "tokenize": tokens,
        "pos_tag_tokens": pos_tag_tokens(tokens),
        "stopword_list": stopword_list,
        "updated_stopword_list": list(get_updated_stopword_list()),
        "remove_stopwords": remove_stopwords(settings.DEMO_STOPWORD_TEXT),
    }


def preprocess_reviews(
    df: pd.DataFrame,
    text_column: str = settings.TEXT_COLUMN,
) -> pd.DataFrame:
    """Apply all cleaning steps to the review column, in the notebook order.

    Args:
        df: The reviews DataFrame. The function does not change it.
        text_column: Name of the column with the review text.

    Returns:
        A new DataFrame with the cleaned text in ``text_column``.

    Why: see knowledge.md, section "10. The Full Preprocessing Pipeline".
    """
    result = df.copy()
    for step_name, step_function in PREPROCESSING_STEPS:
        start = time.perf_counter()
        # notebook: df['review'] = df['review'].apply(<function>)
        result[text_column] = result[text_column].apply(step_function)
        elapsed = time.perf_counter() - start
        logger.info("Step '%s' finished in %.2f s.", step_name, elapsed)
    return result


def save_processed(
    df: pd.DataFrame,
    path: Path = settings.PROCESSED_DATASET_PATH,
) -> Path:
    """Save the preprocessed DataFrame as a CSV file.

    Args:
        df: The preprocessed DataFrame.
        path: Output CSV path.

    Returns:
        The path of the saved file.

    Why: see knowledge.md, section "10. The Full Preprocessing Pipeline".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved preprocessed reviews to %s", path)
    return path


if __name__ == "__main__":
    for name, value in run_demos().items():
        print(f"{name}: {value}")
