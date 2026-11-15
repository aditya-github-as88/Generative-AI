"""Central settings for the project: paths, constants, flags and env loading.

Purpose:
    Keep every path, file name, constant and feature flag in one place, so no
    other module hard-codes a value.

Main contents:
    - Paths built from ``PROJECT_ROOT``.
    - Dataset and column names.
    - NLTK resource names and preprocessing constants.
    - Flags read from the optional ``.env`` file.
    - ``validate_settings()`` that checks the dataset file exists.

Called by:
    Every other module (``utils``, ``src``, ``main.py``, ``tests``).
    This module imports nothing from the project (bottom of the import layers).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"

# Load optional settings from `.env` (no secrets are needed in this project).
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    """Read a true/false value from the environment."""
    value = os.getenv(name, "").strip().lower()
    if value == "":
        return default
    return value in {"1", "true", "yes", "on"}


def _env_int_or_none(name: str) -> int | None:
    """Read a positive integer from the environment, or None if it is empty."""
    value = os.getenv(name, "").strip()
    if value == "":
        return None
    number = int(value)
    return number if number > 0 else None


# --- Dataset ---
# Financial news sentiment dataset: 2 columns, NO header row,
# column order = sentiment label, then text. Labels: positive/neutral/negative.
DATASET_FILENAME: str = "all-data.csv"
DATASET_PATH: Path = RAW_DIR / DATASET_FILENAME
# The file is not valid UTF-8 (it has Latin-1 characters such as accents).
DATASET_ENCODING: str = "latin-1"
DATASET_HAS_HEADER: bool = False
# A marker file next to the CSV. If it exists, the CSV is placeholder data.
PLACEHOLDER_MARKER_PATH: Path = RAW_DIR / "all-data.PLACEHOLDER.md"
PROCESSED_DATASET_PATH: Path = PROCESSED_DIR / "all-data_preprocessed.csv"
SENTIMENT_PLOT_PATH: Path = FIGURES_DIR / "sentiment_distribution.png"

# Original source cited by the notebook (raw text archive, not the CSV).
DATASET_SOURCE_URL: str = "https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news"

TEXT_COLUMN: str = "review"
LABEL_COLUMN: str = "sentiment"

# Column order in the raw CSV (used because the file has no header row).
# Internal names stay "review"/"sentiment" so the rest of the pipeline is unchanged.
CSV_COLUMN_ORDER: tuple[str, str] = (LABEL_COLUMN, TEXT_COLUMN)

# --- Display ---
PREVIEW_ROWS: int = 10           # notebook: df.head(10)
SAMPLE_REVIEW_INDEX: int = 5     # notebook: df["review"][5]

# --- NLTK resources (notebook: nltk.download(...) calls) ---
NLTK_RESOURCES: tuple[str, ...] = (
    "stopwords",                       # stop-word lists
    "punkt",                           # Punkt tokenizer model (older NLTK)
    "punkt_tab",                       # Punkt tokenizer tables (NLTK >= 3.8.2)
    "wordnet",                         # WordNet for the lemmatizer
    "averaged_perceptron_tagger",      # POS tagger (older NLTK)
    "averaged_perceptron_tagger_eng",  # POS tagger (NLTK >= 3.9)
)
NLTK_DATA_DIR: str | None = os.getenv("NLTK_DATA") or None

# --- Preprocessing constants ---
# Keep letters, digits and whitespace; remove everything else.
SPECIAL_CHAR_PATTERN: str = r"[^a-zA-Z0-9\s]"
STOPWORD_LANGUAGE: str = "english"
# Words to KEEP even though NLTK lists them as stop words (negations).
NEGATION_WORD: str = "not"
NEGATION_SUFFIX: str = "n't"

# --- Demo inputs (the exact examples from the notebook) ---
DEMO_HTML: str = """<body>
<h1>This is a test heading.</h1>
<p>Starting a paragraph here...</p>
</body>"""
DEMO_BRACKETS: str = "I bought [apple, banana, orange] yesterday."
DEMO_SPECIAL_CHARS: str = "Hi There! How are you?"
DEMO_LEMMA_WORD: str = "better"
DEMO_LEMMA_POS: str = "a"
DEMO_STEM_TEXT: str = "He likes lemons bananas and oranges. He goes to market."
DEMO_TOKENIZE_TEXT: str = "I enjoy playing football in the rain."
DEMO_STOPWORD_TEXT: str = "The movie was not that great"

# --- Flags ---
SHOW_PLOTS: bool = _env_bool("SHOW_PLOTS", default=False)
SAMPLE_SIZE: int | None = _env_int_or_none("SAMPLE_SIZE")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def validate_settings(dataset_path: Path = DATASET_PATH) -> None:
    """Check that the settings needed to run the pipeline are valid.

    Args:
        dataset_path: Path to the dataset CSV file.

    Raises:
        FileNotFoundError: If the dataset file does not exist. The message
            tells the user what to do.

    Why: see knowledge.md, section "1. Loading the Dataset".
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Action: put '{DATASET_FILENAME}' in '{RAW_DIR}'. "
            "See data/README.md for the download source."
        )
