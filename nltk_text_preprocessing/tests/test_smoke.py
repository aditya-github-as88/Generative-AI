"""Smoke tests: every module imports, every function gives the notebook output,
and the full pipeline runs on the placeholder data.

Run:
    pytest
"""

from __future__ import annotations

import importlib

import pytest

from utils.setup_resources import ensure_nltk_resources

MODULES = [
    "config.settings",
    "utils.logger",
    "utils.setup_resources",
    "src.data_loader",
    "src.eda",
    "src.text_cleaning",
    "src.normalization",
    "src.tokenization",
    "src.stopword_removal",
    "src.pipeline",
    "main",
]


@pytest.fixture(scope="session", autouse=True)
def nltk_ready() -> None:
    """Download NLTK resources one time. Skip all tests if that fails."""
    missing = ensure_nltk_resources()
    if missing:
        pytest.skip(f"NLTK resources not available (offline?): {missing}")


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


# --- Notebook outputs (each expected value is copied from the notebook) ---

def test_strip_html() -> None:
    from config.settings import DEMO_HTML
    from src.text_cleaning import strip_html

    lines = [line for line in strip_html(DEMO_HTML).splitlines() if line.strip()]
    assert lines == ["This is a test heading.", "Starting a paragraph here..."]


def test_remove_between_square_brackets() -> None:
    from src.text_cleaning import remove_between_square_brackets

    assert (
        remove_between_square_brackets("I bought [apple, banana, orange] yesterday.")
        == "I bought apple, banana, orange yesterday."
    )


def test_remove_special_characters() -> None:
    from src.text_cleaning import remove_special_characters

    assert remove_special_characters("Hi There! How are you?") == "Hi There How are you"


def test_lemmatize_word() -> None:
    from src.normalization import lemmatize_word

    assert lemmatize_word("better", pos="a") == "good"


def test_simple_stemmer() -> None:
    from src.normalization import simple_stemmer

    assert (
        simple_stemmer("He likes lemons bananas and oranges. He goes to market.")
        == "he like lemon banana and oranges. he goe to market."
    )


def test_tokenize_and_pos_tag() -> None:
    from src.tokenization import pos_tag_tokens, tokenize

    tokens = tokenize("I enjoy playing football in the rain.")
    assert tokens == ["I", "enjoy", "playing", "football", "in", "the", "rain", "."]
    tags = pos_tag_tokens(tokens)
    assert [token for token, _ in tags] == tokens
    assert dict(tags)["rain"] == "NN"


def test_updated_stopword_list_keeps_negations() -> None:
    from src.stopword_removal import get_stopword_list, get_updated_stopword_list

    full = get_stopword_list()
    updated = get_updated_stopword_list()
    assert "not" in full and "not" not in updated
    assert all(not word.endswith("n't") for word in updated)
    assert "the" in updated


def test_remove_stopwords() -> None:
    from src.stopword_removal import remove_stopwords

    assert remove_stopwords("The movie was not that great") == "movie not great"


# --- End-to-end on the data in data/raw (placeholder or real) ---

def test_pipeline_on_dataset(tmp_path) -> None:
    from config.settings import TEXT_COLUMN
    from src.data_loader import load_dataset
    from src.pipeline import preprocess_reviews, save_processed

    df = load_dataset(sample_size=20)
    cleaned = preprocess_reviews(df)
    assert cleaned.shape == df.shape
    joined = " ".join(cleaned[TEXT_COLUMN])
    assert "<" not in joined and "[" not in joined and "!" not in joined
    out = save_processed(cleaned, tmp_path / "out.csv")
    assert out.exists()


def test_eda_plot(tmp_path) -> None:
    from src.data_loader import load_dataset
    from src.eda import plot_sentiment_distribution

    df = load_dataset(sample_size=20)
    path = plot_sentiment_distribution(df, save_path=tmp_path / "plot.png", show=False)
    assert path.exists() and path.stat().st_size > 0


def test_main_runs() -> None:
    import main

    assert main.main(["--step", "all", "--sample", "20"]) == 0
