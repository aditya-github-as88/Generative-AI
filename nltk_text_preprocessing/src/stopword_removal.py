"""Stop-word removal that keeps negation words.

Purpose:
    Remove common English words that carry little meaning ("the", "is", ...),
    but KEEP "not" and every word that ends in "n't". Negations change the
    sentiment of a review, so they must stay.

Main functions:
    - ``get_stopword_list()``: the full NLTK English list.
    - ``build_updated_stopword_list()``: the list without negations.
    - ``get_updated_stopword_list()``: cached version of the line above.
    - ``remove_stopwords()``

Notebook cells covered:
    "Removing stop words" (``stopword_list``, ``updated_stopword_list``,
    ``remove_stopwords``).

Called by:
    ``src.pipeline``.
"""

from __future__ import annotations

from functools import lru_cache

from nltk.corpus import stopwords

from config.settings import NEGATION_SUFFIX, NEGATION_WORD, STOPWORD_LANGUAGE
from src.tokenization import tokenize


def get_stopword_list(language: str = STOPWORD_LANGUAGE) -> list[str]:
    """Return the NLTK stop-word list for a language.

    Args:
        language: NLTK language name, for example ``"english"``.

    Returns:
        The list of stop words.

    Why: see knowledge.md, section "9. Stop-Word Removal".
    """
    return stopwords.words(language)


def build_updated_stopword_list(stopword_list: list[str]) -> list[str]:
    """Remove "not" and every word that ends in "n't" from a stop-word list.

    Args:
        stopword_list: The original stop-word list.

    Returns:
        A new list without the negation words.

    Why: see knowledge.md, section "9. Stop-Word Removal".
    """
    updated_stopword_list: list[str] = []
    for word in stopword_list:
        if word == NEGATION_WORD or word.endswith(NEGATION_SUFFIX):
            pass
        else:
            updated_stopword_list.append(word)
    return updated_stopword_list


@lru_cache(maxsize=1)
def get_updated_stopword_list() -> tuple[str, ...]:
    """Return the English stop-word list without negations (built one time).

    Returns:
        A tuple of stop words. The tuple replaces the notebook's global
        variable ``updated_stopword_list``.

    Why: see knowledge.md, section "9. Stop-Word Removal".
    """
    return tuple(build_updated_stopword_list(get_stopword_list()))


def remove_stopwords(
    text: str,
    is_lower_case: bool = False,
    stopword_list: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Remove stop words from a text and keep negation words.

    Args:
        text: Input text.
        is_lower_case: Set True if the text is already lowercase. Then the
            function compares tokens as they are. If False, the function
            compares the lowercase form of each token.
        stopword_list: The stop words to remove. ``None`` uses
            ``get_updated_stopword_list()`` (the notebook behavior).

    Returns:
        The text without stop words, with tokens joined by single spaces.
        Example: ``"The movie was not that great"`` -> ``"movie not great"``.

    Why: see knowledge.md, section "9. Stop-Word Removal".
    """
    stop_set = set(stopword_list if stopword_list is not None else get_updated_stopword_list())

    # splitting strings into tokens (list of words)
    tokens = tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        # filtering out the stop words
        filtered_tokens = [token for token in tokens if token not in stop_set]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stop_set]
    return " ".join(filtered_tokens)


if __name__ == "__main__":
    from config.settings import DEMO_STOPWORD_TEXT

    print(get_stopword_list())
    print(list(get_updated_stopword_list()))
    print(remove_stopwords(DEMO_STOPWORD_TEXT))
