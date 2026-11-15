"""Word normalization: lemmatization and stemming.

Purpose:
    Reduce words to a base form so that different forms of one word
    ("likes", "like") count as the same word.

Main functions:
    - ``lemmatize_word()``: WordNet lemmatizer (demo only, as in the notebook).
    - ``simple_stemmer()``: Porter stemmer on every word of a text.

Notebook cells covered:
    "Lemmatization" and "Text Stemming".

Called by:
    ``src.pipeline``.
"""

from __future__ import annotations

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Create the NLTK objects one time. The notebook created a new stemmer for
# every review; one shared object gives the same result and is faster.
_lemmatizer = WordNetLemmatizer()
_stemmer = PorterStemmer()


def lemmatize_word(word: str, pos: str = "n") -> str:
    """Return the lemma (dictionary form) of one word.

    Args:
        word: The word to lemmatize.
        pos: The part-of-speech tag. Valid options are ``"n"`` (noun),
            ``"v"`` (verb), ``"a"`` (adjective), ``"r"`` (adverb) and
            ``"s"`` (satellite adjective).

    Returns:
        The lemma. Example: ``lemmatize_word("better", pos="a")`` -> ``"good"``.

    Why: see knowledge.md, section "5. Lemmatization".
    """
    return _lemmatizer.lemmatize(word, pos=pos)


def simple_stemmer(text: str) -> str:
    """Stem every whitespace-separated word with the Porter stemmer.

    Args:
        text: Input text.

    Returns:
        The stemmed text, in lowercase, with words joined by single spaces.

    Why: see knowledge.md, section "6. Stemming".
    """
    return " ".join([_stemmer.stem(word) for word in text.split()])


if __name__ == "__main__":
    from config.settings import DEMO_LEMMA_POS, DEMO_LEMMA_WORD, DEMO_STEM_TEXT

    print("better :", lemmatize_word(DEMO_LEMMA_WORD, pos=DEMO_LEMMA_POS))
    print(simple_stemmer(DEMO_STEM_TEXT))
