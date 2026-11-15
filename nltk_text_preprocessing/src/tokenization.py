"""Tokenization and part-of-speech (POS) tagging.

Purpose:
    Split text into word tokens, and label each token with its word class.

Main functions:
    - ``tokenize()``
    - ``pos_tag_tokens()``

Notebook cells covered:
    "Tokenization" and "Part-of-speech tagging" (demo only in the notebook).

Called by:
    ``src.pipeline`` and ``src.stopword_removal``.
"""

from __future__ import annotations

import nltk
from nltk.tokenize import word_tokenize


def tokenize(text: str) -> list[str]:
    """Split text into word and punctuation tokens with NLTK.

    Args:
        text: Input text.

    Returns:
        A list of tokens. Example:
        ``"I enjoy playing football in the rain."`` ->
        ``['I', 'enjoy', 'playing', 'football', 'in', 'the', 'rain', '.']``.

    Why: see knowledge.md, section "7. Tokenization".
    """
    return word_tokenize(text)


def pos_tag_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    """Tag each token with its Penn Treebank part-of-speech tag.

    Args:
        tokens: A list of tokens from ``tokenize()``.

    Returns:
        A list of ``(token, tag)`` pairs, for example ``('rain', 'NN')``.

    Why: see knowledge.md, section "8. Part-of-Speech Tagging".
    """
    return nltk.pos_tag(tokens)


if __name__ == "__main__":
    from config.settings import DEMO_TOKENIZE_TEXT

    demo_tokens = tokenize(DEMO_TOKENIZE_TEXT)
    print(demo_tokens)
    print(pos_tag_tokens(demo_tokens))
