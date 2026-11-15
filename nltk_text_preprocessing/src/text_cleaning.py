"""Noise removal and special-character removal.

Purpose:
    Clean raw review text: remove HTML markup, remove square brackets, and
    remove every character that is not a letter, a digit or whitespace.

Main functions:
    - ``strip_html()``
    - ``remove_between_square_brackets()``
    - ``denoise_text()``
    - ``remove_special_characters()``

Notebook cells covered:
    "Removing noisy text" and "Removing special characters".

Called by:
    ``src.pipeline``.
"""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from config.settings import SPECIAL_CHAR_PATTERN

# Short reviews that look like a file name or URL make BeautifulSoup print a
# warning. The text is still parsed correctly, so hide only this warning.
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

_SPECIAL_CHAR_REGEX = re.compile(SPECIAL_CHAR_PATTERN)


def strip_html(text: str) -> str:
    """Remove HTML tags and keep only the visible text.

    Args:
        text: Raw text that can contain HTML such as ``<br />``.

    Returns:
        The text without HTML tags.

    Why: see knowledge.md, section "3. Noise Removal".
    """
    # BeautifulSoup is a useful library for extracting data from HTML and XML documents
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text: str) -> str:
    """Remove the square-bracket characters but keep the text inside them.

    Args:
        text: Input text.

    Returns:
        The text without ``[`` and ``]``.

    Why: see knowledge.md, section "3. Noise Removal".
    """
    text = text.replace("[", "")
    text = text.replace("]", "")
    return text
    # Alternative from the notebook (removes the brackets AND the text inside):
    # return re.sub(r'\[[^]]*\]', '', text)


def denoise_text(text: str) -> str:
    """Apply all noise-removal steps: HTML, then square brackets.

    Args:
        text: Raw review text.

    Returns:
        The review text without HTML tags and square brackets.

    Why: see knowledge.md, section "3. Noise Removal".
    """
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


def remove_special_characters(text: str) -> str:
    """Remove every character that is not a letter, a digit or whitespace.

    Args:
        text: Input text.

    Returns:
        The text with only ``a-z``, ``A-Z``, ``0-9`` and whitespace.

    Why: see knowledge.md, section "4. Special Character Removal".
    """
    return _SPECIAL_CHAR_REGEX.sub("", text)


if __name__ == "__main__":
    from config.settings import DEMO_BRACKETS, DEMO_HTML, DEMO_SPECIAL_CHARS

    print(strip_html(DEMO_HTML))
    print(remove_between_square_brackets(DEMO_BRACKETS))
    print(remove_special_characters(DEMO_SPECIAL_CHARS))
