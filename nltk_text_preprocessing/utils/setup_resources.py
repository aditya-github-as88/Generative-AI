"""Download the NLTK data files that the project needs.

Purpose:
    Replace the notebook's ``nltk.download(...)`` lines. NLTK code (stop words,
    tokenizer, lemmatizer, POS tagger) needs extra data files that ``pip`` does
    not install.

Main functions:
    - ``ensure_nltk_resources()``: download each missing resource once.

Run alone:
    python -m utils.setup_resources

Called by:
    ``main.py`` (at start-up) and ``tests/test_smoke.py``.
"""

from __future__ import annotations

from pathlib import Path

import nltk

from config.settings import NLTK_DATA_DIR, NLTK_RESOURCES
from utils.logger import get_logger

logger = get_logger(__name__)

# Where NLTK keeps each resource inside its data folder.
_RESOURCE_LOCATIONS: dict[str, str] = {
    "stopwords": "corpora/stopwords",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


def _is_installed(resource: str) -> bool:
    """Return True if NLTK can find the resource on disk."""
    location = _RESOURCE_LOCATIONS.get(resource, resource)
    for candidate in (location, f"{location}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_resources(
    resources: tuple[str, ...] = NLTK_RESOURCES,
    download_dir: str | None = NLTK_DATA_DIR,
) -> list[str]:
    """Download every NLTK resource that is not installed yet.

    Args:
        resources: Names of the NLTK resources to check.
        download_dir: Optional folder for NLTK data. ``None`` uses the
            NLTK default folder.

    Returns:
        The names of resources that are still missing after the download.
        An empty list means that all resources are ready.

    Why: see knowledge.md, section "0. Supporting Code".
    """
    if download_dir:
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        if download_dir not in nltk.data.path:
            nltk.data.path.insert(0, download_dir)

    missing: list[str] = []
    for resource in resources:
        if _is_installed(resource):
            logger.debug("NLTK resource '%s' is already installed.", resource)
            continue
        logger.info("Downloading NLTK resource '%s' ...", resource)
        ok = nltk.download(resource, download_dir=download_dir, quiet=True)
        if not ok or not _is_installed(resource):
            missing.append(resource)

    if missing:
        logger.error(
            "These NLTK resources did not download: %s. "
            "Check your internet connection, then run: python -m utils.setup_resources",
            ", ".join(missing),
        )
    else:
        logger.info("All NLTK resources are ready.")
    return missing


if __name__ == "__main__":
    still_missing = ensure_nltk_resources()
    raise SystemExit(1 if still_missing else 0)
