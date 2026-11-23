"""Translating what the operator reads.

Why not Qt's translation system. Half the text an operator sees is written by
modules that are deliberately free of Qt -- the interactive tools, the CAD
importer's refusals, the seam detector's explanation of why it found nothing.
``QObject.tr`` is not available there, and splitting the application's words
across two mechanisms would mean two catalogues and two ways to be out of
date. So there is one function, it needs nothing but the standard library, and
every layer uses it.

How it works. The source strings are English, because the code and its
comments are English and a catalogue keyed by English reads as a glossary
rather than as a lookup table. A catalogue is a flat JSON object mapping the
English to the target language; a string with no entry falls through to the
English, so a half-finished translation degrades to a readable mixture rather
than to blanks.

.. code-block:: python

    from ..i18n import t

    t("Import CAD")
    t("{count} seams, {mm} mm of weld", count=4, mm=312)

Placeholders are named, never positional: a translator has to be free to put
them in a different order, which ``{}`` does not allow.

The language is a per-user setting, remembered between sessions. The library
default is English so that the command line, the REST API and the test suite
keep their wording; the desktop application picks the operator's language at
start-up.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGE_NAMES",
    "available_languages",
    "catalogue_path",
    "language",
    "set_language",
    "settings_path",
    "load_preferred_language",
    "save_preferred_language",
    "t",
]

#: The language the source strings are written in. Needs no catalogue.
DEFAULT_LANGUAGE = "en"

#: Shown in the language menu, each in its own language -- an operator looking
#: for their language should not have to read another one to find it.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "it": "Italiano",
}

_language = DEFAULT_LANGUAGE
_catalogue: dict[str, str] = {}


def _data_dir() -> Path:
    from .wps.tables import DATA_DIR

    return Path(DATA_DIR) / "i18n"


def catalogue_path(code: str) -> Path:
    return _data_dir() / f"{code}.json"


def available_languages() -> list[str]:
    """Every language that can be selected, English first."""
    folder = _data_dir()
    codes = {DEFAULT_LANGUAGE}
    if folder.is_dir():
        codes |= {path.stem for path in folder.glob("*.json")}
    return [DEFAULT_LANGUAGE, *sorted(c for c in codes if c != DEFAULT_LANGUAGE)]


def language() -> str:
    return _language


def set_language(code: str) -> str:
    """Switch language, returning the one actually in use.

    A language with no catalogue is not worth refusing over: the application
    stays readable in English, and the log says what happened.
    """
    global _language, _catalogue

    code = (code or DEFAULT_LANGUAGE).strip().lower()
    if code == DEFAULT_LANGUAGE:
        _language, _catalogue = DEFAULT_LANGUAGE, {}
        return _language

    path = catalogue_path(code)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("no catalogue for %r at %s; staying in English", code, path)
        _language, _catalogue = DEFAULT_LANGUAGE, {}
        return _language
    except (OSError, ValueError) as exc:
        logger.error("catalogue %s is unreadable (%s); staying in English", path, exc)
        _language, _catalogue = DEFAULT_LANGUAGE, {}
        return _language

    _catalogue = {str(k): str(v) for k, v in raw.items() if isinstance(v, str) and v}
    _language = code
    logger.info("language %s: %d strings", code, len(_catalogue))
    return _language


def t(text: str, /, **values: Any) -> str:
    """The operator's wording for *text*, with named placeholders filled in.

    Falls back to the English on a missing entry, and to the English on a
    translation whose placeholders do not match -- a catalogue typo must not
    raise in the middle of drawing a window.
    """
    translated = _catalogue.get(text, text)
    if not values:
        return translated
    try:
        return translated.format(**values)
    except (KeyError, IndexError, ValueError):
        logger.error("bad placeholders in %r (for %r)", translated, text)
        try:
            return text.format(**values)
        except (KeyError, IndexError, ValueError):
            return text


# ---------------------------------------------------------------------------
# remembering the choice
# ---------------------------------------------------------------------------


def settings_path() -> Path:
    """Where the chosen language is kept, beside the recent-projects list."""
    override = os.environ.get("WELDFORGE_SETTINGS")
    if override:
        return Path(override)
    return Path.home() / ".config" / "weldforge" / "settings.json"


def load_preferred_language(fallback: str = DEFAULT_LANGUAGE) -> str:
    """The language this user last chose, or *fallback*.

    ``WELDFORGE_LANGUAGE`` wins, so a shop can pin the language for every
    account from one place without editing anybody's settings file.
    """
    forced = os.environ.get("WELDFORGE_LANGUAGE")
    if forced:
        return forced.strip().lower()
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
    chosen = data.get("language") if isinstance(data, dict) else None
    return str(chosen).strip().lower() if chosen else fallback


def save_preferred_language(code: str) -> None:
    """Remember the choice, leaving anything else in the file alone."""
    path = settings_path()
    data: dict[str, Any] = {}
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            data = existing
    except (OSError, ValueError):
        data = {}
    data["language"] = code
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.error("could not save the language choice to %s: %s", path, exc)
