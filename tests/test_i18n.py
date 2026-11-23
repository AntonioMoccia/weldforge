"""The operator's language.

Two properties matter more than the wording. A missing or broken translation
must leave the application readable in English rather than blank or crashed --
a catalogue is edited by hand and will be wrong sometimes. And switching has
to work in both directions: once a label reads "Importa CAD" the English is
gone unless something remembered it, and a one-way language switch is a trap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from weldforge import i18n


@pytest.fixture(autouse=True)
def _english_afterwards():
    """Leave the process in English: the language is module state."""
    yield
    i18n.set_language(i18n.DEFAULT_LANGUAGE)


# ---------------------------------------------------------------------------
# the catalogue
# ---------------------------------------------------------------------------


def test_italian_is_available_and_english_comes_first():
    assert i18n.available_languages()[0] == i18n.DEFAULT_LANGUAGE
    assert "it" in i18n.available_languages()


def test_the_italian_catalogue_is_valid_and_keeps_its_placeholders():
    """A translation that drops ``{name}`` silently loses the value in it."""
    import re

    raw = json.loads(i18n.catalogue_path("it").read_text(encoding="utf-8"))
    assert raw, "the catalogue is empty"

    for source, translated in raw.items():
        assert isinstance(translated, str) and translated, source
        fields = set(re.findall(r"\{(\w+)", source))
        assert set(re.findall(r"\{(\w+)", translated)) == fields, source


def test_translating_and_coming_back():
    assert i18n.t("Import CAD") == "Import CAD"
    assert i18n.set_language("it") == "it"
    assert i18n.t("Import CAD") == "Importa CAD"
    assert i18n.set_language("en") == "en"
    assert i18n.t("Import CAD") == "Import CAD"


def test_a_string_with_no_entry_falls_back_to_english():
    """A half-finished catalogue must degrade to a mixture, not to blanks."""
    i18n.set_language("it")
    assert i18n.t("a string nobody has translated") == "a string nobody has translated"


def test_placeholders_are_filled_in_either_language():
    assert i18n.t("the bench") == "the bench"
    i18n.set_language("it")
    filled = i18n.t("{what} left where it was", what=i18n.t("the bench"))
    assert "il banco" in filled
    assert "{" not in filled


def test_an_unknown_language_stays_in_english():
    assert i18n.set_language("kl") == "en"
    assert i18n.t("Import CAD") == "Import CAD"


def test_a_broken_catalogue_stays_in_english(tmp_path: Path, monkeypatch):
    bad = tmp_path / "xx.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(i18n, "_data_dir", lambda: tmp_path)

    assert i18n.set_language("xx") == "en"
    assert i18n.t("Import CAD") == "Import CAD"


def test_a_translation_with_the_wrong_placeholder_does_not_raise(tmp_path: Path, monkeypatch):
    """A catalogue typo must not take the window down mid-draw."""
    (tmp_path / "xx.json").write_text(
        json.dumps({"{what} moved": "{quoi} spostato"}), encoding="utf-8"
    )
    monkeypatch.setattr(i18n, "_data_dir", lambda: tmp_path)
    i18n.set_language("xx")

    assert i18n.t("{what} moved", what="the bench") == "the bench moved"


# ---------------------------------------------------------------------------
# remembering the choice
# ---------------------------------------------------------------------------


def test_the_choice_is_remembered(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WELDFORGE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.delenv("WELDFORGE_LANGUAGE", raising=False)

    assert i18n.load_preferred_language("en") == "en"
    i18n.save_preferred_language("it")
    assert i18n.load_preferred_language("en") == "it"


def test_saving_the_language_keeps_other_settings(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"something_else": 7}), encoding="utf-8")
    monkeypatch.setenv("WELDFORGE_SETTINGS", str(settings))

    i18n.save_preferred_language("it")
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data == {"something_else": 7, "language": "it"}


def test_the_environment_can_pin_the_language(tmp_path: Path, monkeypatch):
    """So a shop can set it for every account from one place."""
    monkeypatch.setenv("WELDFORGE_SETTINGS", str(tmp_path / "settings.json"))
    i18n.save_preferred_language("en")
    monkeypatch.setenv("WELDFORGE_LANGUAGE", "it")

    assert i18n.load_preferred_language() == "it"


def test_unreadable_settings_do_not_stop_the_application(tmp_path: Path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text("{{{", encoding="utf-8")
    monkeypatch.setenv("WELDFORGE_SETTINGS", str(settings))
    monkeypatch.delenv("WELDFORGE_LANGUAGE", raising=False)

    assert i18n.load_preferred_language("it") == "it"
