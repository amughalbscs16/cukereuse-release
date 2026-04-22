"""Tests for the canonical phrasing picker."""

from __future__ import annotations

from pathlib import Path

from cukereuse.canonical import (
    _quoted_param_count,
    pick_canonical_step,
    pick_canonical_text,
)
from cukereuse.models import Step


def _step(text: str) -> Step:
    return Step(
        keyword="Given",
        text=text,
        file_path=Path("a.feature"),
        line=1,
        scenario_name="s",
        feature_name="f",
    )


def test_pick_canonical_text_frequency_first() -> None:
    # "popular" appears 3x, "rare" 1x -> popular wins even though it's longer
    texts = ["popular phrasing", "popular phrasing", "popular phrasing", "short"]
    assert pick_canonical_text(texts) == "popular phrasing"


def test_pick_canonical_text_brevity_breaks_frequency_ties() -> None:
    texts = ["I log in now", "I sign in"]
    assert pick_canonical_text(texts) == "I sign in"


def test_pick_canonical_text_prefers_fewer_quoted_params() -> None:
    # Same freq + same length → penalise the quoted-param version.
    texts = ['I act on "x"', "I act on foo"]
    assert pick_canonical_text(texts) == "I act on foo"


def test_pick_canonical_text_frequency_beats_quoted_penalty() -> None:
    # Higher-freq variant with quotes still wins over lower-freq plain variant.
    texts = [
        'the account is "test1"',
        'the account is "test1"',
        'the account is "test1"',
        "the account is default",
    ]
    assert pick_canonical_text(texts) == 'the account is "test1"'


def test_pick_canonical_text_alphabetical_deterministic_tiebreak() -> None:
    # Identical on all criteria → alphabetical
    assert pick_canonical_text(["banana", "apple"]) == "apple"


def test_pick_canonical_text_empty_input() -> None:
    assert pick_canonical_text([]) == ""


def test_pick_canonical_step_returns_representative() -> None:
    members = [
        _step("rare variant"),
        _step("common variant"),
        _step("common variant"),
    ]
    best = pick_canonical_step(members)
    assert best is not None
    assert best.text == "common variant"


def test_pick_canonical_step_none_for_empty() -> None:
    assert pick_canonical_step([]) is None


def test_quoted_param_count_counts_both_quote_styles() -> None:
    assert _quoted_param_count('a "b" c') == 1
    assert _quoted_param_count("x 'y' z") == 1
    assert _quoted_param_count('the "a" of "b"') == 2
    assert _quoted_param_count("no quotes here") == 0
