"""Tests for text-level similarity primitives."""

from __future__ import annotations

import pytest

from cukereuse.similarity import content_hash, length_compatible, lev_ratio, normalize


def test_normalize_trims_and_collapses_whitespace() -> None:
    assert normalize("  hello    world  ") == "hello world"
    assert normalize("one\n\ttwo") == "one two"
    assert normalize("already fine") == "already fine"


def test_normalize_is_case_sensitive() -> None:
    assert normalize("Hello") != normalize("hello")


def test_content_hash_is_stable_across_whitespace_variants() -> None:
    assert content_hash("a  b") == content_hash("a b")
    assert content_hash(" hello") == content_hash("hello ")


def test_content_hash_differs_on_content_change() -> None:
    assert content_hash("foo") != content_hash("bar")
    assert content_hash("I log in") != content_hash("I log out")


def test_content_hash_is_hex_of_expected_width() -> None:
    h = content_hash("anything")
    assert len(h) == 32
    assert all(c in "0123456789abcdef" for c in h)


def test_lev_ratio_identical_is_one() -> None:
    assert lev_ratio("hello", "hello") == 1.0


def test_lev_ratio_completely_different_is_low() -> None:
    assert lev_ratio("abc", "xyz") == pytest.approx(0.0)


def test_lev_ratio_one_edit_is_high() -> None:
    # "the current account is test1" vs "the current account is test2" — exactly
    # the kind of parameterised variant the MVP misses.
    r = lev_ratio(
        'the current account is "test1"',
        'the current account is "test2"',
    )
    assert r > 0.95


def test_lev_ratio_respects_normalization() -> None:
    # Whitespace noise shouldn't lower the score.
    assert lev_ratio("  hello  world  ", "hello world") == 1.0


def test_length_compatible_matches_when_lengths_close() -> None:
    assert length_compatible("hello", "hella", 0.8) is True
    assert length_compatible("abcdefghij", "abcdefghij", 0.9) is True


def test_length_compatible_rejects_on_very_different_lengths() -> None:
    # A 3-char vs 30-char string cannot possibly reach ratio 0.85
    assert length_compatible("abc", "a" * 30, 0.85) is False


def test_length_compatible_handles_empty_strings() -> None:
    assert length_compatible("", "", 0.85) is True
    assert length_compatible("", "x", 0.85) is False
