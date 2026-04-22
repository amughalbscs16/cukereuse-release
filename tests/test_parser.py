"""Unit tests for the Gherkin parser wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from cukereuse.parser import parse_directory, parse_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_minimal_feature_yields_three_steps() -> None:
    result = parse_file(FIXTURES / "minimal.feature")
    assert result.error is None
    assert len(result.steps) == 3

    kws = [s.keyword for s in result.steps]
    assert kws == ["Given", "When", "Then"]

    texts = [s.text for s in result.steps]
    assert texts == [
        'a user named "Alice"',
        "the user says hello",
        'the response is "Hi Alice"',
    ]

    for step in result.steps:
        assert step.feature_name == "Minimal feature with one scenario"
        assert step.scenario_name == "greet"
        assert step.is_background is False
        assert step.is_outline is False
        assert step.line > 0


def test_background_steps_are_flagged() -> None:
    result = parse_file(FIXTURES / "background_and_outline.feature")
    assert result.error is None

    bg_steps = [s for s in result.steps if s.is_background]
    assert [s.text for s in bg_steps] == [
        "the service is running",
        "the user is authenticated",
    ]
    assert all(s.scenario_name == "" for s in bg_steps)


def test_feature_tags_propagate() -> None:
    result = parse_file(FIXTURES / "background_and_outline.feature")
    # Every step should at least have the two feature-level tags.
    for step in result.steps:
        assert "@smoke" in step.tags
        assert "@api" in step.tags


def test_scenario_tags_combine_with_feature_tags() -> None:
    result = parse_file(FIXTURES / "background_and_outline.feature")
    happy_steps = [s for s in result.steps if s.scenario_name == "straightforward login"]
    assert happy_steps, "expected @happy scenario to produce steps"
    assert all("@happy" in s.tags for s in happy_steps)


def test_outline_steps_preserve_placeholders_and_are_flagged() -> None:
    result = parse_file(FIXTURES / "background_and_outline.feature")
    outline_steps = [s for s in result.steps if s.is_outline]
    assert outline_steps, "expected outline steps"
    # The outline body should reference <role> once per step, unchanged.
    role_phrases = [s for s in outline_steps if "<role>" in s.text]
    assert len(role_phrases) == 2  # one When, one Then


def test_doc_string_and_data_table_are_excluded_from_step_text() -> None:
    result = parse_file(FIXTURES / "background_and_outline.feature")
    payload_steps = [s for s in result.steps if s.scenario_name == "multiline payload"]
    posts_step = next(s for s in payload_steps if s.keyword == "When")
    assert posts_step.text == "the client posts the payload"

    table_steps = [s for s in result.steps if s.scenario_name == "payload with table"]
    posts_users = next(s for s in table_steps if s.keyword == "When")
    assert posts_users.text == "the client posts users"


def test_rule_blocks_are_flattened() -> None:
    result = parse_file(FIXTURES / "with_rule.feature")
    assert result.error is None

    names = {s.scenario_name for s in result.steps if not s.is_background}
    assert names == {"missing token", "list users"}

    # Background from Rule 2 applies to "list users" scenario context
    bg_steps = [s for s in result.steps if s.is_background]
    assert any(s.text == "a valid token is provided" for s in bg_steps)


def test_empty_file_parses_to_zero_steps_without_error() -> None:
    result = parse_file(FIXTURES / "empty.feature")
    assert result.error is None
    assert result.steps == ()


def test_malformed_file_captures_soft_error() -> None:
    result = parse_file(FIXTURES / "malformed.feature")
    assert result.error is not None
    assert "parse_error" in result.error
    # Should not have raised; returning a ParseResult with error is the contract.


def test_parse_directory_yields_all_fixtures() -> None:
    results = list(parse_directory(FIXTURES))
    assert len(results) >= 5
    assert all(r.file_path.suffix == ".feature" for r in results)


def test_nonexistent_file_returns_read_error() -> None:
    result = parse_file(FIXTURES / "does_not_exist.feature")
    assert result.error is not None
    assert result.error.startswith("read_error:")
    assert result.steps == ()


@pytest.mark.parametrize(
    "fixture_name, expected_min_steps",
    [
        ("minimal.feature", 3),
        ("background_and_outline.feature", 10),
        ("with_rule.feature", 5),
    ],
)
def test_fixture_step_counts(fixture_name: str, expected_min_steps: int) -> None:
    result = parse_file(FIXTURES / fixture_name)
    assert result.error is None
    assert len(result.steps) >= expected_min_steps
