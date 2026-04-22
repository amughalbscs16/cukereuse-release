"""Pydantic models for the cukereuse data pipeline."""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003  # pydantic resolves annotations at runtime

from pydantic import BaseModel, ConfigDict, Field


class Step(BaseModel):
    """A single Gherkin step extracted from a .feature file.

    `text` is the step phrasing only — doc strings and data tables are treated
    as step arguments and not included here. Placeholder tokens from Scenario
    Outlines (e.g. ``<username>``) are preserved verbatim in the text.
    """

    model_config = ConfigDict(frozen=True)

    keyword: str = Field(..., description="Given | When | Then | And | But")
    text: str
    file_path: Path
    line: int = Field(..., gt=0)
    scenario_name: str
    feature_name: str
    tags: tuple[str, ...] = Field(default_factory=tuple)
    is_background: bool = False
    is_outline: bool = False


class ParseResult(BaseModel):
    """Outcome of parsing a single file — steps plus any soft errors."""

    model_config = ConfigDict(frozen=True)

    file_path: Path
    steps: tuple[Step, ...] = Field(default_factory=tuple)
    error: str | None = None
