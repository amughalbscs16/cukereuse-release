"""Smoke tests — verify the package imports and CLI boots."""

from __future__ import annotations

from typer.testing import CliRunner

import cukereuse
from cukereuse.cli import app


def test_package_has_version() -> None:
    assert cukereuse.__version__ == "0.1.0"


def test_cli_version_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout
