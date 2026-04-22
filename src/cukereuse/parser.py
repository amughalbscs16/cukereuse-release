"""Parse .feature files into a flat stream of :class:`Step` records.

Thin wrapper over ``gherkin-official`` (Cucumber's own authoritative parser).
Scenario Outlines are NOT unrolled — the raw phrasing with ``<placeholder>``
tokens is preserved so duplicate detection sees one pattern per outline, not
one per Examples row.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gherkin.parser import Parser

from cukereuse.models import ParseResult, Step

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

log = logging.getLogger(__name__)

_ALLOWED_KEYWORDS = frozenset({"Given", "When", "Then", "And", "But", "*"})


def _normalize_keyword(raw: str) -> str:
    kw = raw.strip()
    if kw in _ALLOWED_KEYWORDS:
        return kw
    return kw.rstrip("*").strip() or "*"


def _steps_from_block(
    *,
    block: dict[str, Any],
    file_path: Path,
    scenario_name: str,
    feature_name: str,
    tags: tuple[str, ...],
    is_background: bool,
    is_outline: bool,
) -> Iterable[Step]:
    for raw_step in block.get("steps") or ():
        kw = _normalize_keyword(raw_step.get("keyword", ""))
        text = (raw_step.get("text") or "").strip()
        line = int(raw_step.get("location", {}).get("line", 0))
        if not text or line <= 0:
            continue
        yield Step(
            keyword=kw,
            text=text,
            file_path=file_path,
            line=line,
            scenario_name=scenario_name,
            feature_name=feature_name,
            tags=tags,
            is_background=is_background,
            is_outline=is_outline,
        )


def _tag_names(tags: Iterable[dict[str, Any]] | None) -> tuple[str, ...]:
    return tuple((t.get("name") or "").strip() for t in (tags or ()) if t.get("name"))


def parse_file(path: Path) -> ParseResult:
    """Parse a single .feature file. Never raises — soft errors are captured."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ParseResult(file_path=path, error=f"read_error: {exc}")

    try:
        doc = Parser().parse(text)
    except Exception as exc:  # gherkin raises CompositeParserException; keep broad
        log.warning("gherkin parse failed for %s: %s", path, exc)
        return ParseResult(file_path=path, error=f"parse_error: {type(exc).__name__}: {exc}")

    feature = doc.get("feature")
    if not feature:
        return ParseResult(file_path=path)  # empty or only-comments file

    feature_name = (feature.get("name") or "").strip()
    feature_tags = _tag_names(feature.get("tags"))

    steps: list[Step] = []
    for child in feature.get("children") or ():
        if "background" in child:
            bg = child["background"]
            steps.extend(
                _steps_from_block(
                    block=bg,
                    file_path=path,
                    scenario_name="",
                    feature_name=feature_name,
                    tags=feature_tags,
                    is_background=True,
                    is_outline=False,
                )
            )
        elif "scenario" in child:
            sc = child["scenario"]
            # gherkin-official exposes "Scenario" and "Scenario Outline" via the
            # same key; the presence of `examples` identifies an outline.
            is_outline = bool(sc.get("examples"))
            sc_tags = _tag_names(sc.get("tags"))
            steps.extend(
                _steps_from_block(
                    block=sc,
                    file_path=path,
                    scenario_name=(sc.get("name") or "").strip(),
                    feature_name=feature_name,
                    tags=feature_tags + sc_tags,
                    is_background=False,
                    is_outline=is_outline,
                )
            )
        elif "rule" in child:
            # Rule wraps a nested sequence of backgrounds + scenarios.
            for sub in child["rule"].get("children") or ():
                if "background" in sub:
                    steps.extend(
                        _steps_from_block(
                            block=sub["background"],
                            file_path=path,
                            scenario_name="",
                            feature_name=feature_name,
                            tags=feature_tags,
                            is_background=True,
                            is_outline=False,
                        )
                    )
                elif "scenario" in sub:
                    sub_sc = sub["scenario"]
                    is_outline = bool(sub_sc.get("examples"))
                    sub_tags = _tag_names(sub_sc.get("tags"))
                    steps.extend(
                        _steps_from_block(
                            block=sub_sc,
                            file_path=path,
                            scenario_name=(sub_sc.get("name") or "").strip(),
                            feature_name=feature_name,
                            tags=feature_tags + sub_tags,
                            is_background=False,
                            is_outline=is_outline,
                        )
                    )

    return ParseResult(file_path=path, steps=tuple(steps))


def parse_directory(root: Path) -> Iterable[ParseResult]:
    """Parse every ``*.feature`` file under ``root`` (recursive)."""
    for p in sorted(root.rglob("*.feature")):
        if p.is_file():
            yield parse_file(p)
