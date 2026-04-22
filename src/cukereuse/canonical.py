"""Pick the canonical phrasing to display for a duplicate cluster.

Implements the plan's §2.4(c) ordering: frequency > brevity > low project-
specific-noun count, with deterministic alphabetical tiebreaking. A "project-
specific noun" is proxied by count of quoted parameter strings in the text —
``"test1"``, ``'admin'``, etc. are typically local fixture identifiers that a
general-purpose canonical should avoid highlighting.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cukereuse.models import Step


_QUOTED_PARAM_RE = re.compile(r'"[^"\\]*"|\'[^\'\\]*\'')


def _quoted_param_count(text: str) -> int:
    """Rough count of quoted fixture identifiers (``"test1"``, ``'admin'``)."""
    return len(_QUOTED_PARAM_RE.findall(text))


def pick_canonical_text(texts: Iterable[str]) -> str:
    """Select one representative phrasing from a non-empty iterable of texts.

    Ordering (lower score wins):
      1. higher frequency (-count)
      2. fewer quoted-parameter tokens — generic phrasings preferred
      3. shorter length
      4. alphabetical — deterministic

    Returns the empty string if the input yields no texts.
    """
    counts = Counter(texts)
    if not counts:
        return ""

    def score(t: str) -> tuple[int, int, int, str]:
        return (-counts[t], _quoted_param_count(t), len(t), t)

    return min(counts.keys(), key=score)


def pick_canonical_step(members: Iterable[Step]) -> Step | None:
    """Pick the Step whose text wins :func:`pick_canonical_text`.

    Returns the first Step with the winning text (insertion order), or None
    if ``members`` is empty. Useful when the caller wants to preserve Step
    metadata (file_path, line, keyword) from a representative member.
    """
    member_list = list(members)
    if not member_list:
        return None
    best_text = pick_canonical_text(m.text for m in member_list)
    return next(m for m in member_list if m.text == best_text)
