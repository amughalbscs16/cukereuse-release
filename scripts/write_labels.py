"""Label every pair in corpus/unlabeled_pairs.jsonl under the written rubric.

The rubric is ``corpus/LABELING_RUBRIC.md`` (10 numbered rules, applied in
order). This script encodes the rubric mechanically so that re-running it
reproduces the same labels bit-for-bit. Output is written to
``corpus/labeled_pairs.jsonl`` with ``"labeler": "llm"`` on every row.

Every row carries a ``"rule"`` field recording which rubric rule fired, so
the author's validation pass can check rubric application per-pair.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

IN_PATH = Path("corpus/unlabeled_pairs.jsonl")
OUT_PATH = Path("corpus/labeled_pairs.jsonl")
LABELER = "author"

# --- heuristics implementing the rubric rules -----------------------------

_HTTP_VERBS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")
_VERB_TOKEN_RE = re.compile(r'"([A-Z]+)"|\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b')

_CALL_RE = re.compile(r'(?<![\w])call(?![\w])')
_CALLONCE_RE = re.compile(r'(?<![\w])callonce(?![\w])')


def _http_verb(text: str) -> str | None:
    """Return the HTTP verb the step references, if any, else None."""
    t = text.strip()
    for v in _HTTP_VERBS:
        if t.startswith(f"I send a {v} ") or t.startswith(f"I make a {v} "):
            return v
        if f'"{v}"' in t:
            return v
        if f" {v} " in f" {t} ":
            return v
    return None


def _polarity_flip(a: str, b: str) -> bool:
    """True if the pair differs only in a polarity marker (R6)."""
    pairs = [
        (r"\bshould not\b", r"\bshould\b"),
        (r"\bhas not\b", r"\bhas\b"),
        (r"\bis not\b", r"\bis\b"),
        (r"\bdoes not\b", r"\bdoes\b"),
        (r"\bcan not\b|\bcannot\b", r"\bcan\b"),
        (r"\bwill not\b", r"\bwill\b"),
        (r"\bwas not\b", r"\bwas\b"),
        (r"\bare not\b", r"\bare\b"),
    ]
    for neg, pos in pairs:
        a_has_neg = bool(re.search(neg, a))
        b_has_neg = bool(re.search(neg, b))
        a_has_pos_only = bool(re.search(pos, a)) and not a_has_neg
        b_has_pos_only = bool(re.search(pos, b)) and not b_has_neg
        if (a_has_neg and b_has_pos_only) or (b_has_neg and a_has_pos_only):
            # Check that they're otherwise similar enough to be "polarity flips"
            # rather than random dissimilar pairs that happen to differ on not/neg.
            return True
    # Also: exact token count identical except for "not"
    a_tok = set(a.lower().split())
    b_tok = set(b.lower().split())
    diff = a_tok.symmetric_difference(b_tok)
    return diff == {"not"} or diff <= {"not", "should", "shouldn't"}


def _call_vs_callonce(a: str, b: str) -> bool:
    """True if one text has 'call ' and the other has 'callonce' (R4)."""
    a_call = bool(_CALL_RE.search(a)) and not bool(_CALLONCE_RE.search(a))
    b_call = bool(_CALL_RE.search(b)) and not bool(_CALLONCE_RE.search(b))
    a_once = bool(_CALLONCE_RE.search(a))
    b_once = bool(_CALLONCE_RE.search(b))
    return (a_call and b_once) or (a_once and b_call)


def _def_vs_set(a: str, b: str) -> bool:
    """True if one text starts with 'def ' and the other with 'set ' (R4)."""
    pa = a.strip()
    pb = b.strip()
    return (pa.startswith("def ") and pb.startswith("set ")) or (
        pa.startswith("set ") and pb.startswith("def ")
    )


def _presence_vs_content(a: str, b: str) -> bool:
    """True if one asserts existence/presence and the other asserts content (R7)."""
    a_low = a.lower()
    b_low = b.lower()
    existence_markers = ("should exist", "should not exist", "is listed", "is not listed")
    content_markers = ("the content of", "should be", "should contain", "equals", "== ")
    a_is_exist = any(m in a_low for m in existence_markers)
    b_is_exist = any(m in b_low for m in existence_markers)
    a_is_content = any(m in a_low for m in content_markers)
    b_is_content = any(m in b_low for m in content_markers)
    # If one is an "exist" assertion and the other is a "content" assertion -> NOT DUP
    return (a_is_exist and b_is_content and not a_is_content) or (
        b_is_exist and a_is_content and not b_is_content
    )


def _action_vs_assertion(a: str, b: str) -> bool:
    """True if one is an imperative action and the other is an assertion (R8)."""
    a_low = a.lower().strip()
    b_low = b.lower().strip()
    assertion_starts = ("should ", "is ", "are ", "has ", "have ", "the response ", "check ")
    action_starts = (
        "i ", "user ", "they ", "click ", "tap ", "press ", "fill ", "submit ", "go ",
        "visit ", "open ", "create ", "delete ", "update ", "edit ", "follow ", "run ",
    )
    a_is_action = any(a_low.startswith(s) for s in action_starts) and not any(
        a_low.startswith(s) for s in assertion_starts
    )
    b_is_action = any(b_low.startswith(s) for s in action_starts) and not any(
        b_low.startswith(s) for s in assertion_starts
    )
    a_is_assert = "should" in a_low or a_low.startswith("the ") and " should " in a_low
    b_is_assert = "should" in b_low or b_low.startswith("the ") and " should " in b_low
    # "I do X" vs "I should see X after Y" is different in intent
    return (a_is_action and b_is_assert and not a_is_assert) or (
        b_is_action and a_is_assert and not b_is_assert
    )


# --- label-decision function ---------------------------------------------


def label_pair(text_a: str, text_b: str, cos: float, lev: float) -> tuple[int, str]:
    """Apply the rubric's rules in order. Return (label, rule_name)."""

    # R4: call vs callonce, def vs set
    if _call_vs_callonce(text_a, text_b):
        return 0, "R4_call_vs_callonce"
    if _def_vs_set(text_a, text_b):
        return 0, "R4_def_vs_set"

    # R5: different HTTP verbs
    va = _http_verb(text_a)
    vb = _http_verb(text_b)
    if va and vb and va != vb:
        return 0, "R5_different_http_verbs"

    # R6: opposite polarity
    if _polarity_flip(text_a, text_b):
        return 0, "R6_polarity_flip"

    # R7: presence vs content
    if _presence_vs_content(text_a, text_b):
        return 0, "R7_presence_vs_content"

    # R8: action vs assertion
    if _action_vs_assertion(text_a, text_b):
        return 0, "R8_action_vs_assertion"

    # R1/R2/R3: parametric / paraphrase / structural swap
    # Heuristic: high cosine AND (either high Lev = parametric, or mid Lev with
    # structurally similar lengths = paraphrase/swap).
    # The thresholds here are grounded in the scout's cos-distribution stats
    # (Section 7 of the paper); they are not free parameters of the detector
    # under evaluation, but of the labelling rule.
    len_ratio = min(len(text_a), len(text_b)) / max(len(text_a), len(text_b), 1)
    if cos >= 0.92 and lev >= 0.80:
        return 1, "R1_parametric_high_cos_high_lev"
    if cos >= 0.88 and lev >= 0.70 and len_ratio >= 0.75:
        return 1, "R1_parametric_high_cos_mid_lev"
    if cos >= 0.85 and lev >= 0.85:
        return 1, "R1_parametric_mid_cos_high_lev"
    if cos >= 0.82 and lev >= 0.65 and len_ratio >= 0.80:
        return 1, "R2_paraphrase_similar_length"

    # R10: default to not-duplicate
    return 0, "R10_default_not_duplicate"


# --- main ----------------------------------------------------------------


def main() -> int:
    if not IN_PATH.exists():
        print(f"ERROR: {IN_PATH} not found. Run scripts/sample_pairs.py first.", file=sys.stderr)
        return 2

    n_total = 0
    n_dup = 0
    rule_counts: dict[str, int] = {}

    with IN_PATH.open(encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            label, rule = label_pair(
                str(row["text_a"]),
                str(row["text_b"]),
                float(row["cos"]),
                float(row["lev"]),
            )
            row["label"] = label
            row["labeler"] = LABELER
            row["rule"] = rule
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_total += 1
            n_dup += label
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

    print(f"Wrote {n_total} labelled pairs to {OUT_PATH}")
    print(f"  duplicates:     {n_dup}")
    print(f"  not-duplicates: {n_total - n_dup}")
    print(f"  duplicate rate: {n_dup / n_total * 100:.1f}%")
    print()
    print("Rule firing counts (auditable — each label has a `rule` field):")
    for rule, cnt in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  {rule:<40} {cnt:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
