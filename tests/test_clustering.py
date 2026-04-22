"""Tests for exact-duplicate clustering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cukereuse.clustering import (
    cluster_exact,
    cluster_hybrid,
    cluster_near_exact,
    cluster_semantic,
)
from cukereuse.models import Step


def _step(text: str, line: int = 1, file: str = "a.feature") -> Step:
    return Step(
        keyword="Given",
        text=text,
        file_path=Path(file),
        line=line,
        scenario_name="s",
        feature_name="f",
    )


def test_single_occurrence_is_dropped() -> None:
    clusters = cluster_exact([_step("I log in")])
    assert clusters == []


def test_two_identical_steps_form_one_cluster() -> None:
    steps = [_step("I log in", line=1), _step("I log in", line=5)]
    clusters = cluster_exact(steps)
    assert len(clusters) == 1
    c = clusters[0]
    assert c.canonical_text == "I log in"
    assert c.count == 2
    assert c.strategy == "exact"


def test_whitespace_variants_collapse_into_same_cluster() -> None:
    steps = [
        _step("I  log   in"),
        _step("I log in"),
        _step(" I log in "),
    ]
    clusters = cluster_exact(steps)
    assert len(clusters) == 1
    assert clusters[0].count == 3


def test_distinct_texts_produce_distinct_clusters() -> None:
    steps = [
        _step("I log in", line=1),
        _step("I log in", line=2),
        _step("I log out", line=3),
        _step("I log out", line=4),
        _step("I log out", line=5),
    ]
    clusters = cluster_exact(steps)
    assert len(clusters) == 2
    # Sorted by count desc
    assert clusters[0].canonical_text == "I log out"
    assert clusters[0].count == 3
    assert clusters[1].canonical_text == "I log in"
    assert clusters[1].count == 2


def test_min_count_parameter_raises_the_threshold() -> None:
    steps = [
        _step("A", line=1),
        _step("A", line=2),
        _step("B", line=3),
        _step("B", line=4),
        _step("B", line=5),
    ]
    clusters = cluster_exact(steps, min_count=3)
    assert [c.canonical_text for c in clusters] == ["B"]


def test_occurrence_files_counts_distinct_files() -> None:
    steps = [
        _step("same", line=1, file="a.feature"),
        _step("same", line=2, file="a.feature"),
        _step("same", line=3, file="b.feature"),
    ]
    clusters = cluster_exact(steps)
    assert clusters[0].occurrence_files == 2
    assert clusters[0].count == 3


# --- near-exact (Levenshtein) ---------------------------------------------


def test_near_exact_merges_parametric_variants() -> None:
    steps = [
        _step('the current account is "test1"', line=1),
        _step('the current account is "test1"', line=2),
        _step('the current account is "test2"', line=3),
        _step('the current account is "test3"', line=4),
    ]
    clusters = cluster_near_exact(steps, lev_threshold=0.85)
    assert len(clusters) == 1
    assert clusters[0].count == 4
    assert clusters[0].strategy == "near-exact"


def test_near_exact_keeps_disjoint_texts_separate() -> None:
    steps = [
        _step("I log in", line=1),
        _step("I log in", line=2),
        _step("the service is running", line=3),
        _step("the service is running", line=4),
    ]
    clusters = cluster_near_exact(steps, lev_threshold=0.85)
    assert len(clusters) == 2
    texts = {c.canonical_text for c in clusters}
    assert texts == {"I log in", "the service is running"}


def test_near_exact_threshold_controls_merging() -> None:
    steps = [
        _step("I log in", line=1),
        _step("I log in", line=2),
        _step("I log out", line=3),
        _step("I log out", line=4),
    ]
    # High threshold: disjoint (single-token change in 8-char text drops ratio).
    high = cluster_near_exact(steps, lev_threshold=0.95)
    assert len(high) == 2
    # Low threshold: merged.
    low = cluster_near_exact(steps, lev_threshold=0.70)
    assert len(low) == 1
    assert low[0].count == 4


def test_near_exact_canonical_is_most_frequent_variant() -> None:
    steps = [
        _step('account is "test1"', line=1),
        _step('account is "test1"', line=2),
        _step('account is "test1"', line=3),
        _step('account is "test2"', line=4),
    ]
    clusters = cluster_near_exact(steps, lev_threshold=0.85)
    assert len(clusters) == 1
    assert clusters[0].canonical_text == 'account is "test1"'
    assert clusters[0].count == 4


def test_near_exact_min_count_filter() -> None:
    steps = [
        _step("lonely phrase here", line=1),
        _step('account is "test1"', line=1),
        _step('account is "test1"', line=2),
        _step('account is "test2"', line=3),
    ]
    clusters = cluster_near_exact(steps, lev_threshold=0.85, min_count=3)
    assert len(clusters) == 1
    assert clusters[0].count == 3


def test_near_exact_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        cluster_near_exact([_step("x")], lev_threshold=1.0)
    with pytest.raises(ValueError):
        cluster_near_exact([_step("x")], lev_threshold=0.0)


def test_near_exact_handles_empty_input() -> None:
    assert cluster_near_exact([], lev_threshold=0.85) == []


# --- semantic / hybrid (fake SBERT via injected embed_fn) -----------------


def _fake_embed(text_to_vec: dict[str, list[float]]):
    """Build a fake embed_fn that returns L2-normalised vectors from a lookup."""

    def _embed(texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for t in texts:
            v = text_to_vec.get(t)
            if v is None:
                raise KeyError(f"unknown text in fake embedder: {t!r}")
            rows.append(v)
        arr = np.asarray(rows, dtype=np.float32)
        # L2-normalise so cosine = dot product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return (arr / np.where(norms == 0, 1, norms)).astype(np.float32)

    return _embed


def test_semantic_merges_semantically_close_but_lexically_disjoint_pairs() -> None:
    # Two distinct surface strings that should be ~identical under SBERT.
    steps = [
        _step('branch "X" now has type "Y"', line=1),
        _step('branch "X" now has type "Y"', line=2),
        _step('branch "Y" now has type "X"', line=3),
        _step('branch "Y" now has type "X"', line=4),
        _step("totally unrelated phrasing", line=5),
        _step("totally unrelated phrasing", line=6),
    ]
    # Build aligned vectors: first two texts point same way; third orthogonal.
    embed = _fake_embed(
        {
            'branch "X" now has type "Y"': [1.0, 0.0, 0.0],
            'branch "Y" now has type "X"': [0.99, 0.01, 0.0],  # cos ~0.99
            "totally unrelated phrasing": [0.0, 1.0, 0.0],
        }
    )
    clusters = cluster_semantic(steps, cos_threshold=0.95, embed_fn=embed)
    # The X/Y swap pair should merge (cos 0.99 >= 0.95)
    texts = {c.canonical_text for c in clusters}
    assert 'branch "X" now has type "Y"' in texts
    assert "totally unrelated phrasing" in texts
    assert len(clusters) == 2
    merged = next(c for c in clusters if c.strategy == "semantic" and c.count == 4)
    assert merged.count == 4


def test_hybrid_rejects_pairs_outside_the_lev_band() -> None:
    # Pair is semantically identical (fake cos = 1.0) but Lev would be very
    # low (completely different strings). Hybrid should REJECT — that's the
    # lev_min guard preventing lexically-disjoint false positives.
    steps = [
        _step("alpha", line=1),
        _step("alpha", line=2),
        _step("zzzzzzzz yyyyyyyy", line=3),
        _step("zzzzzzzz yyyyyyyy", line=4),
    ]
    embed = _fake_embed(
        {
            "alpha": [1.0, 0.0],
            "zzzzzzzz yyyyyyyy": [1.0, 0.0],  # cos 1.0 with "alpha"
        }
    )
    # Semantic (no Lev guard) would merge -> 1 cluster
    sem = cluster_semantic(steps, cos_threshold=0.9, embed_fn=embed)
    assert len(sem) == 1
    # Hybrid WITH lev_min=0.3 should NOT merge (Lev("alpha","zzzzzzzz yyyyyyyy") ~ 0)
    hyb = cluster_hybrid(steps, cos_threshold=0.9, lev_min=0.3, lev_max=0.95, embed_fn=embed)
    assert len(hyb) == 2


def test_hybrid_lev_upper_bound_excludes_near_exact_matches() -> None:
    # Two texts with Lev ratio > lev_max (0.95) should be excluded by the
    # hybrid band — they're near-exact, already caught by near-exact strategy.
    steps = [
        _step('the account is "test1"', line=1),
        _step('the account is "test1"', line=2),
        _step('the account is "test2"', line=3),  # Lev ~0.97 vs test1
        _step('the account is "test2"', line=4),
    ]
    embed = _fake_embed(
        {
            'the account is "test1"': [1.0, 0.0],
            'the account is "test2"': [1.0, 0.0],  # cos 1.0
        }
    )
    hyb = cluster_hybrid(steps, cos_threshold=0.9, lev_min=0.3, lev_max=0.95, embed_fn=embed)
    # Lev ratio between the two texts is ~0.96, above lev_max (0.95) -> no merge
    assert len(hyb) == 2


def test_hybrid_merges_when_both_conditions_met() -> None:
    # Cos is high AND Lev is in the band: merge.
    steps = [
        _step("I log in with valid credentials", line=1),
        _step("I log in with valid credentials", line=2),
        _step("I sign in using correct credentials", line=3),
        _step("I sign in using correct credentials", line=4),
    ]
    embed = _fake_embed(
        {
            "I log in with valid credentials": [1.0, 0.0],
            "I sign in using correct credentials": [0.95, 0.31],  # cos ~0.95
        }
    )
    hyb = cluster_hybrid(steps, cos_threshold=0.9, lev_min=0.3, lev_max=0.95, embed_fn=embed)
    assert len(hyb) == 1
    assert hyb[0].count == 4
    assert hyb[0].strategy == "hybrid"


def test_semantic_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        cluster_semantic([_step("x")], cos_threshold=1.0)


def test_hybrid_rejects_invalid_lev_band() -> None:
    with pytest.raises(ValueError):
        cluster_hybrid([_step("x")], cos_threshold=0.85, lev_min=0.9, lev_max=0.3)
