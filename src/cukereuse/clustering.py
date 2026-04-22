"""Group steps into duplicate clusters.

For now only the exact-match strategy is implemented. The ``Cluster`` type is
designed to carry future semantic-hybrid results without schema changes — the
``strategy`` field records how the cluster was produced so downstream code can
treat exact and near-duplicate clusters uniformly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from cukereuse.models import Step  # noqa: TCH001  # pydantic resolves at runtime
from cukereuse.similarity import (
    FloatArray,
    content_hash,
    embed_texts,
    iter_high_similarity_pairs,
    length_compatible,
    lev_ratio,
    normalize,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    EmbedFn = Callable[[list[str]], FloatArray]


ClusterStrategy = Literal["exact", "near-exact", "semantic", "hybrid"]


class Cluster(BaseModel):
    """A group of steps judged to be duplicates under some strategy."""

    model_config = ConfigDict(frozen=True)

    canonical_text: str
    members: tuple[Step, ...] = Field(default_factory=tuple)
    strategy: ClusterStrategy = "exact"

    @property
    def count(self) -> int:
        return len(self.members)

    @property
    def occurrence_files(self) -> int:
        return len({m.file_path for m in self.members})


def cluster_exact(steps: Iterable[Step], *, min_count: int = 2) -> list[Cluster]:
    """Group steps whose normalised text is byte-identical.

    Steps below ``min_count`` occurrences are dropped — singleton "clusters"
    aren't actionable consolidation targets.
    """
    buckets: dict[str, list[Step]] = defaultdict(list)
    canonical: dict[str, str] = {}
    for step in steps:
        h = content_hash(step.text)
        buckets[h].append(step)
        canonical.setdefault(h, normalize(step.text))

    clusters = [
        Cluster(canonical_text=canonical[h], members=tuple(members), strategy="exact")
        for h, members in buckets.items()
        if len(members) >= min_count
    ]
    clusters.sort(key=lambda c: (-c.count, c.canonical_text))
    return clusters


class _UnionFind:
    """Compact union-find for cluster merging over a known-size index set."""

    __slots__ = ("_parent", "_rank")

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


def cluster_near_exact(
    steps: Iterable[Step],
    *,
    lev_threshold: float = 0.85,
    min_count: int = 2,
) -> list[Cluster]:
    """Exact-cluster first, then merge clusters whose canonical texts are near-duplicates.

    Two-stage: (1) exact hash-dedupe to collapse byte-identical steps into
    proto-clusters including singletons; (2) all-pairs Levenshtein over the
    proto-cluster canonical texts with a length-compatibility prefilter, then
    union-find to merge above-threshold pairs.

    ``lev_threshold`` must be in (0.0, 1.0). Pairs at exactly 1.0 are already
    collapsed by the exact pass.
    """
    if not 0.0 < lev_threshold < 1.0:
        raise ValueError("lev_threshold must be in (0.0, 1.0)")

    # Stage 1: exact-cluster, retaining singletons so we can merge them.
    proto = cluster_exact(steps, min_count=1)
    if len(proto) <= 1:
        return [p for p in proto if p.count >= min_count]

    canonicals = [p.canonical_text for p in proto]
    n = len(canonicals)

    # Stage 2: pairwise Lev with length prefilter + union-find.
    uf = _UnionFind(n)
    for i in range(n):
        a = canonicals[i]
        for j in range(i + 1, n):
            b = canonicals[j]
            if not length_compatible(a, b, lev_threshold):
                continue
            if lev_ratio(a, b) >= lev_threshold:
                uf.union(i, j)

    # Merge proto-clusters along connected components.
    groups: dict[int, list[Cluster]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(proto[i])

    merged: list[Cluster] = []
    for parts in groups.values():
        total = sum(p.count for p in parts)
        if total < min_count:
            continue
        all_members = tuple(m for p in parts for m in p.members)
        # Canonical = the most-frequent proto-cluster's text. Break ties by
        # shortest text, then by sort order for determinism.
        biggest = min(
            parts,
            key=lambda p: (-p.count, len(p.canonical_text), p.canonical_text),
        )
        merged.append(
            Cluster(
                canonical_text=biggest.canonical_text,
                members=all_members,
                strategy="near-exact",
            )
        )

    merged.sort(key=lambda c: (-c.count, c.canonical_text))
    return merged


def _cluster_with_embedding(
    steps: Iterable[Step],
    *,
    strategy: Literal["semantic", "hybrid"],
    cos_threshold: float,
    lev_min: float,
    lev_max: float,
    min_count: int,
    embed_fn: EmbedFn | None,
) -> list[Cluster]:
    """Shared two-stage driver for semantic and hybrid strategies.

    Stage 1: exact hash-dedupe into proto-clusters (keeping singletons).
    Stage 2: embed proto-canonicals, find pairs with cosine >= threshold,
             (for hybrid) filter by Lev band, union-find merge.
    """
    if not 0.0 < cos_threshold < 1.0:
        raise ValueError("cos_threshold must be in (0.0, 1.0)")
    if strategy == "hybrid" and not 0.0 <= lev_min < lev_max <= 1.0:
        raise ValueError("require 0.0 <= lev_min < lev_max <= 1.0")

    proto = cluster_exact(steps, min_count=1)
    if len(proto) <= 1:
        return [p for p in proto if p.count >= min_count]

    canonicals = [p.canonical_text for p in proto]
    embed = embed_fn if embed_fn is not None else embed_texts
    embs = embed(canonicals)
    n = len(canonicals)

    # Streamed pair iteration keeps peak memory at chunk_size * n * 4 bytes
    # instead of the full n*n dense matrix. At n=200k+, the dense matrix is
    # unrepresentable on any reasonable machine.
    uf = _UnionFind(n)
    for i, j, _cos in iter_high_similarity_pairs(embs, cos_threshold):
        if strategy == "hybrid":
            r = lev_ratio(canonicals[i], canonicals[j])
            if not (lev_min <= r <= lev_max):
                continue
        uf.union(i, j)

    groups: dict[int, list[Cluster]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(proto[i])

    merged: list[Cluster] = []
    for parts in groups.values():
        total = sum(p.count for p in parts)
        if total < min_count:
            continue
        all_members = tuple(m for p in parts for m in p.members)
        biggest = min(
            parts,
            key=lambda p: (-p.count, len(p.canonical_text), p.canonical_text),
        )
        merged.append(
            Cluster(
                canonical_text=biggest.canonical_text,
                members=all_members,
                strategy=strategy,
            )
        )

    merged.sort(key=lambda c: (-c.count, c.canonical_text))
    return merged


def cluster_semantic(
    steps: Iterable[Step],
    *,
    cos_threshold: float = 0.90,
    min_count: int = 2,
    embed_fn: EmbedFn | None = None,
) -> list[Cluster]:
    """Pure SBERT clustering — single-linkage on cosine >= ``cos_threshold``.

    Ablation / upper-bound strategy; vulnerable to transitive chaining like
    Lev-only. Use :func:`cluster_hybrid` for production.
    """
    return _cluster_with_embedding(
        steps,
        strategy="semantic",
        cos_threshold=cos_threshold,
        lev_min=0.0,
        lev_max=1.0,
        min_count=min_count,
        embed_fn=embed_fn,
    )


def cluster_hybrid(
    steps: Iterable[Step],
    *,
    cos_threshold: float = 0.90,
    lev_min: float = 0.3,
    lev_max: float = 0.95,
    min_count: int = 2,
    embed_fn: EmbedFn | None = None,
) -> list[Cluster]:
    """Plan's hybrid strategy: cosine >= threshold AND Lev in [lev_min, lev_max].

    The Lev band is the critical anti-false-positive guard:
    - ``lev > lev_max`` (default 0.95): near-exact pairs already caught by
      the exact or near-exact passes; merging here would duplicate them.
    - ``lev < lev_min`` (default 0.3): semantically close but lexically
      disjoint pairs — often false positives or high-risk merges where the
      maintainer wouldn't actually want consolidation.
    """
    return _cluster_with_embedding(
        steps,
        strategy="hybrid",
        cos_threshold=cos_threshold,
        lev_min=lev_min,
        lev_max=lev_max,
        min_count=min_count,
        embed_fn=embed_fn,
    )
