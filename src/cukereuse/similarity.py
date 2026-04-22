"""Similarity primitives for step text.

Minimal surface for now — whitespace-collapsing normalisation and a stable
content hash. The semantic-embedding path (SBERT) will layer on top in a
later revision; the goal of this module is to own all text-level comparisons
so higher layers never branch on raw strings.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import os
import pickle
import re
from pathlib import Path  # used in helper type hints
from typing import TYPE_CHECKING

import Levenshtein
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

FloatArray = NDArray[np.float32]

_WS_RE = re.compile(r"\s+")

# Lazy model holder — the SBERT model is ~80 MB and takes ~10s to load, so we
# only instantiate on first use and keep a module-level reference afterwards.
_SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
_sbert_model: object | None = None


def normalize(text: str) -> str:
    """Collapse all internal whitespace runs and trim edges.

    No case-folding — Gherkin step phrasings are case-sensitive in practice
    (proper nouns, file paths, identifiers inside quotes).
    """
    return _WS_RE.sub(" ", text).strip()


def content_hash(text: str) -> str:
    """Deterministic 16-byte hash of the normalised text, as a hex string.

    Used as a dict key for exact-duplicate grouping. BLAKE2b with a truncated
    digest is faster than MD5 on CPython 3.11+ and avoids the weak-hash
    warnings flagged by some linters.
    """
    return hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).hexdigest()


def lev_ratio(a: str, b: str) -> float:
    """Normalised Levenshtein similarity: 0.0 (no match) to 1.0 (identical).

    Strings are :func:`normalize`-d first so whitespace doesn't count. This
    is the primary near-exact duplicate signal — it catches parameterised
    variations (``account is "test1"`` vs ``account is "test2"``) that the
    exact-hash pass treats as distinct.
    """
    return Levenshtein.ratio(normalize(a), normalize(b))


def length_compatible(a: str, b: str, threshold: float) -> bool:
    """Cheap prefilter: can ``lev_ratio(a, b)`` possibly reach ``threshold``?

    Levenshtein ratio is ``2 * matches / (len(a) + len(b))``; it's bounded
    above by ``2 * min(len_a, len_b) / (len_a + len_b)``. If that ceiling is
    already below the target threshold, the pair cannot qualify — skip it
    without paying for the full edit-distance computation.
    """
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return la == lb
    ceiling = 2 * min(la, lb) / (la + lb)
    return ceiling >= threshold


# --- SBERT embeddings ------------------------------------------------------


def _embedding_cache_path() -> Path:
    """Location of the on-disk embedding cache.

    Respects ``CUKEREUSE_CACHE_DIR`` for overrides (useful in CI / tests).
    """
    override = os.environ.get("CUKEREUSE_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".cache" / "cukereuse"
    base.mkdir(parents=True, exist_ok=True)
    return base / "embeddings.pkl.gz"


def _load_embedding_cache() -> dict[str, FloatArray]:
    path = _embedding_cache_path()
    if not path.exists():
        return {}
    with contextlib.suppress(Exception):
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)  # file we wrote ourselves
        if isinstance(data, dict):
            return data
    return {}


def _save_embedding_cache(cache: dict[str, FloatArray]) -> None:
    path = _embedding_cache_path()
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _get_sbert_model() -> object:
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer

        _sbert_model = SentenceTransformer(_SBERT_MODEL_NAME)
    return _sbert_model


def embed_texts(
    texts: Sequence[str],
    *,
    batch_size: int = 256,
    use_cache: bool = True,
) -> FloatArray:
    """Return ``(len(texts), 384)`` float32 L2-normalised SBERT embeddings.

    Deduplicates internally by hash so repeated inputs are embedded once.
    Embeddings are cached on disk keyed by :func:`content_hash` of the
    normalised text — subsequent runs on overlapping corpora skip the model
    entirely. Result rows are in input order.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    normed = [normalize(t) for t in texts]
    hashes = [content_hash(t) for t in normed]

    cache = _load_embedding_cache() if use_cache else {}

    # Collect unique missing texts (dedupe against cache AND within this batch).
    missing_order: list[str] = []
    seen: set[str] = set()
    for h in hashes:
        if h in cache or h in seen:
            continue
        seen.add(h)
        missing_order.append(h)

    if missing_order:
        # Map each hash to its first occurrence's normalised text
        first_by_hash = {h: normed[hashes.index(h)] for h in missing_order}
        miss_texts = [first_by_hash[h] for h in missing_order]
        model = _get_sbert_model()
        new_vecs = model.encode(  # type: ignore[attr-defined]
            miss_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        for h, v in zip(missing_order, new_vecs, strict=False):
            cache[h] = v
        if use_cache:
            _save_embedding_cache(cache)

    return np.stack([cache[h] for h in hashes])


def cosine_matrix(embeddings: FloatArray) -> FloatArray:
    """Pairwise cosine similarity assuming rows are already L2-normalised.

    Returns a dense (n, n) matrix. Safe for n up to a few thousand; for
    corpus-scale n use :func:`iter_high_similarity_pairs` instead.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2-D (n_texts, dim)")
    result: FloatArray = embeddings @ embeddings.T
    return result


def iter_high_similarity_pairs(
    embeddings: FloatArray,
    threshold: float,
    *,
    chunk_size: int = 1000,
) -> Iterator[tuple[int, int, float]]:
    """Yield ``(i, j, cos)`` triples for ``i < j`` with cosine ``>= threshold``.

    Chunks the LHS ``chunk_size`` rows at a time so peak memory stays at
    ``chunk_size * n * 4`` bytes instead of the full ``n * n * 4`` dense matrix.
    At n=220k this is the difference between 880 MB per block and 193 GB total.

    Embeddings must be L2-normalised — cosine reduces to dot product.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2-D (n_texts, dim)")
    n = embeddings.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = embeddings[start:end]
        sims = block @ embeddings.T  # (end-start, n)
        # Zero lower-triangle portion relative to the block to emit i<j only.
        # Easier to do per-row: mask out j<=global_i inline.
        for local in range(end - start):
            global_i = start + local
            row = sims[local]
            hits = np.where(row >= threshold)[0]
            for j in hits:
                jj = int(j)
                if jj > global_i:
                    yield global_i, jj, float(row[jj])
