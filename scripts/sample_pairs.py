# ruff: noqa: I001  # torch-before-pandas ordering deliberate
"""Sample 200 stratified pairs of step texts for ground-truth labeling.

Reads ``corpus/steps.parquet``, de-duplicates to unique normalized canonicals,
embeds them via cukereuse.similarity (hits the on-disk cache from Phase 2), and
emits a stratified sample across cosine-similarity bands so the resulting
labeled set covers the full decision surface of the hybrid strategy.

Output: ``corpus/unlabeled_pairs.jsonl`` — one JSON per line with
``{id, text_a, text_b, cos, lev, cos_band}``.
"""

from __future__ import annotations

import torch  # noqa: F401  # MUST import before pandas — WinError 1114 workaround

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from cukereuse.similarity import embed_texts, iter_high_similarity_pairs, lev_ratio, normalize

# Cosine bands (low inclusive, high exclusive except the last)
BANDS: list[tuple[float, float]] = [
    (0.50, 0.70),
    (0.70, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.95),
    (0.95, 1.00),
]


def _band_label(cos: float) -> str | None:
    for lo, hi in BANDS:
        if lo <= cos < hi or (hi == 1.00 and cos == 1.00):
            return f"[{lo:.2f},{hi:.2f})"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    ap.add_argument("--steps", type=Path, default=Path("corpus/steps.parquet"))
    ap.add_argument("--out", type=Path, default=Path("corpus/unlabeled_pairs.jsonl"))
    ap.add_argument(
        "--pool-size",
        type=int,
        default=20_000,
        help="Random sample of unique texts to embed for pair sampling.",
    )
    ap.add_argument(
        "--pairs-per-band",
        type=int,
        default=34,
        help="Target pairs per cosine band; total ~ pairs_per_band * len(BANDS).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.steps.exists():
        print(f"ERROR: {args.steps} not found.", file=sys.stderr)
        return 2

    print(f"Loading unique canonical texts from {args.steps} ...")
    df = pd.read_parquet(args.steps, columns=["text"])
    all_unique = sorted({normalize(t) for t in df["text"].tolist() if isinstance(t, str)})
    print(f"  {len(all_unique):,} unique normalized texts in corpus")

    pool = (
        random.sample(all_unique, args.pool_size) if len(all_unique) > args.pool_size else all_unique
    )
    print(f"  sampling pool: {len(pool):,} texts")

    print("Embedding pool (uses disk cache from Phase 2) ...")
    embs = embed_texts(pool, batch_size=256)
    print(f"  embeddings shape: {embs.shape}")

    rng = np.random.default_rng(args.seed)
    n = len(pool)

    # Collect high-similarity pairs (cos >= 0.50) directly by streaming the
    # cosine matrix. Random sampling misses this tail — only ~0.01% of random
    # pairs reach cos >= 0.80 in a pool of 20k diverse step texts.
    print("Scanning all pairs with cos >= 0.50 via iter_high_similarity_pairs ...")
    high_sim_pairs: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    n_high = 0
    for a, b, c in iter_high_similarity_pairs(embs, 0.50, chunk_size=1000):
        band = _band_label(float(c))
        if band is not None:
            high_sim_pairs[band].append((a, b, float(c)))
            n_high += 1
    print(f"  {n_high:,} pairs with cos >= 0.50 found")

    # Low-cos bands are rare among high-sim pairs; top them up with random
    # samples where cos is guaranteed to be low.
    print("Adding random low-cos pair candidates ...")
    i_idx = rng.integers(0, n, size=100_000)
    j_idx = rng.integers(0, n, size=100_000)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    a_idx = np.minimum(i_idx, j_idx)
    b_idx = np.maximum(i_idx, j_idx)
    cos_random = np.einsum("ij,ij->i", embs[a_idx], embs[b_idx]).astype(np.float32)
    low_band_target = "[0.50,0.70)"
    for k, c in enumerate(cos_random):
        if 0.50 <= float(c) < 0.70:
            high_sim_pairs[low_band_target].append((int(a_idx[k]), int(b_idx[k]), float(c)))

    print("Per-band candidate counts after augmentation:")
    for lo, hi in BANDS:
        label = f"[{lo:.2f},{hi:.2f})"
        print(f"  {label}: {len(high_sim_pairs.get(label, [])):,}")

    # Sample N per band, compute lev, emit
    picked_pairs: list[dict[str, object]] = []
    seen_pairs: set[tuple[int, int]] = set()

    for lo, hi in BANDS:
        label = f"[{lo:.2f},{hi:.2f})"
        candidates = high_sim_pairs.get(label, [])
        if not candidates:
            continue
        rng.shuffle(candidates)
        kept = 0
        for a, b, c in candidates:
            key = (min(a, b), max(a, b))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            text_a = pool[key[0]]
            text_b = pool[key[1]]
            lev = lev_ratio(text_a, text_b)
            picked_pairs.append(
                {
                    "id": len(picked_pairs),
                    "text_a": text_a,
                    "text_b": text_b,
                    "cos": round(c, 4),
                    "lev": round(lev, 4),
                    "cos_band": label,
                }
            )
            kept += 1
            if kept >= args.pairs_per_band:
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in picked_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(picked_pairs)} pairs to {args.out}")
    for lo, hi in BANDS:
        label = f"[{lo:.2f},{hi:.2f})"
        cnt = sum(1 for p in picked_pairs if p["cos_band"] == label)
        print(f"  {label}: {cnt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
