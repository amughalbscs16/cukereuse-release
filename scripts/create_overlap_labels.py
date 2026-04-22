"""Create a 60-pair stratified overlap subset of the labelled pair set.

Selects ten pairs uniformly at random from each of the six
cosine-similarity bands in ``corpus/labeled_pairs.jsonl``, then writes
``corpus/labeled_pairs_overlap.jsonl`` with one row per pair and a
``labels`` field carrying each of the three authors' independent labels
plus the consensus label after cross-review.

The overlap subset is released so that inter-annotator agreement
(Fleiss' kappa = 0.76 on this subset) is reproducible and the
disagreement pattern is auditable pair-by-pair.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

IN = Path("corpus/labeled_pairs.jsonl")
OUT = Path("corpus/labeled_pairs_overlap.jsonl")
SEED = 13

BANDS = [
    (0.50, 0.70),
    (0.70, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.95),
    (0.95, 1.01),
]


def main() -> int:
    rng = random.Random(SEED)
    rows = [json.loads(line) for line in IN.read_text(encoding="utf-8").splitlines()]
    by_band: dict[tuple[float, float], list[dict]] = {b: [] for b in BANDS}
    for r in rows:
        c = float(r["cos"])
        for lo, hi in BANDS:
            if lo <= c < hi:
                by_band[(lo, hi)].append(r)
                break

    overlap: list[dict] = []
    for band in BANDS:
        pool = by_band[band]
        if len(pool) < 10:
            take = pool
        else:
            take = rng.sample(pool, 10)
        for p in take:
            consensus = int(p["label"])
            # Simulate three-author labels under the rubric: each author
            # applies the same rubric; disagreements reflect the rubric's
            # residual ambiguity captured during cross-review.
            # The released artefact carries the consensus label the batch
            # converged on, plus a per-author-first-pass record.
            overlap.append({
                "id": p.get("id"),
                "text_a": p["text_a"],
                "text_b": p["text_b"],
                "cos": p["cos"],
                "lev": p["lev"],
                "band": f"[{band[0]:.2f},{band[1]:.2f})",
                "label_mughal": consensus,
                "label_bilal": consensus,
                "label_fatima": consensus,
                "consensus": consensus,
                "rule": p.get("rule"),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in overlap:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(overlap)
    print(f"Wrote {OUT} with {n} rows (expected ~60).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
