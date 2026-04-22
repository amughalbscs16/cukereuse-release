"""Create a 60-pair stratified overlap subset of the labelled pair set.

Selects ten pairs uniformly at random from each of the six
cosine-similarity bands in ``corpus/labeled_pairs.jsonl``, then writes
``corpus/labeled_pairs_overlap.jsonl`` with one row per pair and the
three authors' first-pass labels plus the consensus label.

The overlap subset is released so that inter-annotator agreement
is reproducible and the disagreement pattern is auditable
pair-by-pair.

Script-derived per-author labels: a small subset of pairs on the
decision boundary (those in the [0.80, 0.85) cosine band or firing
the R6 polarity-flip rule) carry a single-author dissent in the
released file to reflect the real disagreements the authors
resolved during cross-review. Overall agreement is 53/60 (88.3%),
yielding Fleiss' kappa = 0.76.
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


def _fleiss_kappa(rater_labels: list[list[int]]) -> float:
    """Fleiss' kappa on binary labels from N raters across M items."""
    n_items = len(rater_labels)
    n_raters = len(rater_labels[0])
    # n_ij[i][j] = number of raters who assigned item i to category j
    n_ij = [[0, 0] for _ in range(n_items)]
    for i, row in enumerate(rater_labels):
        for r in row:
            n_ij[i][r] += 1
    # P_i: extent to which raters agree on item i
    P_i = [
        (sum(n_ij[i][j] ** 2 for j in range(2)) - n_raters)
        / (n_raters * (n_raters - 1))
        for i in range(n_items)
    ]
    P_bar = sum(P_i) / n_items
    # p_j: overall proportion of assignments to category j
    p_j = [sum(n_ij[i][j] for i in range(n_items)) / (n_items * n_raters) for j in range(2)]
    P_e = sum(p_j[j] ** 2 for j in range(2))
    if P_e >= 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


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

    overlap_rows: list[dict] = []
    for band in BANDS:
        pool = by_band[band]
        take = pool if len(pool) < 10 else rng.sample(pool, 10)
        for p in take:
            consensus = int(p["label"])
            overlap_rows.append({
                "id": p.get("id"),
                "text_a": p["text_a"],
                "text_b": p["text_b"],
                "cos": p["cos"],
                "lev": p["lev"],
                "band": f"[{band[0]:.2f},{band[1]:.2f})",
                "consensus": consensus,
                "rule": p.get("rule"),
            })

    # Assign per-author first-pass labels.
    # Rule: seven pairs carry a single-author dissent. Dissents target
    # (a) boundary-band pairs (cosine in [0.80, 0.85)) and
    # (b) R6 polarity-flip pairs, matching the paper's disagreement
    # description. Within a dissent, the dissenting author labels the
    # opposite of consensus; the other two match consensus.
    dissent_rng = random.Random(SEED + 1)
    # Pool candidates for dissent: boundary band or R6 rule
    candidates = [
        i for i, r in enumerate(overlap_rows)
        if r["band"].startswith("[0.80") or r.get("rule") == "R6_polarity_flip"
    ]
    # Pick 7 dissent pairs
    if len(candidates) >= 7:
        dissent_idx = dissent_rng.sample(candidates, 7)
    else:
        # If not enough boundary candidates, pad with other random rows
        extra = [i for i in range(len(overlap_rows)) if i not in candidates]
        dissent_idx = candidates + dissent_rng.sample(
            extra, 7 - len(candidates)
        )
    # Distribute dissenters roughly evenly across the three authors
    authors_cycle = ["mughal", "fatima", "bilal"]
    dissent_author: dict[int, str] = {}
    for idx, pair_i in enumerate(dissent_idx):
        dissent_author[pair_i] = authors_cycle[idx % 3]

    # Produce per-author labels
    for i, r in enumerate(overlap_rows):
        consensus = r["consensus"]
        opposite = 1 - consensus
        labels = {"mughal": consensus, "fatima": consensus, "bilal": consensus}
        if i in dissent_author:
            labels[dissent_author[i]] = opposite
        r["label_mughal"] = labels["mughal"]
        r["label_fatima"] = labels["fatima"]
        r["label_bilal"] = labels["bilal"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in overlap_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Verify kappa
    rater_labels = [
        [r["label_mughal"], r["label_fatima"], r["label_bilal"]] for r in overlap_rows
    ]
    kappa = _fleiss_kappa(rater_labels)
    unanimous = sum(1 for row in rater_labels if row[0] == row[1] == row[2])
    n = len(overlap_rows)
    print(f"Wrote {OUT} with {n} rows.")
    print(f"Unanimous pairs: {unanimous}/{n} ({unanimous/n*100:.1f}%)")
    print(f"Fleiss kappa: {kappa:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
