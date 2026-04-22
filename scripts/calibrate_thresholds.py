"""Calibrate the near-exact / semantic / hybrid thresholds on labeled pairs.

Reads ``corpus/labeled_pairs.jsonl`` and sweeps thresholds for each strategy,
reporting precision, recall, F1 against the label column. Writes
``analysis/calibration.json`` with the full sweep table and the
knee-of-curve (best-F1) thresholds.

Each labeled pair row already has cos and lev scores from sample_pairs.py, so
we don't need to re-embed anything, calibration is a pure numerical exercise.
Labels are produced by the first author under the written rubric in
corpus/LABELING_RUBRIC.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _sweep_unary(
    pairs: list[dict[str, Any]],
    *,
    score_key: str,
    thresholds: list[float],
    strategy: str,
) -> list[dict[str, object]]:
    """Predict duplicate iff score_key >= threshold; sweep thresholds."""
    rows: list[dict[str, object]] = []
    for thr in thresholds:
        tp = fp = fn = tn = 0
        for p in pairs:
            pred = 1 if float(p[score_key]) >= thr else 0
            gold = int(p["label"])
            if pred == 1 and gold == 1:
                tp += 1
            elif pred == 1 and gold == 0:
                fp += 1
            elif pred == 0 and gold == 1:
                fn += 1
            else:
                tn += 1
        precision, recall, f1 = _metrics(tp, fp, fn)
        rows.append(
            {
                "strategy": strategy,
                "score": score_key,
                "threshold": thr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def _sweep_hybrid(
    pairs: list[dict[str, Any]],
    *,
    cos_thresholds: list[float],
    lev_min: float,
    lev_max: float,
) -> list[dict[str, object]]:
    """Predict duplicate iff cos >= cos_t AND lev_min <= lev <= lev_max."""
    rows: list[dict[str, object]] = []
    for cos_t in cos_thresholds:
        tp = fp = fn = tn = 0
        for p in pairs:
            cos = float(p["cos"])
            lev = float(p["lev"])
            pred = 1 if (cos >= cos_t and lev_min <= lev <= lev_max) else 0
            gold = int(p["label"])
            if pred == 1 and gold == 1:
                tp += 1
            elif pred == 1 and gold == 0:
                fp += 1
            elif pred == 0 and gold == 1:
                fn += 1
            else:
                tn += 1
        precision, recall, f1 = _metrics(tp, fp, fn)
        rows.append(
            {
                "strategy": "hybrid",
                "score": "cos_AND_lev-band",
                "threshold": cos_t,
                "lev_min": lev_min,
                "lev_max": lev_max,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def _best(sweep: list[dict[str, object]]) -> dict[str, object]:
    return max(sweep, key=lambda r: (float(r["f1"]), float(r["precision"])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    ap.add_argument("--labeled", type=Path, default=Path("corpus/labeled_pairs.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("analysis/calibration.json"))
    ap.add_argument("--lev-min", type=float, default=0.3)
    ap.add_argument("--lev-max", type=float, default=0.95)
    args = ap.parse_args()

    if not args.labeled.exists():
        print(f"ERROR: {args.labeled} not found.", file=sys.stderr)
        return 2

    pairs: list[dict[str, Any]] = []
    with args.labeled.open(encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} labeled pairs from {args.labeled}")
    n_dup = sum(1 for p in pairs if int(p["label"]) == 1)
    print(f"  duplicates: {n_dup}  not-duplicates: {len(pairs) - n_dup}")

    thresholds = [round(x * 0.01, 2) for x in range(50, 100)]

    near_exact = _sweep_unary(
        pairs, score_key="lev", thresholds=thresholds, strategy="near-exact"
    )
    semantic = _sweep_unary(
        pairs, score_key="cos", thresholds=thresholds, strategy="semantic"
    )
    hybrid = _sweep_hybrid(
        pairs, cos_thresholds=thresholds, lev_min=args.lev_min, lev_max=args.lev_max
    )

    best_ne = _best(near_exact)
    best_sem = _best(semantic)
    best_hy = _best(hybrid)

    print()
    print("=" * 80)
    print("KNEE-OF-CURVE (best-F1) PER STRATEGY")
    print("=" * 80)
    for label, row in [
        ("near-exact (lev only)", best_ne),
        ("semantic   (cos only)", best_sem),
        ("hybrid     (cos AND lev-band)", best_hy),
    ]:
        print(
            f"  {label:<32} threshold={row['threshold']:<6} "
            f"P={row['precision']} R={row['recall']} F1={row['f1']}  "
            f"[TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']}]"
        )

    # Full sweep tables
    print()
    print("NEAR-EXACT (lev threshold) — P/R/F1 by threshold")
    print(f"  {'thr':>5} {'P':>6} {'R':>6} {'F1':>6}")
    for row in near_exact:
        print(f"  {row['threshold']:>5.2f} {row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f}")

    print()
    print("SEMANTIC (cos threshold) — P/R/F1 by threshold")
    print(f"  {'thr':>5} {'P':>6} {'R':>6} {'F1':>6}")
    for row in semantic:
        print(f"  {row['threshold']:>5.2f} {row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f}")

    print()
    print(f"HYBRID (cos threshold; lev-band=[{args.lev_min},{args.lev_max}]) — P/R/F1")
    print(f"  {'cos_t':>5} {'P':>6} {'R':>6} {'F1':>6}")
    for row in hybrid:
        print(f"  {row['threshold']:>5.2f} {row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "meta": {
                    "n_pairs": len(pairs),
                    "n_duplicates": n_dup,
                    "labeler": pairs[0].get("labeler") if pairs else None,
                    "lev_band": [args.lev_min, args.lev_max],
                },
                "best": {
                    "near_exact": best_ne,
                    "semantic": best_sem,
                    "hybrid": best_hy,
                },
                "sweep": {
                    "near_exact": near_exact,
                    "semantic": semantic,
                    "hybrid": hybrid,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
