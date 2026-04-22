# ruff: noqa: I001
"""License-stratified duplication analysis on corpus/steps.parquet.

Groups the 1.1M-step corpus by ``license_class`` (permissive, copyleft,
unknown, unlicensed), runs exact clustering on each group independently,
and reports per-group: total steps, unique step texts, exact-duplication
rate, cluster count, and the top-5 canonical phrasings by count.

Answers the reviewer question: "does the 80.2% headline duplication rate
hold uniformly across licence classes, or is it an artefact of the
dominant (permissive) stratum?"
"""

from __future__ import annotations

import torch  # noqa: F401  # MUST import before pandas on Windows (WinError 1114)

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from cukereuse.clustering import cluster_exact
from cukereuse.models import Step


def _rows_to_steps(df: pd.DataFrame) -> list[Step]:
    steps: list[Step] = []
    for row in df.itertuples(index=False):
        tags = tuple(row.tags) if row.tags is not None else ()
        steps.append(
            Step(
                keyword=str(row.keyword),
                text=str(row.text),
                file_path=Path(f"{row.repo_slug}/{row.file_path}"),
                line=int(row.line),
                scenario_name=str(row.scenario or ""),
                feature_name=str(row.feature or ""),
                tags=tags,
                is_background=bool(row.is_background),
                is_outline=bool(row.is_outline),
            )
        )
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    ap.add_argument("--steps", type=Path, default=Path("corpus/steps.parquet"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/license_stratified.json"),
    )
    args = ap.parse_args()

    if not args.steps.exists():
        print(f"ERROR: {args.steps} not found.", file=sys.stderr)
        return 2

    print(f"Loading {args.steps} ...")
    df = pd.read_parquet(args.steps)
    print(f"  {len(df):,} total steps across {df['repo_slug'].nunique():,} repos")

    classes = sorted(df["license_class"].dropna().unique().tolist())
    print(f"  license classes: {classes}")

    overall = {
        "n_steps": int(len(df)),
        "n_repos": int(df["repo_slug"].nunique()),
        "n_unique_texts": int(df["text"].str.strip().nunique()),
    }
    overall["exact_dup_rate"] = round(
        1 - overall["n_unique_texts"] / max(overall["n_steps"], 1), 4
    )

    per_class: dict[str, dict[str, object]] = {}
    for cls in classes:
        sub = df[df["license_class"] == cls]
        if len(sub) == 0:
            continue
        print(f"\n--- license_class = {cls} ---")
        n = len(sub)
        repos = int(sub["repo_slug"].nunique())
        uniq = int(sub["text"].str.strip().nunique())
        rate = round(1 - uniq / max(n, 1), 4)
        print(f"  steps: {n:,}  repos: {repos}  unique: {uniq:,}  exact-dup rate: {rate*100:.2f}%")

        print(f"  clustering ({n:,} steps) ...")
        steps = _rows_to_steps(sub)
        clusters = cluster_exact(steps)
        total_in_clusters = sum(c.count for c in clusters)
        print(f"  clusters: {len(clusters):,}")
        print(f"  steps in clusters: {total_in_clusters:,}  ({total_in_clusters / n * 100:.1f}%)")

        top = [
            {"canonical_text": c.canonical_text, "count": c.count, "files": c.occurrence_files}
            for c in clusters[:5]
        ]
        print("  top 5 clusters:")
        for t in top:
            txt = t["canonical_text"]
            if len(txt) > 60:
                txt = txt[:57] + "..."
            print(f"    {t['count']:>6}x  {txt}")

        per_class[cls] = {
            "n_steps": n,
            "n_repos": repos,
            "n_unique_texts": uniq,
            "exact_dup_rate": rate,
            "n_clusters": len(clusters),
            "steps_in_clusters": total_in_clusters,
            "top_5_clusters": top,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"overall": overall, "per_class": per_class}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
