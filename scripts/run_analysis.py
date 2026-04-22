# ruff: noqa: I001  # torch-before-pandas ordering is deliberate — see below
"""Run the cukereuse clustering pipeline on the parquet corpus.

Reads ``steps.parquet`` (emitted by build_corpus.py), groups steps into
duplicate clusters using the chosen strategy, and writes two outputs:

  clusters.parquet       one row per cluster with aggregate stats
  cluster_members.parquet one row per step-in-a-cluster, joinable back

Also emits a summary JSON compatible with the existing HTML/JSON reporter so
the corpus-scale results can be browsed the same way as scout-scale ones.
"""

from __future__ import annotations

# IMPORTANT: torch must import BEFORE pandas on Windows. If pandas loads its
# MKL runtime first, torch's c10.dll fails init with WinError 1114. We eager-
# import torch here even though it's indirectly pulled in by sentence_transformers,
# because ``import pandas`` on the next line would otherwise race ahead.
# See https://github.com/pytorch/pytorch/issues/95964 and related.
import torch  # noqa: F401  # MUST import before pandas (see comment above)

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from cukereuse.clustering import (
    cluster_exact,
    cluster_hybrid,
    cluster_near_exact,
    cluster_semantic,
)
from cukereuse.models import Step
from cukereuse.reporter import Report, write_html, write_json

if TYPE_CHECKING:
    from cukereuse.clustering import Cluster


def _rows_to_steps(df: pd.DataFrame) -> list[Step]:
    """Materialize Step records from the steps.parquet DataFrame."""
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


def _run_strategy(
    strategy: str,
    steps: list[Step],
    *,
    lev_threshold: float,
    cos_threshold: float,
    lev_min: float,
    lev_max: float,
) -> list[Cluster]:
    if strategy == "exact":
        return cluster_exact(steps)
    if strategy == "near-exact":
        return cluster_near_exact(steps, lev_threshold=lev_threshold)
    if strategy == "semantic":
        return cluster_semantic(steps, cos_threshold=cos_threshold)
    if strategy == "hybrid":
        return cluster_hybrid(
            steps,
            cos_threshold=cos_threshold,
            lev_min=lev_min,
            lev_max=lev_max,
        )
    raise ValueError(f"unknown strategy: {strategy}")


def _step_to_keys(step: Step) -> tuple[str, str, int]:
    """Decompose the compound file_path back into (repo_slug, rel_path, line)."""
    raw = str(step.file_path).replace("\\", "/")
    parts = raw.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1], step.line
    return raw, "", step.line


def _license_map(steps_df: pd.DataFrame) -> dict[str, str]:
    """Map repo_slug -> license_class from the steps.parquet data."""
    by_slug: dict[str, str] = {}
    for repo_slug, lic in steps_df[["repo_slug", "license_class"]].itertuples(index=False):
        by_slug.setdefault(repo_slug, lic)
    return by_slug


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    ap.add_argument("--steps", type=Path, default=Path("corpus/steps.parquet"))
    ap.add_argument("--out-clusters", type=Path, default=Path("corpus/clusters.parquet"))
    ap.add_argument(
        "--out-members",
        type=Path,
        default=Path("corpus/cluster_members.parquet"),
    )
    ap.add_argument("--out-html", type=Path, default=Path("analysis/report.html"))
    ap.add_argument("--out-json", type=Path, default=Path("analysis/report.json"))
    ap.add_argument(
        "--strategy",
        choices=["exact", "near-exact", "semantic", "hybrid"],
        default="hybrid",
    )
    ap.add_argument("--lev-threshold", type=float, default=0.92)
    ap.add_argument("--cos-threshold", type=float, default=0.90)
    ap.add_argument("--lev-min", type=float, default=0.3)
    ap.add_argument("--lev-max", type=float, default=0.95)
    args = ap.parse_args()

    if not args.steps.exists():
        print(f"ERROR: {args.steps} not found. Run build_corpus.py first.", file=sys.stderr)
        return 2

    # Preload SBERT before we blow up the process heap with 1M+ pydantic
    # Step instances. Observed failure mode: with 1.1M steps materialised,
    # torch's c10.dll init fails (WinError 1114). Loading the model first
    # reserves the DLL's address space before fragmentation kicks in.
    if args.strategy in {"semantic", "hybrid"}:
        from cukereuse.similarity import _get_sbert_model

        print("Pre-loading SBERT model ...")
        _get_sbert_model()

    print(f"Loading {args.steps} ...")
    t0 = time.time()
    steps_df = pd.read_parquet(args.steps)
    print(f"  {len(steps_df):,} steps across {steps_df['repo_slug'].nunique():,} repos")

    print("Materializing Step records ...")
    steps = _rows_to_steps(steps_df)
    print(f"  {len(steps):,} Step instances")

    print(f"Clustering (strategy={args.strategy}) ...")
    clusters = _run_strategy(
        args.strategy,
        steps,
        lev_threshold=args.lev_threshold,
        cos_threshold=args.cos_threshold,
        lev_min=args.lev_min,
        lev_max=args.lev_max,
    )
    print(f"  -> {len(clusters):,} duplicate clusters")

    # Aggregate cluster rows
    print("Building clusters.parquet / cluster_members.parquet ...")
    license_by_slug = _license_map(steps_df)
    cluster_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for cid, c in enumerate(clusters):
        classes = [license_by_slug.get(_step_to_keys(m)[0], "unknown") for m in c.members]
        class_counts = Counter(classes)
        top_class = class_counts.most_common(1)[0][0] if class_counts else "unknown"
        permissive_frac = class_counts.get("permissive", 0) / max(len(classes), 1)

        distinct_variants = sorted({m.text for m in c.members})
        member_examples = [v for v in distinct_variants if v != c.canonical_text][:5]
        n_repos = len({_step_to_keys(m)[0] for m in c.members})

        cluster_rows.append(
            {
                "cluster_id": cid,
                "canonical_text": c.canonical_text,
                "strategy": c.strategy,
                "count": c.count,
                "occurrence_files": c.occurrence_files,
                "n_distinct_repos": n_repos,
                "n_distinct_variants": len(distinct_variants),
                "member_examples": member_examples,
                "top_license_class": top_class,
                "permissive_fraction": round(permissive_frac, 4),
            }
        )
        for m in c.members:
            repo_slug, rel, line = _step_to_keys(m)
            member_rows.append(
                {
                    "cluster_id": cid,
                    "repo_slug": repo_slug,
                    "file_path": rel,
                    "line": line,
                    "keyword": m.keyword,
                    "text": m.text,
                    "scenario": m.scenario_name,
                    "is_background": m.is_background,
                    "is_outline": m.is_outline,
                }
            )

    args.out_clusters.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cluster_rows).to_parquet(args.out_clusters, index=False, compression="zstd")
    pd.DataFrame(member_rows).to_parquet(args.out_members, index=False, compression="zstd")

    # HTML/JSON via existing reporter for browsable results
    report = Report(
        root_path=args.steps.resolve(),
        strategy=args.strategy,
        n_feature_files=int(steps_df.drop_duplicates(["repo_slug", "file_path"]).shape[0]),
        n_parse_errors=0,  # reflected in build_corpus.py; we lose that here
        n_steps=len(steps),
        n_unique_step_texts=int(steps_df["text"].str.strip().nunique()),
        n_duplicate_clusters=len(clusters),
        clusters=tuple(clusters),
    )
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    write_html(report, args.out_html)
    write_json(report, args.out_json)

    # Summary
    dup_rate = report.exact_duplication_rate * 100
    summary = {
        "strategy": args.strategy,
        "n_steps": len(steps),
        "n_unique_step_texts": report.n_unique_step_texts,
        "n_duplicate_clusters": len(clusters),
        "exact_duplication_rate_pct": round(dup_rate, 2),
        "elapsed_seconds": round(time.time() - t0, 1),
        "top_5_clusters": [
            {
                "canonical_text": c.canonical_text,
                "count": c.count,
                "occurrence_files": c.occurrence_files,
            }
            for c in clusters[:5]
        ],
    }
    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print(f"clusters.parquet: {args.out_clusters}")
    print(f"cluster_members.parquet: {args.out_members}")
    print(f"HTML report: {args.out_html}")
    print(f"JSON report: {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
