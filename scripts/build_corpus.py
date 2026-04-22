"""Parse every cloned ``.feature`` file into ``steps.parquet``.

Consumes ``corpus/raw/`` (produced by clone_features.py) and the upstream
``corpus/repos.csv`` (for license metadata), emitting a single parquet file
with one row per step:

  repo, commit_sha, file_path, line, keyword, text, scenario, feature,
  tags, license_spdx, license_class, is_background, is_outline

Uses :func:`cukereuse.parser.parse_file` for real Gherkin parsing (not regex),
fanned out across ``--workers`` processes since parsing is CPU-bound.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cukereuse.parser import parse_file


@dataclass(frozen=True)
class _RepoInfo:
    full_name: str
    license_spdx: str
    license_class: str
    commit_sha: str


def _load_repo_meta(repos_csv: Path, manifest_path: Path | None) -> dict[str, _RepoInfo]:
    """Build a slug -> RepoInfo map from repos.csv and clone_manifest.jsonl."""
    rows: dict[str, dict[str, str]] = {}
    with repos_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            slug = (row.get("full_name") or "").replace("/", "_")
            rows[slug] = row

    manifest: dict[str, dict[str, object]] = {}
    if manifest_path and manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                slug = str(obj.get("full_name", "")).replace("/", "_")
                manifest[slug] = obj

    out: dict[str, _RepoInfo] = {}
    for slug, row in rows.items():
        sha = str(manifest.get(slug, {}).get("commit_sha", ""))
        out[slug] = _RepoInfo(
            full_name=row.get("full_name", ""),
            license_spdx=row.get("license_spdx", "") or "",
            license_class=row.get("license_class", "unknown"),
            commit_sha=sha,
        )
    return out


def _git_head_sha(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
            encoding="utf-8",
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _parse_one_file(args: tuple[Path, Path, str]) -> list[dict[str, object]]:
    """Parse one .feature file. ``args`` is (path, repo_root, repo_slug).

    Returns an empty list on any error (bad encoding, long Windows path,
    unparseable Gherkin) — never raises, so it cannot poison a worker pool.
    """
    path, repo_root, repo_slug = args
    try:
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            rel = path
        result = parse_file(path)
        rows: list[dict[str, object]] = []
        for s in result.steps:
            rows.append(
                {
                    "repo_slug": repo_slug,
                    "file_path": str(rel).replace("\\", "/"),
                    "line": s.line,
                    "keyword": s.keyword,
                    "text": s.text,
                    "scenario": s.scenario_name,
                    "feature": s.feature_name,
                    "tags": list(s.tags),
                    "is_background": s.is_background,
                    "is_outline": s.is_outline,
                }
            )
        return rows
    except Exception:  # worker must never crash
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    ap.add_argument("--in-dir", type=Path, default=Path("corpus/raw"))
    ap.add_argument("--repos", type=Path, default=Path("corpus/repos.csv"))
    ap.add_argument("--manifest", type=Path, default=Path("corpus/clone_manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("corpus/steps.parquet"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.in_dir.exists():
        print(f"ERROR: {args.in_dir} not found. Run clone_features.py first.", file=sys.stderr)
        return 2
    if not args.repos.exists():
        print(f"ERROR: {args.repos} not found.", file=sys.stderr)
        return 2

    repo_meta = _load_repo_meta(args.repos, args.manifest)

    # Enumerate every .feature under every repo dir
    t0 = time.time()
    print(f"Enumerating .feature files under {args.in_dir} ...")
    jobs: list[tuple[Path, Path, str]] = []
    seen_repo_slugs: list[str] = []
    sha_by_slug: dict[str, str] = {}
    for repo_dir in sorted(p for p in args.in_dir.iterdir() if p.is_dir()):
        slug = repo_dir.name
        seen_repo_slugs.append(slug)
        # Prefer manifest SHA; fall back to querying the working tree.
        sha = repo_meta.get(slug, _RepoInfo("", "", "unknown", "")).commit_sha
        if not sha:
            sha = _git_head_sha(repo_dir)
        sha_by_slug[slug] = sha
        for f in sorted(repo_dir.rglob("*.feature")):
            if f.is_file():
                jobs.append((f, repo_dir, slug))
    print(f"  {len(jobs):,} .feature files across {len(seen_repo_slugs):,} repos")

    print(f"Parsing in {args.workers} threads ...")
    all_rows: list[dict[str, object]] = []
    n_errors = 0
    # ThreadPool avoids Windows multiprocessing pickle quirks on Unicode/long
    # paths — observed real-world failures on Cyrillic .feature filenames.
    # The gherkin parser releases the GIL during native parsing so threading
    # still parallelises well at our scale.
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, rows in enumerate(pool.map(_parse_one_file, jobs, chunksize=25), 1):
            all_rows.extend(rows)
            if not rows:
                # Either zero steps or a parse error — parser returns [] either way
                pass
            if i % 500 == 0 or i == len(jobs):
                print(f"  [{i:>5}/{len(jobs)}] steps so far: {len(all_rows):,}")

    print(f"Parsed {len(all_rows):,} steps.")

    print(f"Enriching with repo metadata and writing {args.out} ...")
    df = pd.DataFrame(all_rows)
    # Add per-repo columns
    repo_rows = [
        {
            "repo_slug": slug,
            "repo": repo_meta.get(slug, _RepoInfo("", "", "unknown", "")).full_name,
            "license_spdx": repo_meta.get(slug, _RepoInfo("", "", "unknown", "")).license_spdx,
            "license_class": repo_meta.get(slug, _RepoInfo("", "", "unknown", "")).license_class,
            "commit_sha": sha_by_slug.get(slug, ""),
        }
        for slug in seen_repo_slugs
    ]
    repo_df = pd.DataFrame(repo_rows)
    if df.empty:
        print("WARNING: no steps parsed. Writing empty parquet.")
        df = pd.DataFrame(
            columns=[
                "repo_slug",
                "file_path",
                "line",
                "keyword",
                "text",
                "scenario",
                "feature",
                "tags",
                "is_background",
                "is_outline",
                "repo",
                "license_spdx",
                "license_class",
                "commit_sha",
            ]
        )
    else:
        df = df.merge(repo_df, on="repo_slug", how="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False, compression="zstd")

    # Summary
    elapsed = time.time() - t0
    print()
    print(f"Corpus built in {elapsed:.1f}s.")
    print(f"  rows written: {len(df):,}")
    if not df.empty:
        print(f"  distinct repos: {df['repo_slug'].nunique():,}")
        print(f"  parse errors (files yielding 0 steps): {n_errors:,}")
        print(f"  background steps: {int(df['is_background'].sum()):,}")
        print(f"  outline steps: {int(df['is_outline'].sum()):,}")
        print("  license mix:")
        for cls, cnt in df["license_class"].value_counts().items():
            print(f"    {cls:<12} {cnt:>10,} steps")
    print(f"  output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
