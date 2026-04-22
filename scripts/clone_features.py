"""Shallow + sparse-checkout clone ``.feature`` files from discovered repos.

Consumes the ``repos.csv`` emitted by ``mine_github.py`` and fetches only the
``*.feature`` files for each repo into ``--out-dir``. Uses git's partial-clone
(``--filter=blob:none``) and sparse-checkout features so we pull ~kilobytes
per repo instead of entire histories.

Key design choices:
- Skips repos whose clone target already exists (resumable).
- Parallel up to ``--workers`` (default 16); each subprocess is IO-bound.
- Pins the ``default_branch`` at clone time; we don't record commit SHA yet —
  that belongs to the build_corpus step which reads ``git rev-parse HEAD``.
- Emits a ``clone_manifest.jsonl`` alongside ``--out-dir`` with one row per
  repo: owner, name, status, n_feature_files, disk_kb, commit_sha.

Based on the scout's proven recipe (see probe/SCOUT_REPORT.md). Note the
two bugs we found during scouting: (1) subprocess stdin must be redirected
to /dev/null inside a `while read` loop; (2) `git sparse-checkout init` no
longer takes `--quiet` or `--no-cone` in modern git — use
``git sparse-checkout set --no-cone <patterns>`` directly.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WORKERS = 16
SPARSE_PATTERNS = ("**/*.feature",)


@dataclass(frozen=True)
class CloneTask:
    full_name: str
    clone_url: str
    default_branch: str


@dataclass
class CloneResult:
    full_name: str
    status: str  # ok / skip_existing / clone_failed / sparse_failed / zero_features
    n_feature_files: int = 0
    disk_kb: int = 0
    commit_sha: str = ""
    error: str = ""


def _repo_slug(full_name: str) -> str:
    return full_name.replace("/", "_")


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> tuple[int, str]:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,  # scout lesson: prevent stdin drain in loops
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        return -1, f"timeout after {exc.timeout}s"
    return result.returncode, (result.stderr or result.stdout or "").strip()


def _clone_one(task: CloneTask, out_root: Path) -> CloneResult:
    dest = out_root / _repo_slug(task.full_name)
    if dest.exists() and any(dest.rglob("*.feature")):
        return CloneResult(
            full_name=task.full_name,
            status="skip_existing",
            n_feature_files=sum(1 for _ in dest.rglob("*.feature")),
            disk_kb=_dir_kb(dest),
            commit_sha=_read_head(dest),
        )
    # Fresh start: nuke any incomplete attempt
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    # 1. shallow partial clone, no checkout yet
    rc, err = _run(
        [
            "git",
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--filter=blob:none",
            "--no-checkout",
            task.clone_url,
            str(dest),
        ]
    )
    if rc != 0:
        return CloneResult(full_name=task.full_name, status="clone_failed", error=err[:200])

    # 2. sparse-checkout to .feature files only (no --cone, no --quiet)
    rc, err = _run(["git", "sparse-checkout", "set", "--no-cone", *SPARSE_PATTERNS], cwd=dest)
    if rc != 0:
        return CloneResult(full_name=task.full_name, status="sparse_failed", error=err[:200])

    # 3. checkout (default_branch was detected during clone; HEAD is set)
    rc, err = _run(["git", "checkout", "--quiet"], cwd=dest)
    if rc != 0:
        return CloneResult(full_name=task.full_name, status="sparse_failed", error=err[:200])

    n = sum(1 for _ in dest.rglob("*.feature"))
    if n == 0:
        shutil.rmtree(dest, ignore_errors=True)
        return CloneResult(full_name=task.full_name, status="zero_features")

    return CloneResult(
        full_name=task.full_name,
        status="ok",
        n_feature_files=n,
        disk_kb=_dir_kb(dest),
        commit_sha=_read_head(dest),
    )


def _dir_kb(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) // 1024


def _read_head(p: Path) -> str:
    rc, out = _run(["git", "rev-parse", "HEAD"], cwd=p)
    return out.strip() if rc == 0 else ""


def _load_repos(repos_csv: Path, max_repos: int | None) -> list[CloneTask]:
    tasks: list[CloneTask] = []
    with repos_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("clone_url") or "").strip()
            full = (row.get("full_name") or "").strip()
            branch = (row.get("default_branch") or "").strip() or "main"
            if not url or not full:
                continue
            tasks.append(CloneTask(full_name=full, clone_url=url, default_branch=branch))
            if max_repos and len(tasks) >= max_repos:
                break
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    ap.add_argument("--repos", type=Path, default=Path("corpus/repos.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("corpus/raw"))
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--max-repos", type=int, default=None)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("corpus/clone_manifest.jsonl"),
        help="Per-repo result log (one JSON per line).",
    )
    args = ap.parse_args()

    if not args.repos.exists():
        print(f"ERROR: {args.repos} does not exist. Run mine_github.py first.", file=sys.stderr)
        return 2

    tasks = _load_repos(args.repos, args.max_repos)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Cloning .feature files from {len(tasks):,} repos -> {args.out_dir}")
    print(f"  workers={args.workers}  manifest={args.manifest}")

    results: list[CloneResult] = []
    with (
        cf.ThreadPoolExecutor(max_workers=args.workers) as pool,
        args.manifest.open("w", encoding="utf-8") as manifest_fp,
    ):
        futures = {pool.submit(_clone_one, t, args.out_dir): t for t in tasks}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            manifest_fp.write(
                json.dumps(
                    {
                        "full_name": res.full_name,
                        "status": res.status,
                        "n_feature_files": res.n_feature_files,
                        "disk_kb": res.disk_kb,
                        "commit_sha": res.commit_sha,
                        "error": res.error,
                    }
                )
                + "\n"
            )
            manifest_fp.flush()
            if i % 50 == 0 or i == len(tasks):
                good = sum(1 for r in results if r.status == "ok")
                skip = sum(1 for r in results if r.status == "skip_existing")
                fail = sum(1 for r in results if r.status.endswith("failed"))
                empty = sum(1 for r in results if r.status == "zero_features")
                print(
                    f"  [{i:>4}/{len(tasks)}] "
                    f"ok={good}  skip={skip}  fail={fail}  empty={empty}"
                )

    good = [r for r in results if r.status in {"ok", "skip_existing"}]
    total_files = sum(r.n_feature_files for r in good)
    total_kb = sum(r.disk_kb for r in good)
    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s. {len(good):,} repos with .feature files.")
    print(f"  total feature files: {total_files:,}")
    print(f"  total disk: {total_kb / 1024:.1f} MB")
    print(f"  manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
