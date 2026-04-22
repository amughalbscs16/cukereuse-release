"""Re-fetch raw ``.feature`` files from the corpus pointers.

The Zenodo / GitHub release bundle is pointer-based: we publish repo+commit+path
triples so users can reconstruct the exact file content without us having to
redistribute copyleft-licensed material. This script performs the reconstruction.

Reads ``steps.parquet`` (or a JSONL) for (repo, commit_sha, file_path) triples,
deduplicates to the file level, and fetches each file via
``https://raw.githubusercontent.com/<repo>/<sha>/<path>`` into ``--out-dir``.

Respects GitHub's unauthenticated raw-file access (generous; no explicit rate
limit) but adds a configurable sleep between requests to stay polite.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

RAW_URL = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"


def _fetch(url: str, dest: Path, timeout: float) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cukereuse-rehydrate/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # https only
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, str(exc)[:120]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    ap.add_argument("--steps", type=Path, default=Path("corpus/steps.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("corpus/raw_rehydrated"))
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Sleep between fetches (seconds). Default 50ms ~= 20 req/s.",
    )
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    if not args.steps.exists():
        print(f"ERROR: {args.steps} not found.", file=sys.stderr)
        return 2

    df = pd.read_parquet(args.steps)
    required = {"repo", "commit_sha", "file_path"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing columns: {missing}", file=sys.stderr)
        return 2

    triples = (
        df[["repo", "commit_sha", "file_path"]].drop_duplicates().dropna().reset_index(drop=True)
    )
    if args.max_files:
        triples = triples.head(args.max_files)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_skip = 0
    n_fail = 0
    fail_log = args.out_dir.parent / "rehydrate_failures.txt"
    fail_lines: list[str] = []
    t0 = time.time()

    total = len(triples)
    print(f"Rehydrating {total:,} distinct .feature files to {args.out_dir}")
    for i, (repo, sha, rel) in enumerate(triples.itertuples(index=False), 1):
        if not repo or not sha or not rel:
            n_skip += 1
            continue
        dest = args.out_dir / str(repo).replace("/", "_") / str(rel)
        if dest.exists():
            n_skip += 1
            continue
        url = RAW_URL.format(repo=repo, sha=sha, path=str(rel).replace("\\", "/"))
        ok, err = _fetch(url, dest, args.timeout)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            fail_lines.append(f"{repo}\t{sha}\t{rel}\t{err}")
        if args.sleep > 0:
            time.sleep(args.sleep)
        if i % 500 == 0 or i == total:
            elapsed = time.time() - t0
            print(
                f"  [{i:>5}/{total}] ok={n_ok} skip={n_skip} fail={n_fail} "
                f"elapsed={elapsed:.0f}s"
            )

    if fail_lines:
        fail_log.write_text("\n".join(fail_lines), encoding="utf-8")
        print(f"  failures logged to {fail_log}")

    print()
    print(f"Done. fetched={n_ok}  skipped={n_skip}  failed={n_fail}")
    print(f"  out: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
