"""Discover public GitHub repos containing .feature files.

Uses the Repo Search and Code Search REST endpoints via an already-authenticated
``gh`` CLI. Does NOT require Google Cloud / BigQuery — that was the plan's
optimal path but has a setup burden. This is the fallback that works with the
gh auth users already have.

Discovery strategy:
  1. Repo Search: all repos where Linguist classifies Gherkin as the primary
     language, stars >= ``--min-stars``. ~171 repos at stars>=10.
  2. Code Search: paginated across queries that combine a Gherkin-specific
     keyword with ``extension:feature`` — REST Code Search requires a keyword
     qualifier. Each query is capped at 1,000 results by the API.
  3. Dedupe by ``owner/name``.
  4. Fetch repo metadata for repos seen only via Code Search (licenses, stars,
     last_pushed, default_branch).
  5. Emit repos.csv with the columns build_corpus.py expects.

Output is resumable: intermediate API responses are cached to
``scripts/mine_cache/`` so re-runs skip completed work.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CACHE_DIR = Path("scripts/mine_cache")
DEFAULT_OUT = Path("corpus/repos.csv")
PERMISSIVE_SPDX = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "ISC",
    "Unlicense",
    "0BSD",
}
COPYLEFT_SPDX = {
    "GPL-2.0",
    "GPL-3.0",
    "LGPL-2.1",
    "LGPL-3.0",
    "AGPL-3.0",
    "MPL-2.0",
    "EPL-1.0",
    "EPL-2.0",
}

# --- rate-limit-aware gh api helpers --------------------------------------


def _gh_api_raw(endpoint: str) -> str:
    """Invoke ``gh api <endpoint>`` and return stdout. Raises on non-zero."""
    result = subprocess.run(
        ["gh", "api", endpoint],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
    )
    if result.returncode != 0:
        msg = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"gh api {endpoint} failed: {msg[:200]}")
    return result.stdout


def gh_api_json(endpoint: str, *, retries: int = 3) -> dict[str, Any]:
    """Call a GitHub API endpoint via gh and return parsed JSON.

    Retries once on rate-limit errors, sleeping until the reset.
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return json.loads(_gh_api_raw(endpoint))
        except RuntimeError as exc:
            last_err = exc
            msg = str(exc).lower()
            if "rate limit" in msg or "403" in msg:
                print(f"    rate-limited; sleeping 65s (attempt {attempt + 1}/{retries})")
                time.sleep(65)
                continue
            raise
    raise RuntimeError(f"gh api {endpoint} exhausted retries: {last_err}")


# --- cache helpers --------------------------------------------------------


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _load_cache_json(name: str) -> Any:
    path = _cache_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache_json(name: str, data: Any) -> None:
    path = _cache_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# --- discovery ------------------------------------------------------------


def discover_gherkin_primary(min_stars: int, min_pushed: str | None) -> list[dict[str, Any]]:
    """Repos where Gherkin is the predominant detected language."""
    cache_key = f"gherkin_primary_stars{min_stars}.json"
    cached = _load_cache_json(cache_key)
    if cached is not None:
        print(f"  [cache] gherkin-primary repos: {len(cached)}")
        return cached  # type: ignore[no-any-return]

    all_items: list[dict[str, Any]] = []
    filters = f"language:Gherkin+stars:%3E={min_stars}"
    if min_pushed:
        filters += f"+pushed:%3E={min_pushed}"
    # API cap: 10 pages of 100 for search endpoints.
    for page in range(1, 11):
        endpoint = (
            f"/search/repositories?q={filters}&sort=stars&order=desc" f"&per_page=100&page={page}"
        )
        res = gh_api_json(endpoint)
        items = res.get("items") or []
        if not items:
            break
        all_items.extend(items)
        total = int(res.get("total_count") or 0)
        print(
            f"    gherkin-primary page={page}: +{len(items)} (total so far {len(all_items)}/{total})"
        )
        if len(all_items) >= total:
            break
        time.sleep(2.1)  # Search API: 30 req/min

    _save_cache_json(cache_key, all_items)
    return all_items


def discover_via_code_search(keywords: list[str], min_pushed: str | None) -> set[str]:
    """Find repos containing .feature files via Code Search.

    Returns a set of ``owner/name`` strings. Each keyword + ``extension:feature``
    query paginates up to 1,000 results; we dedupe repos across all queries.
    """
    cache_key = "code_search_repos.json"
    cached = _load_cache_json(cache_key)
    if cached is not None and isinstance(cached, list):
        print(f"  [cache] code-search repos: {len(cached)}")
        return set(cached)

    repos: set[str] = set()
    for keyword in keywords:
        # URL-encoded keyword must quote spaces/colons.
        encoded = keyword.replace('"', "%22").replace(" ", "+").replace(":", "%3A")
        base = f"q={encoded}+extension%3Afeature"
        if min_pushed:
            base += f"+pushed%3A%3E%3D{min_pushed}"
        print(f"  code-search keyword: {keyword}")
        for page in range(1, 11):
            endpoint = f"/search/code?{base}&per_page=100&page={page}"
            try:
                res = gh_api_json(endpoint)
            except RuntimeError as exc:
                print(f"    page {page} error (continuing): {str(exc)[:100]}")
                break
            items = res.get("items") or []
            if not items:
                break
            before = len(repos)
            for item in items:
                repo = item.get("repository") or {}
                full_name = repo.get("full_name")
                if full_name:
                    repos.add(full_name)
            print(f"    page={page}: +{len(repos) - before} new (total {len(repos)})")
            time.sleep(6.1)  # Code Search: 10 req/min

    _save_cache_json(cache_key, sorted(repos))
    return repos


def fetch_repo_metadata(full_name: str) -> dict[str, Any] | None:
    """Get repo metadata — license, stars, dates — via /repos/{owner}/{name}."""
    safe = full_name.replace("/", "_").replace(":", "_")
    cache_key = f"metadata/{safe}.json"
    cached = _load_cache_json(cache_key)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    try:
        data = gh_api_json(f"/repos/{full_name}")
    except RuntimeError as exc:
        print(f"    metadata fetch failed for {full_name}: {str(exc)[:120]}")
        return None

    _save_cache_json(cache_key, data)
    return data


# --- csv emission --------------------------------------------------------


def license_class(spdx: str | None) -> str:
    if not spdx:
        return "unlicensed"
    if spdx in PERMISSIVE_SPDX:
        return "permissive"
    if spdx in COPYLEFT_SPDX:
        return "copyleft"
    if spdx == "NOASSERTION":
        return "unknown"
    return "unknown"


def write_repos_csv(repos: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "full_name",
                "clone_url",
                "default_branch",
                "stargazers_count",
                "last_pushed",
                "license_spdx",
                "license_class",
                "language",
                "archived",
            ]
        )
        for r in repos:
            lic = (r.get("license") or {}).get("spdx_id")
            w.writerow(
                [
                    r.get("full_name") or "",
                    r.get("clone_url") or "",
                    r.get("default_branch") or "",
                    int(r.get("stargazers_count") or 0),
                    (r.get("pushed_at") or "")[:10],
                    lic or "",
                    license_class(lic),
                    r.get("language") or "",
                    bool(r.get("archived")),
                ]
            )


# --- main ----------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    ap.add_argument("--min-stars", type=int, default=10)
    ap.add_argument(
        "--min-pushed",
        default=None,
        help="YYYY-MM-DD — only include repos pushed on/after this date.",
    )
    ap.add_argument(
        "--skip-code-search",
        action="store_true",
        help="Only use Repo Search (faster, but misses non-Gherkin-primary repos).",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print(f"[1/4] Discovering Gherkin-primary repos (stars >= {args.min_stars})...")
    primary = discover_gherkin_primary(args.min_stars, args.min_pushed)
    primary_by_name = {r["full_name"]: r for r in primary if r.get("full_name")}
    print(f"  -> {len(primary_by_name):,} repos from Repo Search")

    code_search_names: set[str] = set()
    if not args.skip_code_search:
        print("[2/4] Discovering via Code Search (slow — 10 req/min)...")
        code_search_names = discover_via_code_search(
            keywords=['"Feature:"', '"Scenario:"', '"Background:"'],
            min_pushed=args.min_pushed,
        )
        print(f"  -> {len(code_search_names):,} repos from Code Search (incl. overlap)")

    new_names = code_search_names - primary_by_name.keys()
    print(f"[3/4] Fetching metadata for {len(new_names):,} Code-Search-only repos...")
    code_search_metadata: list[dict[str, Any]] = []
    for idx, name in enumerate(sorted(new_names), 1):
        if idx % 25 == 0:
            print(f"  ...{idx}/{len(new_names)}")
        md = fetch_repo_metadata(name)
        if md:
            code_search_metadata.append(md)
        time.sleep(0.8)  # 5000/hr bucket = 75/min = ~0.8s/req

    combined = list(primary_by_name.values()) + code_search_metadata
    # Filter archived + unlicensed-with-zero-stars; keep the rest.
    filtered = [
        r
        for r in combined
        if not r.get("archived") and int(r.get("stargazers_count") or 0) >= args.min_stars
    ]
    print(f"[4/4] Writing {len(filtered):,} repos to {args.out}")
    write_repos_csv(filtered, args.out)

    classes: dict[str, int] = {}
    for r in filtered:
        classes[license_class((r.get("license") or {}).get("spdx_id"))] = (
            classes.get(license_class((r.get("license") or {}).get("spdx_id")), 0) + 1
        )
    print(f"  license mix: {classes}")
    print(
        f"  done: {args.out} (timestamp: {datetime.now(timezone.utc).isoformat(timespec='seconds')})"
    )


if __name__ == "__main__":
    sys.exit(main())
