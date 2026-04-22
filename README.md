# cukereuse

**Static, paraphrase-robust duplicate step detection for Cucumber/Gherkin `.feature` files.**

Detects duplicate and near-duplicate step definitions using a layered pipeline — exact hash → normalised Levenshtein → sentence-transformer (SBERT `all-MiniLM-L6-v2`) embeddings → hybrid cluster collapse. No test execution required; works on any repository in any language binding (Java/Cucumber-JVM, Python/behave/pytest-bdd, Ruby/Cucumber-Ruby, JS/cucumber-js, .NET/SpecFlow, Behat/PHP).

Released alongside an empirical corpus of 347 public GitHub repositories, approximately 23.7k parsed `.feature` files and 1.11M Gherkin steps, a 1,020-pair labelled calibration set with a written rubric, and a Cognitive Dimensions of Notations analysis of Gherkin. The full empirical argument is in [`paper/main.pdf`](paper/main.pdf).

## Install

```bash
git clone https://github.com/amughalbscs16/cukereuse.git
cd cukereuse
uv sync                  # uses pyproject.toml; torch is pulled from the CPU index
uv run cukereuse --help
```

## CLI

```bash
# Summary statistics over a directory of feature files
cukereuse stats ./features

# Detect duplicates — four strategies
cukereuse find-duplicates ./features                                    # default: exact hash
cukereuse find-duplicates ./features --strategy near-exact              # + Levenshtein
cukereuse find-duplicates ./features --strategy semantic                # + SBERT
cukereuse find-duplicates ./features --strategy hybrid                  # cos AND Lev-band

# Run the full pipeline (analysis + top clusters)
cukereuse analyze ./features --strategy hybrid --output results.json
```

Every invocation writes an HTML report with inline CSS, collapsible cluster cards, and file:line links, plus a stable-schema JSON sibling for downstream tooling.

## Strategies and when to use each

| Operation | Recommended strategy | Default threshold |
|-----------|---------------------|-------------------|
| Exact-duplicate enumeration        | `exact` (BLAKE2b)     | — |
| Pair-level duplicate classification| `near-exact` (Lev)    | 0.80 (primary) / 0.71 (score-free) |
| Paraphrase-aware cluster collapse  | `hybrid` (cos ∧ Lev-band) | cos 0.82, Lev ∈ [0.3, 0.95] |
| Semantic pair classification       | `semantic` (SBERT)    | cos 0.82 |

Thresholds are calibrated against 1,020 labelled pairs under two independent evaluation protocols (primary rubric and a score-free second-pass relabelling). Calibration tables, bootstrap 95% confidence intervals, and the pair-vs-cluster threshold discussion are in §7 of the paper.

## Corpus

The `corpus/` directory contains the full mined dataset as parquet files (~46 MB combined):

| File | Content |
|------|---------|
| `steps.parquet`                    | 1,113,616 rows, one per step; columns include `repo`, `commit_sha`, `file_path`, `line`, `keyword`, `text`, `license_spdx`, `license_class`. |
| `clusters_exact.parquet`           | 82,545 exact-duplicate clusters. |
| `clusters_hybrid.parquet`          | 65,242 hybrid paraphrase-aware clusters. |
| `cluster_members_*.parquet`        | Per-cluster membership (which steps belong to which cluster). |
| `labeled_pairs.jsonl`              | 1,020 labelled pairs released with per-pair rule trail. |
| `LABELING_RUBRIC.md`               | The written rubric: 10 ordered decision rules. |
| `repos.csv`, `clone_manifest.jsonl`| Sample manifest with pinned commit SHAs. |

The verbatim raw `.feature`-file bodies are **not redistributed** here (approximately 418 MB of content inheriting copyleft obligations from their source repositories). `scripts/rehydrate.py` reconstructs them on demand from the pinned commit SHAs:

```bash
uv run python scripts/rehydrate.py
```

## Reproducing the corpus pipeline

```bash
# 1. Discover repositories with .feature files via GitHub REST Search
uv run python scripts/mine_github.py --min-stars 10

# 2. Shallow + sparse clone the .feature files only
uv run python scripts/clone_features.py --workers 16

# 3. Parse every feature into steps.parquet
uv run python scripts/build_corpus.py --workers 8

# 4. Run the chosen strategy, emit clusters.parquet + HTML/JSON reports
uv run python scripts/run_analysis.py --strategy hybrid

# 5. Sample and label pairs for calibration
uv run python scripts/sample_pairs.py
uv run python scripts/write_labels.py

# 6. Reproduce the revision analyses (baselines, bootstrap CIs, score-free relabel,
#    licence chi-square, size-vs-duplication scatter)
uv run python scripts/revision_analyses.py

# 7. Regenerate all paper figures
uv run python scripts/generate_figures.py
```

Mining is resumable — API responses are cached under `scripts/mine_cache/`. Re-runs skip completed work.

## Paper

The full research argument, including the Cognitive Dimensions of Notations analysis of Gherkin, the dual-protocol evaluation methodology, and the repository-size confound discussion, is in [`paper/main.pdf`](paper/main.pdf). The LaTeX source and `references.bib` are in the same directory.

## Positioning

The paper is **orthogonal to** execution-based scenario-duplicate detection (Binamungu et al., Manchester, 2018-2023) — different analysis level (step-text vocabulary clustering versus scenario-level behavioural equivalence). A production BDD toolchain could productively deploy both. The corpus is, to our knowledge, the largest released cross-organisational BDD corpus; the 1,020-pair calibration set and rubric are released alongside for independent replication.

## Licence

Apache-2.0 for source code and the analytical schema. See [LICENSE](LICENSE).

The corpus parquet rows carry a `license_spdx` column per step so that downstream users can filter by upstream licence class (`permissive`, `copyleft`, `unknown`, `unlicensed`). The verbatim raw `.feature` files are not redistributed here; their licences are preserved at each source repository and `rehydrate.py` fetches from that source.

## Citation

See [CITATION.cff](CITATION.cff).
