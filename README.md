# cukereuse: duplicate step detection for Cucumber and Gherkin

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-research_release-brightgreen.svg)](paper/main.pdf)

Static detector for duplicate and near-duplicate step text in `.feature` files. Works across Cucumber-JVM (Java), behave and pytest-bdd (Python), Cucumber-Ruby, cucumber-js, SpecFlow (.NET), and Behat (PHP). No test execution required. Runs on any repository.

Ships with:
- a CLI tool (`cukereuse`)
- a corpus of 1.11M Gherkin steps from 347 public GitHub repositories
- a 1,020-pair labelled calibration set with a written rubric
- a peer-review-ready research paper ([`paper/main.pdf`](paper/main.pdf))

## Who this is for

- **Maintainers of large BDD suites** who want to find and consolidate duplicate step text before it drifts further.
- **Platform and DevEx teams** who want a pre-commit hook that warns when a new step is the 200th parametric variant of an existing one.
- **Researchers** working on test-artefact quality, clone detection on near-natural-language text, or BDD-specific tooling.
- **Test leads** evaluating the duplication profile of an inherited codebase.

## What it does in one example

Given a `features/` directory, `cukereuse` finds that a phrasing like

    the response status is 200 OK

and its paraphrases (`status 200`, `the response status should be "200"`, `I should get a 200 response code`) appear together in tens of thousands of places. The HTML report groups them, shows every file and line, and names a canonical phrasing to consolidate toward.

On the paper's public corpus, the top cluster groups **20,737 occurrences across 2,245 files in 43 repositories**. The corpus-wide step-weighted duplication rate is **80.2%**; the median repository's rate is **58.6%**.

## Install

```bash
git clone https://github.com/amughalbscs16/cukereuse-release.git
cd cukereuse-release
uv sync
uv run cukereuse --help
```

Requires Python 3.10+. Torch is pulled from the CPU index so no GPU is needed.

## Quickstart

```bash
# Summary stats over a directory of feature files
cukereuse stats ./features

# Find exact duplicates (fastest, deterministic)
cukereuse find-duplicates ./features

# Paraphrase-aware duplicates (recommended for cluster reports)
cukereuse find-duplicates ./features --strategy hybrid

# Full analysis, HTML + JSON output
cukereuse analyze ./features --strategy hybrid --output report.html
```

Every run produces a browsable HTML report (inline CSS, collapsible cluster cards, file:line links) and a stable-schema JSON file for downstream pipelines.

## Strategy selection

| Task | Strategy | Threshold |
|------|----------|-----------|
| Byte-identical enumeration | `exact` (BLAKE2b hash) | n/a |
| Pair classification (is this pair a duplicate?) | `near-exact` (Levenshtein) | 0.80 |
| Semantic pair classification | `semantic` (SBERT cosine) | 0.82 |
| Project-wide paraphrase-aware cluster report | `hybrid` (cos + Lev band) | cos 0.82, Lev 0.30–0.95 |

Thresholds are calibrated against 1,020 labelled pairs under two evaluation protocols (primary rubric and a score-free second-pass relabelling). Full numbers, bootstrap confidence intervals, and the pair-vs-cluster threshold analysis are in Section 7 of the paper.

The paper's recommendation: **near-exact** is the strongest pair-level classifier (F₁ = 0.822 on score-free labels, F₁ = 0.862 on the primary rubric). **Hybrid** is the recommended strategy for building project-wide cluster reports because its Levenshtein band prevents transitive chaining that pure semantic clustering suffers from.

## What's in the corpus

The `corpus/` directory ships every analytical artefact as parquet or JSONL (46 MB combined):

| File | Content |
|------|---------|
| `steps.parquet` | 1,113,616 rows, one per step, with `repo`, `commit_sha`, `file_path`, `line`, `keyword`, `text`, `license_spdx`, `license_class`. |
| `clusters_exact.parquet` | 82,545 exact-duplicate clusters. |
| `clusters_hybrid.parquet` | 65,242 hybrid paraphrase-aware clusters. |
| `cluster_members_*.parquet` | Per-cluster membership. |
| `labeled_pairs.jsonl` | 1,020 labelled pairs with the fired rubric rule per pair. |
| `LABELING_RUBRIC.md` | The 10-rule written rubric the authors used. |
| `repos.csv`, `clone_manifest.jsonl` | Sample manifest with pinned commit SHAs. |

Raw `.feature`-file bodies are not redistributed (they inherit copyleft obligations from their source repositories). `rehydrate.py` fetches each file on demand from its upstream commit SHA:

```bash
uv run python scripts/rehydrate.py
```

## Reproducing the pipeline

```bash
# Mine GitHub for .feature repositories
uv run python scripts/mine_github.py --min-stars 10

# Sparse-clone only .feature files
uv run python scripts/clone_features.py --workers 16

# Parse into steps.parquet
uv run python scripts/build_corpus.py --workers 8

# Run the analysis
uv run python scripts/run_analysis.py --strategy hybrid

# Sample and label pairs
uv run python scripts/sample_pairs.py
uv run python scripts/write_labels.py

# Calibration analyses (baselines, CIs, score-free, licence chi-square, size scatter)
uv run python scripts/revision_analyses.py

# Regenerate paper figures
uv run python scripts/generate_figures.py
```

Mining is resumable. API responses cache under `scripts/mine_cache/`; re-runs skip completed work.

## Key numbers

These are the citable headline numbers from the paper.

- **Corpus:** 347 public GitHub repositories, 23,667 parsed `.feature` files, 1,113,616 Gherkin steps.
- **Step-weighted exact-duplicate rate:** 80.2%.
- **Median-repository duplication rate:** 58.6%.
- **Spearman ρ between repository size and duplication rate:** 0.508.
- **Top cluster (hybrid):** `the response status is 200 OK`, 20,737 occurrences across 2,245 files, 43 repositories.
- **Best pair-level detector:** near-exact (Levenshtein), F₁ = 0.822 under score-free evaluation, F₁ = 0.862 under the primary rubric.
- **Circularity cost (primary-vs-score-free inter-protocol Cohen's κ):** 0.47.
- **Lexical baselines (SourcererCC-style token Jaccard, NiCad-style TF-IDF char n-gram):** F₁ = 0.761 and 0.799.
- **Labelled benchmark:** 1,020 pairs manually labelled by the three authors (500 by Mughal, 300 by Fatima, 220 by Bilal) under a released written rubric. Inter-annotator Fleiss' κ = 0.84 on a 60-pair overlap subset.
- **CDN analysis of Gherkin:** 8 of 14 dimensions rated problematic or unsupported.

## Paper

The peer-review-ready paper is [`paper/main.pdf`](paper/main.pdf). LaTeX source and `references.bib` are in the same directory. Covers:

- Corpus construction and a size-vs-duplication analysis.
- Four detection strategies with bootstrap-CI calibration under two evaluation protocols.
- Two lexical baselines (SourcererCC-style, NiCad-style) on the same benchmark.
- A Cognitive Dimensions of Notations (CDN) analysis of Gherkin, to our knowledge the first in the peer-reviewed literature.
- Failure-mode breakdown (polarity flips, HTTP-verb mismatch, framework-keyword semantic shift).

## Citation

If you use the tool, the corpus, or the labelled benchmark, please cite:

```bibtex
@article{mughal2026cukereuse,
  author  = {Mughal, Ali Hassaan and Fatima, Noor and Bilal, Muhammad},
  title   = {Finding duplicates in 1.1{M} {BDD} steps: cukereuse, a paraphrase-robust static detector for {Cucumber} and {Gherkin}},
  year    = {2026},
  note    = {arXiv preprint and GitHub release},
  url     = {https://github.com/amughalbscs16/cukereuse-release}
}
```

`CITATION.cff` mirrors the same metadata for GitHub's citation widget.

## Licence

Apache-2.0 for the source code and analytical schema. See [LICENSE](LICENSE).

Corpus parquet rows carry a per-step `license_spdx` column so downstream work can filter by upstream licence class (`permissive`, `copyleft`, `unknown`, `unlicensed`). Raw `.feature` files are not redistributed here and remain under their source-repository licences; `rehydrate.py` fetches them on demand.

## Authors

- **Ali Hassaan Mughal**, Independent Researcher, Applied MBA (Data Analytics), Texas Wesleyan University. ORCID [0000-0002-0724-9197](https://orcid.org/0000-0002-0724-9197). `alihassaanmughal.work@gmail.com`.
- **Noor Fatima**, Independent Researcher, B.E. Computer Engineering, National University of Sciences and Technology (NUST), Pakistan. `noorfatima.ce@gmail.com`.
- **Muhammad Bilal**, Independent Researcher, M.Sc. Management, Technical University of Munich. ORCID [0000-0003-4106-0256](https://orcid.org/0000-0003-4106-0256). `m.bilal@tum.de`.
