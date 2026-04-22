# Datasheet for the Cukereuse Corpus

Following the template from Gebru et al., *Datasheets for Datasets*, CACM 64(12), 2021.

**Version:** 0.1 (April 2026). Corresponds to the commit hash below at time of release.

---

## 1. Motivation

- **For what purpose was the dataset created?** To support empirical study of duplicate and near-duplicate step definitions in public Cucumber/Gherkin `.feature` files, and to enable evaluation of the `cukereuse` static duplicate detector.
- **Who created the dataset and who funds it?** Ali Hassaan Mughal (project lead, Texas Wesleyan University) and Muhammad Bilal (Technical University of Munich). No external funding.
- **Any other comments?** The corpus is published as a pointer-based release (no raw-file redistribution of copyleft material) plus a permissive-licensed showcase directory.

## 2. Composition

- **What do the instances represent?** Step-level records extracted from `.feature` files in public GitHub repositories. One row per Given/When/Then/And/But step.
- **How many instances are there?** **1,113,616 steps** across **23,667 parsed `.feature` files** (25,034 enumerated on disk; 1,367 are empty, tag-only, or fail the gherkin grammar) spanning **347 repositories** (v0.1, April 2026).
- **What data does each instance contain?**
  - `repo` — GitHub `owner/name`
  - `commit_sha` — pinned commit (40-char hex)
  - `file_path` — relative path of the `.feature` file inside the repo
  - `line` — line number of the step in its source file
  - `keyword` — `Given` / `When` / `Then` / `And` / `But`
  - `text` — the step phrasing (doc strings / data tables stripped)
  - `scenario` — enclosing scenario name (empty string for Background steps)
  - `feature` — enclosing feature name
  - `tags` — list of Gherkin tags (feature-level + scenario-level, merged)
  - `is_background` — boolean; the step belongs to a `Background:` block
  - `is_outline` — boolean; the step belongs to a `Scenario Outline:` block
  - `license_spdx` — SPDX identifier of the source repo's license (from GitHub's API)
  - `license_class` — `permissive` / `copyleft` / `unlicensed` / `unknown`
- **Is there a label or target?** `corpus/labeled_pairs.jsonl` contains **1,020 labeled pairs** (494 duplicates, 526 not-duplicates) stratified across cosine-similarity bands for threshold calibration. Labels were produced by the first author under a written rubric (`corpus/LABELING_RUBRIC.md`, ten ordered decision rules); each row carries a `labeler` field (`"author"`) and a `rule` field identifying which rubric clause fired. Single-annotator labels are a known threat to validity; independent-annotator replication is a stated direction for future work.
- **Are there recommended data splits?** None yet; all records are in a single parquet file. Splits per research use (train/test, by license class, by repo size) should be constructed by the consumer.
- **Are there errors, noise, or redundancies?** Expected:
  - **Duplication is the subject of study.** 80.2% of steps are exact-text duplicates of another step in the corpus — this is intentional and is the central empirical phenomenon.
  - Some Scenario Outline placeholder tokens (`<role>`, `<id>`) are preserved verbatim rather than unrolled.
  - Whitespace inside step text is preserved; only leading/trailing whitespace is trimmed.
  - Files that failed the gherkin-official parser were skipped silently (workers return empty row lists on any parse error). Observed causes: UTF-8 BOM at start of file (1C-CPM Russian-locale repos), unterminated doc strings.
- **Does the dataset contain confidential/offensive/sensitive content?** No PII is retained. Git commit-author metadata is stripped at the pointer level. A regex sweep for emails and IP addresses was applied to the permissive showcase directory (`corpus/examples/`) prior to release. Features harvested from public GitHub are subject to the original repo's license and the GitHub ToS.

### Coarse breakdown (v0.1)

| Axis | Count |
|---|---:|
| repositories (post-filter: stars ≥ 10, not archived) | 347 |
| `.feature` files (parsed into steps) | 23,667 |
| `.feature` files (enumerated on disk) | 25,034 |
| steps (total) | 1,113,616 |
| unique normalized step texts | 220,312 |
| Background steps | 61,214 |
| Scenario Outline steps | 66,020 |
| duplicate clusters (exact strategy) | 82,545 |
| clusters with ≥ 1000 occurrences | 86 |

### License class distribution (steps, not repos)

| class | steps | share |
|---|---:|---:|
| permissive   | 635,586 | 57.1% |
| copyleft     | 232,297 | 20.9% |
| unknown (NOASSERTION / detector failure) | 174,077 | 15.6% |
| unlicensed (no LICENSE file) | 71,656 |  6.4% |

**Showcase eligibility:** the permissive-licensed subset (57.1% of steps, ~200 repos) is redistributable in `corpus/examples/` with LICENSE + NOTICE preserved. Everything else is pointer-only via `scripts/rehydrate.py`.

## 3. Collection process

- **How was the data acquired?** Three-stage pipeline under `scripts/`:
  1. **Discovery** via GitHub's REST Search API (`gh` CLI auth), combining (a) `/search/repositories?q=language:Gherkin+stars:>=10` and (b) `/search/code?q="Feature:"+extension:feature` / `"Scenario:"` / `"Background:"`. We pivoted away from the plan's original GH Archive + BigQuery path because it requires Google Cloud billing setup; the REST fallback hits a 1,000-results-per-query cap but is sufficient at this scale. 1,333 distinct repos surfaced; 377 passed the stars/archived filters.
  2. **Shallow + sparse clone** via `git clone --depth 1 --filter=blob:none --no-checkout` followed by `git sparse-checkout set --no-cone '**/*.feature'`. Only blobs for matching paths are downloaded. Pinned commit SHAs are recorded in `corpus/clone_manifest.jsonl`. 368 repos successfully cloned, 725 MB disk (~2 KB/step).
  3. **Parsing** via the cukereuse wrapper around `gherkin-official` v29 (Cucumber's authoritative parser), fanned across a thread pool. Emits `corpus/steps.parquet` (zstd-compressed).
- **Over what timeframe?** Mining and parsing were run on 2026-04-19. The discovery queries capture GitHub's state on that day; pinned commit SHAs make the corpus content deterministic beyond that point.
- **Was there an ethical review?** Data is drawn exclusively from public GitHub repositories. License compatibility is tracked per repo. No deanonymisation attempts; no aggregation beyond the step level.

## 4. Preprocessing / cleaning / labeling

- **Was any preprocessing done?**
  - Text normalisation: whitespace runs are collapsed to single spaces and edges are trimmed (`cukereuse.similarity.normalize`). No case-folding — Gherkin is case-sensitive in practice.
  - Doc strings (`"""…"""`) and data tables (`| … |`) are treated as step arguments and excluded from the `text` column — they are step *arguments*, not step *phrasings*. The official gherkin AST exposes them on separate fields which we drop.
  - Scenario Outlines are NOT unrolled: the outline body with `<placeholder>` tokens appears as one row, rather than one row per Examples table row.
- **Is raw data saved?** Only pointers (`repo`, `commit_sha`, `file_path`) are saved long-term. Raw feature files can be reconstructed via `scripts/rehydrate.py` (fetches `https://raw.githubusercontent.com/<repo>/<sha>/<path>`). The permissive-licensed subset is also redistributed directly in `corpus/examples/`.

## 5. Uses

- **Tasks already supported by this corpus (v0.1):**
  - Threshold calibration for the cukereuse duplicate detector (see `probe/SCOUT_REPORT.md`).
  - Empirical evidence for the paper's headline finding: 80.2% of steps in a random sample of 347 BDD projects are exact-text duplicates of another step in the same corpus.
- **Other tasks the corpus could support:**
  - Empirical study of BDD practices across languages/ecosystems (Java/Cucumber-JVM vs Ruby/Cucumber vs PHP/Behat vs Python/behave).
  - Pretraining / evaluation of language models on test-specification text.
  - Research into automatic test refactoring, step-definition generation, tag taxonomies.
  - Cognitive-Dimensions-of-Notations analysis of BDD notations — each Section 4 dimension of the paper is grounded in a concrete example pulled from this corpus.
- **Is there anything that a dataset consumer could do that should NOT be done?**
  - Do not redistribute copyleft-licensed feature files as raw content without complying with source terms. Use the pointers + `rehydrate.py` for any content outside the permissive showcase.
  - Do not use the pointers to violate the source repos' ToS (rate limits, scraping, etc.).

## 6. Distribution

- **Release channels:** GitHub repository at [amughalbscs16/cukereuse](https://github.com/amughalbscs16/cukereuse) (Apache-2.0 for source code and schema). The release bundle includes: `repos.csv`, `clone_manifest.jsonl`, `steps.parquet`, `clusters_exact.parquet`, `clusters_hybrid.parquet`, `cluster_members_exact.parquet`, `cluster_members_hybrid.parquet`, `labeled_pairs.jsonl`, `LABELING_RUBRIC.md`, this DATA_CARD, and a README pointing at rehydration.
- **Restrictions:** The parquet files contain analytical metadata (canonical step text, SPDX class per row). Verbatim raw `.feature` file bodies are not redistributed; `rehydrate.py` fetches each original file from its upstream repository on demand at the pinned commit SHA, preserving the source licence's obligations.

## 7. Maintenance

- **Who maintains it?** The authors of the `cukereuse` repository (Ali Hassaan Mughal and Muhammad Bilal).
- **Errata?** Filed as GitHub issues on [amughalbscs16/cukereuse](https://github.com/amughalbscs16/cukereuse).
- **Will the dataset be updated?** A frozen snapshot corresponds to each paper version, tagged as a GitHub release (`v0.1`, `v0.2`, …). Supersession of earlier versions is documented in the release notes.
