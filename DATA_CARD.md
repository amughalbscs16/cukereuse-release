# Datasheet for the cukereuse corpus

Follows the template from Gebru et al., *Datasheets for Datasets*, CACM 64(12), 2021.

**Version:** 0.1 (April 2026). Corresponds to the repository commit at release.

---

## 1. Motivation

- **For what purpose was the dataset created?** To enable empirical study of duplicate and near-duplicate step text in public Cucumber/Gherkin `.feature` files, and to calibrate and evaluate the `cukereuse` static duplicate detector.
- **Who created the dataset and who funds it?** Ali Hassaan Mughal (Texas Wesleyan University), Noor Fatima (National University of Sciences and Technology, Pakistan), and Muhammad Bilal (Technical University of Munich). No external funding.
- **Any other comments?** The corpus is a pointer-based release. Derived analytical metadata (parquet, JSONL) is redistributed under Apache-2.0. Raw `.feature`-file bodies are not redistributed; they are reconstructed on demand from pinned commit SHAs and remain under each source repository's licence.

## 2. Composition

- **What do the instances represent?** Step-level records extracted from `.feature` files in public GitHub repositories. One row per Given/When/Then/And/But step.
- **How many instances are there?** 1,113,616 steps across 23,667 parsed `.feature` files (25,034 enumerated on disk; 1,367 are empty, tag-only, or fail the gherkin grammar) spanning 347 repositories.
- **What data does each instance contain?**
  - `repo`, `commit_sha` (40-char hex), `file_path`, `line`.
  - `keyword` (`Given` / `When` / `Then` / `And` / `But`).
  - `text` (the step phrasing; doc strings and data tables are stripped).
  - `scenario`, `feature` (enclosing names).
  - `tags` (feature-level plus scenario-level tag list, merged).
  - `is_background`, `is_outline` (booleans).
  - `license_spdx` (SPDX identifier from GitHub's licence endpoint).
  - `license_class` (`permissive` / `copyleft` / `unknown` / `unlicensed`).
- **Is there a label or target?** `corpus/labeled_pairs.jsonl` contains 1,020 labelled pairs (494 duplicates, 526 not-duplicates) stratified across six cosine-similarity bands. Labels were produced manually by the three authors (500 pairs by Ali Hassaan Mughal, 300 by Noor Fatima, 220 by Muhammad Bilal) under a shared written rubric in `corpus/LABELING_RUBRIC.md` (ten ordered decision rules). Every row carries a `labeler` field and a `rule` field recording which rubric clause fired, so rubric application is auditable pair by pair. All three authors cross-reviewed boundary cases after each 200-pair batch to converge on consistent R4-R8 application. A 60-pair stratified overlap subset was independently labelled by all three authors before the main batch, yielding Fleiss' $\kappa = 0.84$ (almost perfect agreement on the Landis-Koch scale).
- **Are there recommended data splits?** No built-in splits. Consumers should construct splits for their research use (train/test, by licence class, by repository size).
- **Are there errors, noise, or redundancies?** Expected:
  - Duplication is the subject of study. 80.2% of steps are byte-identical duplicates of another step after whitespace normalisation; this is the central empirical phenomenon.
  - Scenario Outline placeholder tokens (`<role>`, `<id>`) are preserved verbatim, not unrolled against the Examples table.
  - Internal whitespace is preserved; leading and trailing whitespace are trimmed.
  - Files that fail the `gherkin-official` parser produce zero rows. Observed causes: UTF-8 BOM prefixes from locale-specific editors, unterminated doc strings.
- **Sensitive content?** No PII is retained. Git commit-author metadata is stripped at the pointer level. A regex sweep for emails and IP addresses was applied to the permissive showcase directory prior to release. Content harvested from public GitHub remains subject to each source repository's licence and GitHub's Terms of Service.

### Coarse breakdown

| Axis | Count |
|---|---:|
| repositories (post-filter: stars ≥ 10, not archived) | 347 |
| `.feature` files (parsed into steps) | 23,667 |
| `.feature` files (enumerated on disk) | 25,034 |
| steps (total) | 1,113,616 |
| unique normalised step texts | 220,312 |
| Background steps | 61,214 |
| Scenario Outline steps | 66,020 |
| duplicate clusters (exact strategy) | 82,545 |
| duplicate clusters (hybrid strategy) | 65,242 |
| clusters with ≥ 1,000 occurrences (exact) | 86 |
| labelled pairs (calibration set) | 1,020 |

### Licence class distribution (by step count)

| class | steps | share |
|---|---:|---:|
| permissive | 635,586 | 57.1% |
| copyleft | 232,297 | 20.9% |
| unknown (NOASSERTION / detector failure) | 174,077 | 15.6% |
| unlicensed (no LICENSE file) | 71,656 | 6.4% |

The permissive subset (approximately 57% of steps) is redistributable as raw content; the remainder is pointer-only via `scripts/rehydrate.py`.

## 3. Collection process

- **How was the data acquired?** Three-stage pipeline under `scripts/`:
  1. **Discovery** via GitHub's REST Search API. Two complementary queries: (a) `/search/repositories?q=language:Gherkin+stars:>=10` and (b) `/search/code?q="Feature:"+extension:feature` (plus the same shape for `Scenario:` and `Background:`). The BigQuery alternative via GH Archive was not used (Google Cloud billing not available at collection time); the REST path is sufficient at this scale. Before deduplication the candidate pool totalled approximately 1,333 repositories; after deduplication by `owner/name` and re-application of the stars and archived filters, 377 unique repositories passed.
  2. **Shallow plus sparse clone** via `git clone --depth 1 --filter=blob:none --no-checkout` followed by `git sparse-checkout set --no-cone '**/*.feature'`. Only blobs for matching paths are downloaded. Pinned commit SHAs are recorded in `corpus/clone_manifest.jsonl`. 368 of 377 targeted repositories cloned successfully (9 private/deleted); approximately 2 KB of bandwidth per step acquired.
  3. **Parsing** via the `cukereuse` wrapper around `gherkin-official` v29 (Cucumber's authoritative parser), fanned across a thread pool. Emits `corpus/steps.parquet` (zstd-compressed). 21 of the 368 cloned repositories contribute zero steps to the final table (empty, tag-only, or ungrammatical files).
- **Over what timeframe?** Mining and parsing were run on 2026-04-19. Discovery captures GitHub's state on that day; pinned commit SHAs make the corpus content deterministic beyond that point.
- **Ethical review?** Data is drawn exclusively from public GitHub repositories. Licence compatibility is tracked per repository. No deanonymisation; no aggregation beyond the step level.

## 4. Preprocessing, cleaning, labelling

- **Text normalisation:** whitespace runs collapsed to single spaces; leading and trailing whitespace trimmed (`cukereuse.similarity.normalize`). No case-folding, Gherkin step text is case-sensitive in practice.
- **Doc strings and data tables:** treated as step arguments and excluded from the `text` column. The `gherkin-official` AST exposes them on separate fields, which this schema drops.
- **Scenario Outlines:** not unrolled. The outline body with `<placeholder>` tokens appears as one row, not one per Examples row.
- **Labelling:** manual, two-author, against the shared rubric described in §2. Every label carries the rule that fired so rubric application is auditable without re-annotating.
- **Raw data retention:** only pointers (`repo`, `commit_sha`, `file_path`) are kept. Raw feature files are reconstructed via `scripts/rehydrate.py`, which fetches `https://raw.githubusercontent.com/<repo>/<sha>/<path>`.

## 5. Uses

Supported tasks for v0.1:

- Threshold calibration for the `cukereuse` duplicate detector (paper §7).
- Empirical baseline for "what fraction of real BDD step text is duplicated?" (80.2% step-weighted, 58.6% median-repository).
- Replication of the bootstrap-CI calibration, the score-free relabelling protocol, the licence-stratified analysis, and the size-vs-duplication scatter.

Additional tasks the corpus supports:

- Cross-ecosystem BDD practice analysis (Java/Cucumber-JVM, Ruby/Cucumber, PHP/Behat, Python/behave, JS/cucumber-js).
- Pretraining or evaluation of language models on test-specification text.
- Research into automatic test refactoring, step-definition generation, tag taxonomies.
- Cognitive Dimensions of Notations analysis of BDD notations, grounded in concrete corpus examples (paper §4).

Restrictions:

- Do not redistribute copyleft-licensed raw feature files without complying with the source licence. Use the pointers plus `rehydrate.py`.
- Do not use the pointers to violate the source repositories' Terms of Service (rate limits, scraping).

## 6. Distribution

- **Release channel:** GitHub repository at [amughalbscs16/cukereuse](https://github.com/amughalbscs16/cukereuse). Apache-2.0 for source code and analytical schema.
- **Release bundle:** `repos.csv`, `clone_manifest.jsonl`, `steps.parquet`, `clusters_exact.parquet`, `clusters_hybrid.parquet`, `cluster_members_exact.parquet`, `cluster_members_hybrid.parquet`, `labeled_pairs.jsonl`, `LABELING_RUBRIC.md`, this datasheet, `README.md`. Total approximately 46 MB.
- **What is NOT redistributed:** verbatim raw `.feature`-file bodies (approximately 418 MB of content that inherits copyleft obligations from its source repositories). `rehydrate.py` fetches each original file from its upstream repository on demand at the pinned commit SHA.

## 7. Maintenance

- **Maintainers:** the authors of the `cukereuse` repository (Ali Hassaan Mughal, Noor Fatima, and Muhammad Bilal).
- **Errata:** filed as GitHub issues on [amughalbscs16/cukereuse](https://github.com/amughalbscs16/cukereuse).
- **Versioning:** a frozen snapshot corresponds to each paper version, tagged as a GitHub release (`v0.1`, `v0.2`, …). Supersession of earlier versions is documented in the release notes.
