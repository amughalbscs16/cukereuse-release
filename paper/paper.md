---
title: 'cukereuse: A paraphrase-robust static detector for duplicate Gherkin steps in Behavior-Driven Development test suites'
tags:
  - Python
  - software testing
  - behavior-driven development
  - clone detection
  - Gherkin
  - Cucumber
authors:
  - name: Ali Hassaan Mughal
    orcid: 0000-0002-0724-9197
    corresponding: true
    affiliation: 1
  - name: Noor Fatima
    affiliation: 2
  - name: Muhammad Bilal
    orcid: 0000-0003-4106-0256
    affiliation: 3
affiliations:
  - name: Independent Researcher, Texas Wesleyan University, USA
    index: 1
  - name: Independent Researcher, National University of Sciences and Technology (NUST), Pakistan
    index: 2
  - name: Independent Researcher, Technical University of Munich, Germany
    index: 3
date: 24 April 2026
bibliography: paper.bib
---

# Summary

Behavior-Driven Development (BDD), introduced by @north2006introducing,
expresses software requirements as example interactions with the system
using a structured `Given-When-Then` natural-language notation known as
Gherkin [@gherkin]. As BDD test suites grow, semantically equivalent
steps proliferate under different surface phrasings — *"the response
is 200"*, *"status should be 200 OK"*, *"I should get a 200 response
code"* — creating maintenance burden that existing clone-detection
tooling does not address on a purely static basis. `cukereuse` is an
open-source Python tool that detects exact and paraphrase-level
duplicate Gherkin steps across any Cucumber-compatible repository
using a layered pipeline of BLAKE2b hashing, Levenshtein ratio
matching [@levenshtein1966], and sentence-transformer embeddings
[@reimers2019sentencebert]. The tool ships with a 1.1-million-step
corpus mined from 347 public GitHub repositories, a 1,020-pair
labelled calibration set with documented inter-annotator agreement,
and a written labelling rubric, enabling both practical use by BDD
maintainers and empirical research on test-artefact quality.

# Statement of need

BDD adoption at industrial scale has produced test suites containing
tens of thousands of Gherkin steps per project, and the quality of
these specifications is now an established concern in the BDD
research literature [@binamungu2020characterising]. Linguistic drift
across the steps is endemic: our accompanying corpus study
[@mughal2026cukereuse] reports a step-weighted exact-duplicate rate
of 80.2% and a median repository duplication rate of 58.6% across
347 cross-organisation repositories, with a single canonical cluster
(*"the response status is 200 OK"*) grouping 20,737 occurrences
across 2,245 files in 43 repositories.

Existing approaches do not cover this problem well.
@binamungu2018detecting detect duplicate BDD examples via dynamic
execution tracing, which requires running the tests to observe
step-definition binding and restricts the approach to projects with
working test infrastructure and current dependencies — a common
obstacle in legacy or abandoned suites. @irshad2022supporting apply
Normalised Compression Distance to identify BDD refactoring
candidates at industrial scale within a single organisation, but
the approach operates at the scenario level rather than the
step-text level and has not been released as cross-organisation
tooling. Generic code-clone detectors designed for programming
languages, such as SourcererCC [@sajnani2016] and NiCad
[@roy2008nicad] (see @roy2009comparison for a broader taxonomy of
clone-detection techniques), handle token-level similarity well but
are not tuned for Gherkin's paraphrase-heavy, near-natural-language
character: on our calibration benchmark these baselines achieve
F$_1$ = 0.761 and 0.799 respectively, against F$_1$ = 0.822 for
`cukereuse`'s near-exact strategy under score-free evaluation.

`cukereuse` fills this gap with a purely static, framework-agnostic
detector that runs without test execution and supports Cucumber-JVM
(Java), `behave` and `pytest-bdd` (Python), Cucumber-Ruby,
cucumber-js (JavaScript), SpecFlow (.NET), and Behat (PHP). Three
audiences are served: maintainers of large BDD suites consolidating
drift before it worsens; platform and developer-experience teams
building pre-commit hooks that warn when a new step is the n-th
paraphrase of an existing one; and researchers working on
test-artefact quality, near-natural-language clone detection, and
BDD-specific tooling.

# Functionality

The CLI exposes four detection strategies, calibrated against the
1,020-pair labelled benchmark under bootstrap confidence intervals
[@mughal2026cukereuse]: `exact` (BLAKE2b hashing) for byte-identical
enumeration; `near-exact` (Levenshtein ratio, threshold 0.80), which
is the strongest pair-level classifier in the accompanying study;
`semantic` (SBERT cosine, threshold 0.82); and `hybrid`, which
combines SBERT cosine with a Levenshtein band (0.30–0.95) to
prevent the transitive chaining that pure semantic clustering
suffers from, for paraphrase-aware cluster reports. Each run
produces both a browsable HTML report with collapsible cluster
cards and file-line links, and a stable-schema JSON output for
downstream pipelines, alongside a recommended canonical phrasing
per cluster.

The released corpus ships every analytical artefact as Parquet and
JSONL (46 MB combined): per-step rows with repository, commit SHA,
file path, line, keyword, text, and SPDX licence class; exact and
hybrid cluster membership; and the 1,020-pair labelled pair set
with the fired rubric rule per pair. Inter-annotator agreement on
a 60-pair overlap subset is Fleiss' $\kappa = 0.84$. Raw `.feature`
file bodies are not redistributed; they are fetched on demand from
pinned upstream commit SHAs under their source licences. The mining
and analysis pipeline is fully scripted and resumable, enabling
corpus extension on new repository samples.

# Related tooling

Cucumber's own official tooling provides no duplicate-step
detection; existing IDE integrations offer exact-match step
auto-complete only. Runtime-based approaches
[@binamungu2018detecting] detect duplicate step definitions (the
code behind the steps) rather than duplicate step text, and
require executable test infrastructure. Generic clone detectors
[@sajnani2016; @roy2008nicad] treat Gherkin as ordinary source
code and miss paraphrase-level equivalence. To our knowledge,
`cukereuse` is the first open-source, static, paraphrase-aware,
cross-framework detector for duplicate Gherkin step text released
with a cross-organisation corpus of this scale.

# Acknowledgements

We thank the open-source communities of Cucumber, `behave`,
`pytest-bdd`, Cucumber-Ruby, cucumber-js, SpecFlow, and Behat for
their tooling and documentation, and the maintainers of the 347
public repositories whose `.feature` files form the basis of the
corpus. This work received no external funding.

# References
