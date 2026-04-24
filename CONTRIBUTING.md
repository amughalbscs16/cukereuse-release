# Contributing to cukereuse

Thanks for your interest. This file describes how to report issues,
propose changes, and set up a local development environment.

By participating in this project, you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs

Open a GitHub issue using the **Bug report** template. Please include:

- A short, specific title describing what went wrong.
- The exact command you ran.
- The exact output or error (paste verbatim — screenshots are okay for
  cluster reports but not for terminal output).
- Your operating system, Python version (`python --version`), and
  `cukereuse` version (`cukereuse --version`).
- If you can share a minimal `.feature` file or snippet that reproduces
  the issue, please attach it.

If the issue involves a specific repository in the public corpus, the
repository identifier (from `corpus/repos.csv`) and commit SHA (from
`corpus/clone_manifest.jsonl`) are usually the shortest reproduction
path.

## Suggesting features

Open a GitHub issue using the **Feature request** template. The most
useful suggestions explain:

- The workflow or use case the feature would support.
- How the current behaviour falls short for that workflow.
- What the output of the new feature would look like (CLI flag,
  report change, new command, etc.).

We are especially interested in requests from maintainers of real
BDD suites — a concrete operational example is more valuable than an
abstract capability request.

## Development setup

We use [`uv`](https://docs.astral.sh/uv/) for dependency management and
project orchestration.

```console
$ git clone https://github.com/amughalbscs16/cukereuse-release.git
$ cd cukereuse-release
$ uv sync --dev
```

This installs runtime dependencies, test dependencies, and the dev
toolchain (pytest, ruff, mypy, matplotlib). Torch is pulled from the
CPU wheel index; no GPU is required.

### Running the test suite

```console
$ uv run pytest
$ uv run pytest --cov=cukereuse --cov-report=term-missing
```

### Running the CLI against a small corpus

A handful of fixture `.feature` files live under `tests/fixtures/`.

```console
$ uv run cukereuse stats tests/fixtures
$ uv run cukereuse find-duplicates tests/fixtures --strategy hybrid
```

### Linting and type-checking

```console
$ uv run ruff check src tests
$ uv run mypy src
```

Both are run by CI on every push and pull request.

## Pull requests

Small, focused pull requests are much easier to review than sweeping
changes. A good pull request:

- Has a clear title summarising the change in one line.
- Has a description explaining *why* the change is needed, not just
  what it does.
- Includes tests for any behavioural change (new clusters, new CLI
  flags, parser fixes, etc.).
- Keeps unrelated refactors in separate commits or pull requests.
- Passes `pytest`, `ruff check`, and `mypy src` locally before you push.

For non-trivial changes, please open an issue first to discuss the
approach. For documentation fixes, typo corrections, or dependency
bumps, a direct PR is fine.

## Reporting security issues

Please do *not* open a public issue for security vulnerabilities.
Instead email the corresponding author
(`alihassaanmughal.work@gmail.com`) with a description of the issue
and steps to reproduce. We will acknowledge within 7 days and aim to
provide a timeline for a fix within 14 days.

## Acknowledgement of contributors

All accepted contributions are recorded in the git history.
Substantive contributions (new features, substantial bug fixes,
documentation overhauls) are additionally acknowledged in the
release notes. If you would like to be listed in `CITATION.cff` for
an ongoing contribution, please open an issue so we can discuss
scope.
