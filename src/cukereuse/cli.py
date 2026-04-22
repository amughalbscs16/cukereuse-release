from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from cukereuse import __version__
from cukereuse.clustering import (
    Cluster,
    cluster_exact,
    cluster_hybrid,
    cluster_near_exact,
    cluster_semantic,
)
from cukereuse.parser import parse_directory
from cukereuse.reporter import Report, write_html, write_json

app = typer.Typer(
    name="cukereuse",
    help="Static, paraphrase-robust duplicate step detection for Cucumber/Gherkin.",
    no_args_is_help=True,
)

FeaturesDir = Annotated[
    Path,
    typer.Argument(
        exists=True, file_okay=False, dir_okay=True, help="Directory of .feature files."
    ),
]


def _run_pipeline(
    path: Path,
    console: Console,
    *,
    strategy: str = "exact",
    lev_threshold: float = 0.92,
    cos_threshold: float = 0.85,
    lev_min: float = 0.3,
    lev_max: float = 0.95,
) -> tuple[Report, list[Cluster]]:
    console.print(f"[bold]Parsing[/bold] {path} ...")
    results = list(parse_directory(path))

    steps = [s for r in results if r.error is None for s in r.steps]
    n_errors = sum(1 for r in results if r.error is not None)
    n_files = len(results)
    n_unique = len({s.text.strip() for s in steps})

    console.print(
        f"Parsed [bold]{n_files}[/bold] files "
        f"(errors: [red]{n_errors}[/red]) -> [bold]{len(steps):,}[/bold] steps."
    )
    if not steps:
        return (
            Report(
                root_path=path.resolve(),
                n_feature_files=n_files,
                n_parse_errors=n_errors,
                strategy=strategy,
            ),
            [],
        )

    if strategy == "exact":
        console.print("[bold]Clustering[/bold] (exact duplicates) ...")
        clusters = cluster_exact(steps)
    elif strategy == "near-exact":
        console.print(
            f"[bold]Clustering[/bold] (exact + Levenshtein, threshold={lev_threshold}) ..."
        )
        clusters = cluster_near_exact(steps, lev_threshold=lev_threshold)
    elif strategy == "semantic":
        console.print(
            f"[bold]Clustering[/bold] (SBERT semantic, cos>={cos_threshold}) — "
            "first run downloads ~80MB model ..."
        )
        clusters = cluster_semantic(steps, cos_threshold=cos_threshold)
    elif strategy == "hybrid":
        console.print(
            f"[bold]Clustering[/bold] (hybrid: cos>={cos_threshold} AND "
            f"lev in [{lev_min}, {lev_max}]) — first run downloads ~80MB model ..."
        )
        clusters = cluster_hybrid(
            steps,
            cos_threshold=cos_threshold,
            lev_min=lev_min,
            lev_max=lev_max,
        )
    else:
        raise typer.BadParameter(
            f"Unknown strategy '{strategy}'. " "Choose one of: exact, near-exact, semantic, hybrid."
        )

    report = Report(
        root_path=path.resolve(),
        strategy=strategy,
        n_feature_files=n_files,
        n_parse_errors=n_errors,
        n_steps=len(steps),
        n_unique_step_texts=n_unique,
        n_duplicate_clusters=len(clusters),
        clusters=tuple(clusters),
    )
    console.print(
        f"Found [bold]{len(clusters):,}[/bold] duplicate clusters "
        f"({report.exact_duplication_rate * 100:.1f}% of steps are exact duplicates)."
    )
    return report, clusters


def _print_top_clusters(report: Report, console: Console, top_n: int = 20) -> None:
    if not report.clusters:
        return
    table = Table(title=f"Top {min(top_n, len(report.clusters))} duplicate clusters")
    table.add_column("count", justify="right", style="green")
    table.add_column("files", justify="right", style="dim")
    table.add_column("canonical text", style="white")
    for cluster in report.clusters[:top_n]:
        snippet = cluster.canonical_text
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        table.add_row(str(cluster.count), str(cluster.occurrence_files), snippet)
    console.print(table)


@app.command()
def version() -> None:
    """Print the installed cukereuse version."""
    typer.echo(__version__)


StrategyOpt = Annotated[
    str,
    typer.Option(
        "--strategy",
        help=(
            "Clustering strategy: exact | near-exact | semantic | hybrid. "
            "hybrid is the plan's recommended default: cos AND lev-band."
        ),
    ),
]

LevThreshOpt = Annotated[
    float,
    typer.Option(
        "--lev-threshold",
        help=(
            "Levenshtein ratio threshold for near-exact strategy (0,1). "
            "Default 0.92 is the empirical knee on real BDD suites."
        ),
    ),
]

CosThreshOpt = Annotated[
    float,
    typer.Option(
        "--cos-threshold",
        help=(
            "Cosine similarity threshold for semantic/hybrid (0,1). Default 0.90 "
            "is the empirical knee on real BDD suites — below 0.90, MiniLM "
            "conflates domain-heavy vocabulary (account/license/token) and "
            "single-linkage union-find chains hundreds of unrelated variants."
        ),
    ),
]

LevMinOpt = Annotated[
    float,
    typer.Option(
        "--lev-min",
        help="Lower Lev bound for hybrid band [lev_min, lev_max]. Default 0.3.",
    ),
]

LevMaxOpt = Annotated[
    float,
    typer.Option(
        "--lev-max",
        help="Upper Lev bound for hybrid band [lev_min, lev_max]. Default 0.95.",
    ),
]


@app.command()
def analyze(
    path: FeaturesDir,
    strategy: StrategyOpt = "exact",
    lev_threshold: LevThreshOpt = 0.92,
    cos_threshold: CosThreshOpt = 0.90,
    lev_min: LevMinOpt = 0.3,
    lev_max: LevMaxOpt = 0.95,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Optional JSON output path.")
    ] = None,
) -> None:
    """Run the full pipeline on a directory of .feature files (stats + duplicates)."""
    console = Console()
    report, _ = _run_pipeline(
        path,
        console,
        strategy=strategy,
        lev_threshold=lev_threshold,
        cos_threshold=cos_threshold,
        lev_min=lev_min,
        lev_max=lev_max,
    )
    _print_top_clusters(report, console)
    if output:
        write_json(report, output)
        console.print(f"[green]Wrote JSON report:[/green] {output}")


@app.command("find-duplicates")
def find_duplicates(
    path: FeaturesDir,
    strategy: StrategyOpt = "exact",
    lev_threshold: LevThreshOpt = 0.92,
    cos_threshold: CosThreshOpt = 0.90,
    lev_min: LevMinOpt = 0.3,
    lev_max: LevMaxOpt = 0.95,
    output: Annotated[Path, typer.Option("--output", "-o", help="HTML report path.")] = Path(
        "report.html"
    ),
    json_output: Annotated[
        Path | None,
        typer.Option("--json", help="Also write a JSON report. Default: <output>.json"),
    ] = None,
) -> None:
    """Detect duplicate/near-duplicate steps and emit reports."""
    console = Console()
    report, _ = _run_pipeline(
        path,
        console,
        strategy=strategy,
        lev_threshold=lev_threshold,
        cos_threshold=cos_threshold,
        lev_min=lev_min,
        lev_max=lev_max,
    )
    _print_top_clusters(report, console)

    write_html(report, output)
    console.print(f"[green]Wrote HTML report:[/green] {output}")
    json_path = json_output or output.with_suffix(".json")
    write_json(report, json_path)
    console.print(f"[green]Wrote JSON report:[/green] {json_path}")


@app.command()
def stats(path: FeaturesDir) -> None:
    """Summary statistics over a directory of .feature files (no duplicate detection)."""
    console = Console()
    results = list(parse_directory(path))
    steps = [s for r in results if r.error is None for s in r.steps]
    n_errors = sum(1 for r in results if r.error is not None)

    table = Table(title=f"cukereuse stats: {path}")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("feature files parsed", f"{len(results):,}")
    table.add_row("parse errors", f"{n_errors:,}")
    table.add_row("total steps", f"{len(steps):,}")
    table.add_row("unique step texts", f"{len({s.text.strip() for s in steps}):,}")
    table.add_row("background steps", f"{sum(1 for s in steps if s.is_background):,}")
    table.add_row("outline steps", f"{sum(1 for s in steps if s.is_outline):,}")
    console.print(table)


@app.command()
def calibrate(
    labeled: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Path to labeled-pairs JSONL."),
    ],
    lev_min: Annotated[
        float, typer.Option("--lev-min", help="Hybrid Lev band lower bound.")
    ] = 0.3,
    lev_max: Annotated[
        float, typer.Option("--lev-max", help="Hybrid Lev band upper bound.")
    ] = 0.95,
) -> None:
    """Compute precision/recall/F1 across thresholds from a labeled-pairs JSONL file.

    Expects rows with ``{id, text_a, text_b, cos, lev, label, ...}``. Sweeps
    thresholds in 0.50..0.99 and reports the best-F1 point per strategy
    (near-exact / semantic / hybrid).
    """
    import json

    from rich.table import Table as RichTable

    rows: list[dict[str, Any]] = []
    with labeled.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        typer.echo("No rows in labeled file.", err=True)
        raise typer.Exit(2)

    required = {"cos", "lev", "label"}
    missing = required - set(rows[0].keys())
    if missing:
        typer.echo(f"Labeled JSONL missing required keys: {missing}", err=True)
        raise typer.Exit(2)

    thresholds = [round(x * 0.01, 2) for x in range(50, 100)]
    best: dict[str, dict[str, float | int] | None] = {
        "near-exact": None,
        "semantic": None,
        "hybrid": None,
    }

    def _score(pred: list[int], gold: list[int]) -> tuple[float, float, float, tuple[int, int, int, int]]:
        tp = sum(1 for p, g in zip(pred, gold, strict=False) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(pred, gold, strict=False) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(pred, gold, strict=False) if p == 0 and g == 1)
        tn = sum(1 for p, g in zip(pred, gold, strict=False) if p == 0 and g == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1, (tp, fp, fn, tn)

    gold = [int(r["label"]) for r in rows]
    for t in thresholds:
        pred_ne = [1 if float(r["lev"]) >= t else 0 for r in rows]
        pred_sem = [1 if float(r["cos"]) >= t else 0 for r in rows]
        pred_hy = [
            1
            if (float(r["cos"]) >= t and lev_min <= float(r["lev"]) <= lev_max)
            else 0
            for r in rows
        ]
        for name, pred in (("near-exact", pred_ne), ("semantic", pred_sem), ("hybrid", pred_hy)):
            p, r_, f1, counts = _score(pred, gold)
            key = (f1, p)
            current = best[name]
            if current is None or key > (float(current["f1"]), float(current["precision"])):
                best[name] = {
                    "threshold": t,
                    "precision": round(p, 4),
                    "recall": round(r_, 4),
                    "f1": round(f1, 4),
                    "tp": counts[0],
                    "fp": counts[1],
                    "fn": counts[2],
                    "tn": counts[3],
                }

    table = RichTable(
        title=f"cukereuse calibrate: {labeled}  (N={len(rows)}, duplicates={sum(gold)})",
    )
    table.add_column("strategy", style="bold")
    table.add_column("best_thr", justify="right")
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("F1", justify="right", style="green")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("TN", justify="right")
    for name in ("near-exact", "semantic", "hybrid"):
        b = best[name]
        assert b is not None
        table.add_row(
            name,
            f"{b['threshold']:.2f}",
            f"{b['precision']:.3f}",
            f"{b['recall']:.3f}",
            f"{b['f1']:.3f}",
            str(b["tp"]),
            str(b["fp"]),
            str(b["fn"]),
            str(b["tn"]),
        )
    Console().print(table)


if __name__ == "__main__":
    app()
