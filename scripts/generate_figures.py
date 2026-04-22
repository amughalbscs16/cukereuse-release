# ruff: noqa: I001
"""Generate publication-quality PDF figures from the cukereuse corpus.

Reads the analysis artefacts already on disk:
  corpus/steps.parquet
  corpus/clusters_hybrid.parquet
  corpus/labeled_pairs.jsonl
  analysis/calibration.json
  analysis/license_stratified.json

Writes PDFs to ./figures/ by default (override with --out-dir).
"""

from __future__ import annotations

import torch  # noqa: F401  # MUST import before pandas on Windows

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- global style --------------------------------------------------------

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 11.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)

COL_PERM = "#2a6f4b"
COL_COPY = "#c5563c"
COL_UNKN = "#9e3c85"
COL_UNLI = "#666666"
COL_EXACT = "#1f77b4"
COL_NEAR = "#ff7f0e"
COL_SEM = "#2ca02c"
COL_HYB = "#d62728"


def _human(n: int) -> str:
    """Format an integer with thousands separators (commas)."""
    return f"{n:,}"


# --- figure 1: pipeline flowchart ---------------------------------------


def fig_pipeline(out_dir: Path) -> None:
    """Three side-by-side vertical panels: Discovery, Materialisation, Analysis.

    Each panel is a straight top-to-bottom chain with uniformly-sized boxes
    and single vertical arrows between stages.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 5.6))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.03, wspace=0.18)

    def draw_panel(ax, title, color, stages):
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.text(
            1.5, 9.55, title,
            ha="center", va="center",
            fontweight="bold", fontsize=11,
        )

        n = len(stages)
        top = 8.7
        bottom = 0.3
        box_h = 1.15
        if n > 1:
            gap = ((top - bottom) - n * box_h) / (n - 1)
        else:
            gap = 0.0
        box_w = 2.6
        box_x = 0.2

        centers = []
        for i, text in enumerate(stages):
            y = top - i * (box_h + gap) - box_h
            r = patches.FancyBboxPatch(
                (box_x, y), box_w, box_h,
                boxstyle="round,pad=0.08",
                linewidth=1.1,
                edgecolor="#222",
                facecolor=color,
                alpha=0.92,
            )
            ax.add_patch(r)
            ax.text(
                box_x + box_w / 2, y + box_h / 2, text,
                ha="center", va="center", fontsize=9.0,
            )
            centers.append((box_x + box_w / 2, y, y + box_h))

        for i in range(n - 1):
            x, _ybot, ytop_i = centers[i]
            _, ybot_next, _ytop_next = centers[i + 1]
            ax.annotate(
                "",
                xy=(x, ybot_next + box_h),
                xytext=(x, ytop_i - box_h),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#333",
                    "lw": 1.3,
                    "shrinkA": 2,
                    "shrinkB": 2,
                    "mutation_scale": 14,
                },
            )

    discovery = [
        "GitHub REST Search API\n(repo + code queries)",
        "1,333 candidates\nafter three keyword shapes",
        "Filter: stars \u2265 10,\nnot archived",
        "377 unique repositories\ntagged by SPDX licence",
    ]
    materialisation = [
        "Sparse clone\n(*.feature, blob:none)",
        "368 repos cloned\n(9 removed/private)",
        "gherkin-official parser\n(parallel workers)",
        "23,667 files parsed\n1,113,616 steps",
        "steps.parquet\n(zstd-compressed)",
    ]
    analysis = [
        "Exact hash (BLAKE2b)\n\u2192 82,545 clusters",
        "Lev ratio + length guard\n(near-exact layer)",
        "MiniLM-L6-v2 cosine\n(semantic layer)",
        "Union-find\n\u2192 65,242 hybrid clusters",
        "HTML + JSON reports\n+ parquet artefacts",
    ]

    draw_panel(axes[0], "1. Discovery", "#e3f2fd", discovery)
    draw_panel(axes[1], "2. Materialisation", "#fff3e0", materialisation)
    draw_panel(axes[2], "3. Analysis", "#e8f5e9", analysis)

    fig.savefig(out_dir / "fig_pipeline.pdf")
    plt.close(fig)


# --- figure 2: cluster size distribution (CCDF log-log) -----------------


def fig_cluster_sizes(out_dir: Path) -> None:
    df = pd.read_parquet("corpus/clusters_hybrid.parquet")
    sizes = np.sort(df["count"].to_numpy())[::-1]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    rank = np.arange(1, len(sizes) + 1)
    # subsample for plotting to reduce PDF size while keeping shape
    if len(sizes) > 5000:
        idx = np.unique(np.round(np.logspace(0, np.log10(len(sizes)), 1500)).astype(int)) - 1
        idx = idx[idx < len(sizes)]
        ax.loglog(rank[idx], sizes[idx], marker=".", markersize=3, linestyle="-", linewidth=0.7, color=COL_HYB)
    else:
        ax.loglog(rank, sizes, marker=".", markersize=3, linestyle="-", linewidth=0.7, color=COL_HYB)

    ax.set_xlabel("cluster rank (log scale)")
    ax.set_ylabel("cluster size (occurrences, log scale)")
    ax.set_title(f"Cluster-size distribution (hybrid strategy, n={_human(len(sizes))} clusters)")

    # annotate top cluster
    ax.annotate(
        f"largest cluster: {_human(int(sizes[0]))}\n\u201cthe response status is 200 OK\u201d",
        xy=(1, sizes[0]),
        xytext=(6, sizes[0] * 0.35),
        fontsize=7.5,
        arrowprops={"arrowstyle": "->", "color": "#555", "lw": 0.6},
    )
    # median
    med = np.median(sizes)
    ax.axhline(med, color="#888", linewidth=0.6, linestyle="--")
    ax.text(
        len(sizes) * 0.35,
        med * 1.25,
        f"median: {int(med)}",
        fontsize=7.5,
        color="#555",
    )

    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(out_dir / "fig_cluster_sizes.pdf")
    plt.close(fig)


# --- figure 3: license-stratified duplication --------------------------


def fig_license_stratified(out_dir: Path) -> None:
    data = json.loads(Path("analysis/license_stratified.json").read_text(encoding="utf-8"))
    per = data["per_class"]
    pooled_rate = data["overall"]["exact_dup_rate"] * 100

    order = ["permissive", "copyleft", "unknown", "unlicensed"]
    colors = [COL_PERM, COL_COPY, COL_UNKN, COL_UNLI]
    rates = [per[c]["exact_dup_rate"] * 100 for c in order]
    steps = [per[c]["n_steps"] for c in order]
    repos = [per[c]["n_repos"] for c in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [3, 2], "wspace": 0.35})

    # Left: duplication rate bars
    bars = ax1.bar(order, rates, color=colors, edgecolor="black", linewidth=0.6, width=0.7)
    ax1.axhline(pooled_rate, color="black", linewidth=0.7, linestyle=":")
    ax1.text(
        3.55,
        pooled_rate + 1.5,
        f"pooled: {pooled_rate:.1f}%",
        fontsize=8,
        color="black",
        style="italic",
        ha="right",
    )
    for b, r, n, rp in zip(bars, rates, steps, repos, strict=False):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.8,
            f"{r:.1f}%",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() / 2,
            f"{n / 1000:.0f}k steps\n{rp} repos",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
    ax1.set_ylabel("exact-duplication rate (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Duplication rate by licence class")
    ax1.grid(axis="y", alpha=0.3)
    ax1.grid(axis="x", visible=False)

    # Right: step count pie (showing which class dominates the pooled stat)
    total = sum(steps)
    ax2.pie(
        steps,
        labels=[f"{c}\n{_human(n)}" for c, n in zip(order, steps, strict=False)],
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        textprops={"fontsize": 8},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.72,
    )
    ax2.set_title(f"Corpus composition (step count, total {_human(total)})")

    fig.savefig(out_dir / "fig_license_stratified.pdf")
    plt.close(fig)


# --- figure 4: threshold sweep (F1 vs threshold + PR curve) ------------


def fig_threshold_sweep(out_dir: Path) -> None:
    cal = json.loads(Path("analysis/calibration.json").read_text(encoding="utf-8"))
    sweep = cal["sweep"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.28})

    # Distinct linestyle + marker shape + small horizontal offset for the
    # hybrid best-F1 dot so the three strategies remain visually separable
    # when their best thresholds collide at 0.82.
    strategies = [
        ("semantic",   "semantic (cosine)",              COL_SEM,  "-",  "o",  0.0),
        ("near_exact", "near-exact (Levenshtein)",       COL_NEAR, "-",  "^",  0.0),
        ("hybrid",     "hybrid (cos $\\wedge$ Lev-band)", COL_HYB,  "--", "s", 0.010),
    ]
    for strat_key, label, col, ls, marker, dot_x_offset in strategies:
        rows = sweep[strat_key]
        thr = [r["threshold"] for r in rows]
        f1 = [r["f1"] for r in rows]
        p = [r["precision"] for r in rows]
        r = [r["recall"] for r in rows]

        ax1.plot(thr, f1, linewidth=1.3, linestyle=ls, label=label, color=col)
        best = max(rows, key=lambda x: x["f1"])
        ax1.scatter(
            [best["threshold"] + dot_x_offset], [best["f1"]],
            color=col, s=56, zorder=5,
            edgecolor="black", linewidth=0.7, marker=marker,
        )

        ax2.plot(r, p, linewidth=1.3, linestyle=ls, label=label, color=col, alpha=0.95)

    # Add best-F1 labels OUTSIDE data clusters to avoid overlap
    best_points = []
    for strat_key, label, col in [
        ("semantic", "semantic", COL_SEM),
        ("near_exact", "near-exact", COL_NEAR),
        ("hybrid", "hybrid", COL_HYB),
    ]:
        rows = sweep[strat_key]
        best = max(rows, key=lambda x: x["f1"])
        best_points.append((strat_key, label, col, best))

    # Annotate best-F1 points in an annotations box to the right of the F1 panel
    annot_text = "Best F$_1$ operating points:"
    for _, label, _, best in best_points:
        annot_text += f"\n  {label:<11} F$_1$={best['f1']:.3f} @ thr={best['threshold']:.2f}"
    ax1.text(
        0.05,
        0.05,
        annot_text,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#888", "alpha": 0.9},
    )

    ax1.set_xlabel("decision threshold")
    ax1.set_ylabel("F$_1$")
    ax1.set_title("F$_1$ versus threshold")
    ax1.set_xlim(0.5, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_title("Precision versus recall")
    ax2.set_xlim(0, 1.03)
    ax2.set_ylim(0, 1.03)
    ax2.legend(loc="lower left", frameon=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.savefig(out_dir / "fig_threshold_sweep.pdf")
    plt.close(fig)


# --- figure 5: top clusters horizontal bar ------------------------------


def fig_top_clusters(out_dir: Path) -> None:
    df = pd.read_parquet("corpus/clusters_hybrid.parquet").sort_values("count", ascending=False).head(20)
    # Truncate long labels cleanly at word boundary
    def _trunc(t: str, n: int = 60) -> str:
        if len(t) <= n:
            return t
        cut = t[: n - 3]
        return cut.rsplit(" ", 1)[0] + "..."

    labels = [_trunc(t) for t in df["canonical_text"].tolist()]
    counts = df["count"].to_numpy()[::-1]
    files = df["occurrence_files"].to_numpy()[::-1]
    labels_rev = labels[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        range(len(labels_rev)),
        counts,
        color=COL_HYB,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.87,
        height=0.72,
    )
    for bar, cnt, f in zip(bars, counts, files, strict=False):
        ax.text(
            cnt + max(counts) * 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{_human(int(cnt))}  ({f} files)",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(len(labels_rev)))
    ax.set_yticklabels(labels_rev, fontsize=8.5, fontfamily="monospace")
    ax.set_xlabel("occurrences (hybrid strategy)")
    ax.set_title(f"Top 20 duplicate clusters in the {_human(1113616)}-step corpus")
    ax.set_xlim(0, max(counts) * 1.28)
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)

    fig.savefig(out_dir / "fig_top_clusters.pdf")
    plt.close(fig)


# --- figure 6: CDN radar chart ------------------------------------------


def fig_cdn_radar(out_dir: Path) -> None:
    dims = [
        ("Viscosity",             2),
        ("Hidden dependencies",   2),
        ("Premature commitment",  2),
        ("Role-expressiveness",   3),
        ("Consistency",           2),
        ("Diffuseness",           2),
        ("Error-proneness",       2),
        ("Hard mental ops",       4),
        ("Closeness of mapping",  3),
        ("Progressive evaluation", 1),
        ("Provisionality",        1),
        ("Visibility",            2),
        ("Abstraction",           3),
        ("Secondary notation",    3),
    ]
    labels = [d[0] for d in dims]
    values = [d[1] for d in dims]
    n = len(dims)
    angles = [i / n * 2 * math.pi for i in range(n)] + [0]
    values_plot = [*values, values[0]]

    fig, ax = plt.subplots(figsize=(8, 7.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["Unsupp.", "Problem.", "Mixed", "Moderate", "Good"], fontsize=7.5)
    ax.tick_params(axis="x", pad=12)
    ax.grid(True, alpha=0.4)
    ax.spines["polar"].set_visible(False)

    theta_fill = np.linspace(0, 2 * math.pi, 200)
    ax.fill_between(theta_fill, 0, 2, color="#ffcccc", alpha=0.35, linewidth=0)
    ax.fill_between(theta_fill, 4, 5, color="#ccffcc", alpha=0.35, linewidth=0)

    ax.plot(angles, values_plot, color=COL_HYB, linewidth=1.6, marker="o", markersize=5)
    ax.fill(angles, values_plot, color=COL_HYB, alpha=0.22)

    ax.set_title(
        "Cognitive Dimensions ratings for Gherkin\n"
        "(red band: problematic / unsupported,\u2003 green band: good)",
        pad=28,
        fontsize=10.5,
    )

    fig.savefig(out_dir / "fig_cdn_radar.pdf")
    plt.close(fig)


# --- figure 7: step-length distribution ----------------------------------


def fig_step_lengths(out_dir: Path) -> None:
    df = pd.read_parquet("corpus/steps.parquet", columns=["text"])
    lengths = df["text"].str.len().to_numpy()
    pcts = np.percentile(lengths, [50, 90, 95, 99])
    labels = [50, 90, 95, 99]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(lengths, bins=np.logspace(0, 3.5, 80), color=COL_EXACT, alpha=0.82, edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("step-text length (characters, log scale)")
    ax.set_ylabel("step count")
    ax.set_title(f"Step-text length distribution (n={_human(len(lengths))})")

    # stack percentile labels vertically so they don't overlap
    ymax = ax.get_ylim()[1]
    for i, (p, v) in enumerate(zip(labels, pcts, strict=False)):
        ax.axvline(v, linestyle="--", linewidth=0.5, color="#555", alpha=0.7)
        ax.text(v * 1.07, ymax * (0.92 - i * 0.08), f"p{p}={int(v)}", fontsize=8, color="#333")

    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(out_dir / "fig_step_lengths.pdf")
    plt.close(fig)


# --- figure 8: strategy comparison diagram -----------------------------


def fig_strategy_ladder(out_dir: Path) -> None:
    """Visualises the four detection strategies in the precision-vs-compute
    trade-off space; bubble area is proportional to recall on the labelled
    set."""
    cal = json.loads(Path("analysis/calibration.json").read_text(encoding="utf-8"))
    best = cal["best"]

    # Derive best operating points for semantic, near_exact, hybrid; exact is
    # trivial P=1 R=limited-by-exact-coverage.
    # Exact P/R on labelled set: compute directly.
    df_lbl = [json.loads(line) for line in Path("corpus/labeled_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    tp = sum(1 for r in df_lbl if r["label"] == 1 and abs(r["lev"] - 1.0) < 1e-6)
    fp = sum(1 for r in df_lbl if r["label"] == 0 and abs(r["lev"] - 1.0) < 1e-6)
    fn_exact = sum(1 for r in df_lbl if r["label"] == 1 and abs(r["lev"] - 1.0) >= 1e-6)
    p_exact = tp / max(tp + fp, 1)
    r_exact = tp / max(tp + fn_exact, 1)

    strategies = [
        ("exact\nhash",       0.5, p_exact if p_exact > 0 else 1.0, max(r_exact, 0.05), "byte-identical only",          COL_EXACT),
        ("near-exact\nLev",   1.5, best["near_exact"]["precision"], best["near_exact"]["recall"], "parametric variants",   COL_NEAR),
        ("semantic\nSBERT",   3.0, best["semantic"]["precision"],   best["semantic"]["recall"],   "paraphrases",           COL_SEM),
        ("hybrid\ncos $\\wedge$ Lev-band", 3.5, best["hybrid"]["precision"], best["hybrid"]["recall"], "disciplined semantic", COL_HYB),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    for name, cost, p, r, desc, col in strategies:
        size = max(r, 0.05) * 1800
        ax.scatter([cost], [p], s=size, color=col, alpha=0.35, edgecolor="black", linewidth=0.8)
        ax.scatter([cost], [p], s=70, color=col, edgecolor="black", linewidth=0.8, zorder=5)
        ax.annotate(name, xy=(cost, p), xytext=(cost, p + 0.06), ha="center", fontsize=9, fontweight="bold")
        ax.annotate(desc, xy=(cost, p), xytext=(cost, p - 0.07), ha="center", fontsize=7.8, color="#555", style="italic")
        ax.annotate(f"R={r:.2f}", xy=(cost, p), xytext=(cost, p - 0.12), ha="center", fontsize=7.5, color=col)

    ax.set_xlabel("compute cost per pair (relative)")
    ax.set_ylabel("precision on the 1,020-pair labelled set")
    ax.set_title("Four detection strategies: precision vs compute; bubble area \u221d recall")
    ax.set_xlim(-0.2, 4.8)
    ax.set_ylim(0.3, 1.15)
    ax.set_xticks([0.5, 1.5, 3.0, 3.5])
    ax.set_xticklabels(["hash", "Levenshtein", "SBERT", "hybrid"], fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.savefig(out_dir / "fig_strategy_ladder.pdf")
    plt.close(fig)


# --- main ---------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ("pipeline flowchart", fig_pipeline),
        ("cluster-size distribution", fig_cluster_sizes),
        ("licence-stratified duplication", fig_license_stratified),
        ("threshold sweep + PR curves", fig_threshold_sweep),
        ("top-20 clusters", fig_top_clusters),
        ("CDN radar", fig_cdn_radar),
        ("step-length histogram", fig_step_lengths),
        ("strategy trade-off ladder", fig_strategy_ladder),
    ]
    for name, fn in figures:
        print(f"  generating: {name} ...", flush=True)
        try:
            fn(args.out_dir)
            print("    done")
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)

    print(f"\nAll figures in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
