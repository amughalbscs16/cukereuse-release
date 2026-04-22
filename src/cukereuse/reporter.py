"""Report emission — JSON (stable schema) and single-file HTML."""

from __future__ import annotations

import datetime as _dt
import html
import json
from pathlib import Path  # noqa: TCH003  # pydantic resolves at runtime

from pydantic import BaseModel, ConfigDict, Field

from cukereuse.clustering import Cluster  # noqa: TCH001  # pydantic resolves at runtime


class Report(BaseModel):
    """Top-level report produced by the analyze / find-duplicates pipeline."""

    model_config = ConfigDict(frozen=True)

    root_path: Path
    strategy: str = "exact"
    generated_at: _dt.datetime = Field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc))
    n_feature_files: int = 0
    n_parse_errors: int = 0
    n_steps: int = 0
    n_unique_step_texts: int = 0
    n_duplicate_clusters: int = 0
    clusters: tuple[Cluster, ...] = Field(default_factory=tuple)

    @property
    def exact_duplication_rate(self) -> float:
        if self.n_steps == 0:
            return 0.0
        return 1 - (self.n_unique_step_texts / self.n_steps)


def write_json(report: Report, path: Path) -> None:
    """Emit the report as machine-readable JSON with a stable schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "root_path": str(report.root_path),
        "strategy": report.strategy,
        "generated_at": report.generated_at.isoformat(),
        "summary": {
            "n_feature_files": report.n_feature_files,
            "n_parse_errors": report.n_parse_errors,
            "n_steps": report.n_steps,
            "n_unique_step_texts": report.n_unique_step_texts,
            "n_duplicate_clusters": report.n_duplicate_clusters,
            "exact_duplication_rate": round(report.exact_duplication_rate, 4),
        },
        "clusters": [
            {
                "canonical_text": c.canonical_text,
                "strategy": c.strategy,
                "count": c.count,
                "occurrence_files": c.occurrence_files,
                "members": [
                    {
                        "keyword": m.keyword,
                        "text": m.text,
                        "file_path": str(m.file_path),
                        "line": m.line,
                        "scenario_name": m.scenario_name,
                        "feature_name": m.feature_name,
                        "is_background": m.is_background,
                        "is_outline": m.is_outline,
                        "tags": list(m.tags),
                    }
                    for m in c.members
                ],
            }
            for c in report.clusters
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


_HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>cukereuse report</title>
<style>
 body{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:#1a1a1a;max-width:1100px;margin:2em auto;padding:0 1em}
 h1{margin-bottom:.2em}
 .sub{color:#666;margin-top:0}
 .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.6em;margin:1em 0 2em}
 .stat{background:#f6f8fa;border:1px solid #e1e4e8;border-radius:6px;padding:.6em .8em}
 .stat .k{font-size:11px;text-transform:uppercase;color:#666;letter-spacing:.03em}
 .stat .v{font-size:20px;font-weight:600;margin-top:.1em}
 details{border:1px solid #e1e4e8;border-radius:6px;margin-bottom:.4em;background:#fff}
 details>summary{cursor:pointer;padding:.6em .9em;list-style:none;display:flex;align-items:center;gap:.8em}
 details>summary::-webkit-details-marker{display:none}
 summary .count{background:#2a6f4b;color:#fff;border-radius:10px;padding:1px 8px;font-size:12px;min-width:3em;text-align:center}
 summary .filecnt{background:#e1e4e8;color:#333;border-radius:10px;padding:1px 8px;font-size:11px}
 summary .text{font-family:"SF Mono",Menlo,Consolas,monospace;flex:1;word-break:break-word}
 .members{padding:0 1em 1em 1em;font-family:"SF Mono",Menlo,Consolas,monospace;font-size:12px;color:#333}
 .members table{border-collapse:collapse;width:100%}
 .members td{padding:2px 6px;border-bottom:1px solid #f0f0f0;vertical-align:top}
 .members td.file{color:#0550ae}
 .members td.line{color:#666;text-align:right;width:4em}
 .members td.kw{color:#9e3c85;width:4.5em}
 .bg{color:#6a737d;font-style:italic}
 .ol{color:#c5563c;font-style:italic}
 .footer{color:#888;font-size:12px;margin-top:2em}
</style>
</head>
<body>
"""


def write_html(report: Report, path: Path, *, top_n: int | None = 500) -> None:
    """Emit a single-file HTML report — inline CSS, collapsible cluster cards."""
    path.parent.mkdir(parents=True, exist_ok=True)
    esc = html.escape

    clusters = list(report.clusters)
    shown_note = ""
    if top_n is not None and len(clusters) > top_n:
        shown_note = (
            f'<p class="sub">Showing top {top_n} of {len(clusters):,} clusters (by count). '
            f"Full data in the JSON sibling report.</p>"
        )
        clusters = clusters[:top_n]

    parts = [_HTML_HEAD]
    parts.append("<h1>cukereuse report</h1>")
    parts.append(
        f'<p class="sub">Root: <code>{esc(str(report.root_path))}</code> · '
        f"Strategy: <b>{esc(report.strategy)}</b> · "
        f"Generated {esc(report.generated_at.isoformat(timespec='seconds'))}</p>"
    )

    parts.append('<div class="stats">')
    for label, value in [
        ("feature files", f"{report.n_feature_files:,}"),
        ("parse errors", f"{report.n_parse_errors:,}"),
        ("total steps", f"{report.n_steps:,}"),
        ("unique step texts", f"{report.n_unique_step_texts:,}"),
        ("duplicate clusters", f"{report.n_duplicate_clusters:,}"),
        ("exact dup rate", f"{report.exact_duplication_rate * 100:.1f}%"),
    ]:
        parts.append(
            f'<div class="stat"><div class="k">{esc(label)}</div><div class="v">{esc(value)}</div></div>'
        )
    parts.append("</div>")

    parts.append(shown_note)
    parts.append("<h2>Top duplicate clusters</h2>")

    for cluster in clusters:
        parts.append("<details>")
        parts.append(
            f'<summary><span class="count">{cluster.count}</span>'
            f'<span class="filecnt">{cluster.occurrence_files} files</span>'
            f'<span class="text">{esc(cluster.canonical_text)}</span></summary>'
        )
        parts.append('<div class="members"><table>')
        for m in cluster.members:
            kw_class = "kw" + (" bg" if m.is_background else "") + (" ol" if m.is_outline else "")
            parts.append(
                "<tr>"
                f'<td class="{kw_class}">{esc(m.keyword)}</td>'
                f'<td class="file">{esc(str(m.file_path))}</td>'
                f'<td class="line">{m.line}</td>'
                "</tr>"
            )
        parts.append("</table></div></details>")

    parts.append(
        '<p class="footer">Generated by <a href="https://github.com/amughalbscs16/cukereuse">'
        "cukereuse</a>. Only exact-match duplicates are shown — semantic (SBERT) clustering "
        "layers on top in a later release.</p>"
    )
    parts.append("</body></html>")

    path.write_text("".join(parts), encoding="utf-8")
