"""Multi-checkpoint / multi-method paired evaluation progression analyzer and HTML viewer."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from tinker_inspect.schema import EvalEpisode

_PRIMARY_METRIC_PRIORITY = ("success", "outcome", "correct", "reward")
_HOME_PATH_RE = re.compile(r"(/Users/[^/\s\"']+|/home/[^/\s\"']+|[A-Za-z]:\\Users\\[^\\\s\"']+)")

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
}


def _c(text: str, style: str, *, color: bool) -> str:
    if not color:
        return text
    return f"{_ANSI.get(style, '')}{text}{_ANSI['reset']}"


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _HOME_PATH_RE.sub("~", value)
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    return value


def _safe_json_for_script(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return (
        raw.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _bootstrap_ci_mean_diff(
    diffs: list[float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (diffs[0], diffs[0])
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        samples.append(total / n)
    samples.sort()
    lo_idx = max(0, int(0.025 * n_boot))
    hi_idx = min(n_boot - 1, int(0.975 * n_boot))
    return (samples[lo_idx], samples[hi_idx])


def _episode_metrics(ep: EvalEpisode) -> dict[str, float]:
    out = dict(ep.metrics)
    out.setdefault("reward", ep.reward)
    return out


def _episode_output_tokens(ep: EvalEpisode) -> float | None:
    if ep.output_tokens is not None:
        return float(ep.output_tokens)
    val = ep.metrics.get("output_tokens")
    return float(val) if val is not None else None


def _episode_used_tool(ep: EvalEpisode) -> float | None:
    if "used_python" in ep.metrics:
        return float(ep.metrics["used_python"])
    if "used_tool" in ep.metrics:
        return float(ep.metrics["used_tool"])
    if ep.messages:
        return 1.0 if any(bool(m.tool_calls) for m in ep.messages) else 0.0
    return None


@dataclass
class SeriesPoint:
    label: str
    num_episodes: int
    failed_episodes: int
    primary_mean: float
    delta_vs_base: float
    ci_low: float
    ci_high: float
    wins_vs_base: int
    regressions_vs_base: int
    ties_vs_base: int
    metric_means: dict[str, float]
    mean_output_tokens: float | None
    mean_turns: float
    mean_answer_chars: float
    used_tool_rate: float | None
    mean_tool_errors: float | None


@dataclass
class SeriesTaskRow:
    task_key: str
    task_id: str | int | None
    prompt: str
    scores: list[float]
    deltas_vs_base: list[float]
    final_delta: float
    flipped_solved: bool
    flipped_regressed: bool


@dataclass
class EvalSeriesSummary:
    title: str
    primary_metric: str
    num_tasks: int
    points: list[SeriesPoint]
    tasks: list[SeriesTaskRow]


def _group_episodes(
    runs: Sequence[tuple[str, list[EvalEpisode]]],
    *,
    by_prompt: bool,
) -> tuple[list[dict[str, list[EvalEpisode]]], list[str]]:
    grouped: list[dict[str, list[EvalEpisode]]] = []
    for _, episodes in runs:
        bucket: dict[str, list[EvalEpisode]] = defaultdict(list)
        for ep in episodes:
            key = " ".join(ep.prompt.split()) if by_prompt else ep.task_key
            if key:
                bucket[key].append(ep)
        grouped.append(bucket)

    common_sets = [set(b.keys()) for b in grouped]
    shared = set.intersection(*common_sets) if common_sets else set()
    ordered_keys = [k for k in grouped[0] if k in shared]
    return grouped, ordered_keys


def analyze_eval_series(
    runs: Sequence[tuple[str, list[EvalEpisode]]],
    *,
    primary_metric: str | None = None,
    title: str | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> EvalSeriesSummary:
    """Analyze N >= 2 paired evaluation checkpoints/methods on their exact common task set."""
    if len(runs) < 2:
        raise ValueError(f"analyze_eval_series requires at least 2 runs, got {len(runs)}.")

    grouped, common_keys = _group_episodes(runs, by_prompt=False)
    min_unique = min((len(b) for b in grouped), default=0)
    if not common_keys or (min_unique > 0 and len(common_keys) < 0.5 * min_unique):
        prompt_grouped, prompt_keys = _group_episodes(runs, by_prompt=True)
        if len(prompt_keys) > len(common_keys):
            grouped = prompt_grouped
            common_keys = prompt_keys

    if not common_keys:
        raise ValueError("No common tasks found across evaluation series.")

    # Discover common numeric metrics across all runs (over paired episodes).
    run_metric_sets: list[set[str]] = []
    for bucket in grouped:
        keys_in_run: set[str] = set()
        for k in common_keys:
            for ep in bucket[k]:
                keys_in_run.update(_episode_metrics(ep).keys())
        run_metric_sets.append(keys_in_run)
    common_metrics_set = set.intersection(*run_metric_sets) if run_metric_sets else {"reward"}
    ordered_metrics = [m for m in _PRIMARY_METRIC_PRIORITY if m in common_metrics_set]
    ordered_metrics += sorted(m for m in common_metrics_set if m not in ordered_metrics)

    if primary_metric is not None:
        if primary_metric not in common_metrics_set:
            raise ValueError(
                f"Primary metric {primary_metric!r} not present across all runs. "
                f"Available common metrics: {sorted(common_metrics_set)}"
            )
        chosen_primary = primary_metric
    else:
        chosen_primary = ordered_metrics[0] if ordered_metrics else "reward"

    # Compute per-task primary means for every point (apples-to-apples over exact common_keys).
    task_primary_matrix: list[list[float]] = []
    for bucket in grouped:
        per_task: list[float] = []
        for k in common_keys:
            vals = [_episode_metrics(ep).get(chosen_primary, ep.reward) for ep in bucket[k]]
            per_task.append(_mean(vals))
        task_primary_matrix.append(per_task)

    base_primary = task_primary_matrix[0]

    # Check if tool usage signals exist anywhere across the series.
    any_tool_signal = any(
        any(
            ("used_python" in ep.metrics or "used_tool" in ep.metrics)
            or any(bool(m.tool_calls) for m in ep.messages)
            for k in common_keys
            for ep in bucket[k]
        )
        for bucket in grouped
    )

    points: list[SeriesPoint] = []
    for i, (label, _raw_eps) in enumerate(runs):
        bucket = grouped[i]
        paired_eps = [ep for k in common_keys for ep in bucket[k]]
        cur_primary = task_primary_matrix[i]
        diffs = [c - b for b, c in zip(base_primary, cur_primary, strict=True)]

        if i == 0:
            delta = 0.0
            ci_lo, ci_hi = 0.0, 0.0
            wins, regs, ties = 0, 0, len(common_keys)
        else:
            delta = _mean(diffs)
            ci_lo, ci_hi = _bootstrap_ci_mean_diff(diffs, n_boot=n_boot, seed=seed)
            wins = sum(1 for d in diffs if d > 1e-9)
            regs = sum(1 for d in diffs if d < -1e-9)
            ties = len(diffs) - wins - regs

        metric_means: dict[str, float] = {}
        for m in ordered_metrics:
            per_task_m: list[float] = []
            for k in common_keys:
                mvals = [_episode_metrics(ep)[m] for ep in bucket[k] if m in _episode_metrics(ep)]
                if mvals:
                    per_task_m.append(_mean(mvals))
            if per_task_m:
                metric_means[m] = _mean(per_task_m)

        out_tok_tasks: list[float] = []
        turns_tasks: list[float] = []
        chars_tasks: list[float] = []
        tool_use_tasks: list[float] = []
        tool_err_tasks: list[float] = []

        for k in common_keys:
            eps_k = bucket[k]
            ot_vals = [v for ep in eps_k if (v := _episode_output_tokens(ep)) is not None]
            if ot_vals:
                out_tok_tasks.append(_mean(ot_vals))
            turns_tasks.append(_mean([float(ep.num_turns) for ep in eps_k]))
            chars_tasks.append(_mean([float(ep.final_answer_chars) for ep in eps_k]))
            if any_tool_signal:
                tu_vals = [v for ep in eps_k if (v := _episode_used_tool(ep)) is not None]
                if tu_vals:
                    tool_use_tasks.append(_mean(tu_vals))
            te_vals = [
                float(ep.metrics["tool_errors"]) for ep in eps_k if "tool_errors" in ep.metrics
            ]
            if te_vals:
                tool_err_tasks.append(_mean(te_vals))

        points.append(
            SeriesPoint(
                label=label,
                num_episodes=len(paired_eps),
                failed_episodes=sum(1 for ep in paired_eps if not ep.ok),
                primary_mean=_mean(cur_primary),
                delta_vs_base=delta,
                ci_low=ci_lo,
                ci_high=ci_hi,
                wins_vs_base=wins,
                regressions_vs_base=regs,
                ties_vs_base=ties,
                metric_means=metric_means,
                mean_output_tokens=_mean(out_tok_tasks) if out_tok_tasks else None,
                mean_turns=_mean(turns_tasks),
                mean_answer_chars=_mean(chars_tasks),
                used_tool_rate=_mean(tool_use_tasks) if tool_use_tasks else None,
                mean_tool_errors=_mean(tool_err_tasks) if tool_err_tasks else None,
            )
        )

    tasks: list[SeriesTaskRow] = []
    for t_idx, k in enumerate(common_keys):
        first_ep = grouped[0][k][0]
        prompt = first_ep.prompt
        if not prompt:
            for bucket in grouped[1:]:
                if bucket[k] and bucket[k][0].prompt:
                    prompt = bucket[k][0].prompt
                    break
        scores = [task_primary_matrix[p_idx][t_idx] for p_idx in range(len(runs))]
        deltas = [s - scores[0] for s in scores]
        final_delta = scores[-1] - scores[0]
        tasks.append(
            SeriesTaskRow(
                task_key=first_ep.task_key,
                task_id=first_ep.task_id,
                prompt=prompt,
                scores=scores,
                deltas_vs_base=deltas,
                final_delta=final_delta,
                flipped_solved=(scores[0] < 0.5 and scores[-1] >= 0.5),
                flipped_regressed=(scores[0] >= 0.5 and scores[-1] < 0.5),
            )
        )

    resolved_title = title or f"{runs[0][0]} -> {runs[-1][0]} ({len(runs)} checkpoints)"
    return EvalSeriesSummary(
        title=resolved_title,
        primary_metric=chosen_primary,
        num_tasks=len(common_keys),
        points=points,
        tasks=tasks,
    )


def series_to_json(summary: EvalSeriesSummary) -> str:
    """Serialize an EvalSeriesSummary to formatted JSON."""
    return json.dumps(asdict(summary), indent=2)


def render_series_terminal(summary: EvalSeriesSummary, *, color: bool = True) -> str:
    """Render an aligned ANSI terminal summary table for a paired eval series."""
    has_fmt = any("format_ok" in p.metric_means for p in summary.points)
    has_tok = any(p.mean_output_tokens is not None for p in summary.points)
    has_tool = any(p.used_tool_rate is not None for p in summary.points)
    label_w = max(14, min(32, max(len(p.label) for p in summary.points)))

    lines: list[str] = [
        _c(
            f"Eval Series :: {summary.title} "
            f"({summary.num_tasks} paired tasks, metric={summary.primary_metric})",
            "bold",
            color=color,
        )
    ]

    hdr = (
        f"  {'#':>2s}  {'Label':<{label_w}s}  {'Primary':>7s}  {'Delta':>8s}  "
        f"{'95% CI':>18s}  {'W/R/T':>10s}"
    )
    if has_fmt:
        hdr += f"  {'FormatOK':>8s}"
    if has_tok:
        hdr += f"  {'OutTok':>7s}"
    if has_tool:
        hdr += f"  {'Tool%':>6s}"
    hdr += f"  {'Turns':>5s}"
    lines.append(_c(hdr, "dim", color=color))

    for idx, pt in enumerate(summary.points, start=1):
        lbl = pt.label[:label_w]
        if idx == 1:
            delta_raw = "base"
            delta_str = f"{delta_raw:>8s}"
            ci_str = f"{'—':>18s}"
            wrt_str = f"{'—':>10s}"
        else:
            delta_raw = f"{pt.delta_vs_base:+.3f}"
            delta_padded = f"{delta_raw:>8s}"
            if pt.ci_low > 0:
                delta_str = _c(delta_padded, "green", color=color)
            elif pt.ci_high < 0:
                delta_str = _c(delta_padded, "red", color=color)
            else:
                delta_str = delta_padded
            ci_str = f"[{pt.ci_low:+.3f}, {pt.ci_high:+.3f}]"
            wrt_str = f"{pt.wins_vs_base}/{pt.regressions_vs_base}/{pt.ties_vs_base}"

        row = (
            f"  {idx:>2d}  {lbl:<{label_w}s}  {pt.primary_mean:7.3f}  {delta_str}  "
            f"{ci_str:>18s}  {wrt_str:>10s}"
        )
        if has_fmt:
            fv = pt.metric_means.get("format_ok")
            row += f"  {fv:8.3f}" if fv is not None else f"  {'n/a':>8s}"
        if has_tok:
            ot = pt.mean_output_tokens
            row += f"  {ot:7.1f}" if ot is not None else f"  {'n/a':>7s}"
        if has_tool:
            tr = pt.used_tool_rate
            row += f"  {tr * 100:5.1f}%" if tr is not None else f"  {'n/a':>6s}"
        row += f"  {pt.mean_turns:5.2f}"
        lines.append(row)

    flipped_up = sum(1 for t in summary.tasks if t.flipped_solved)
    flipped_dn = sum(1 for t in summary.tasks if t.flipped_regressed)
    lines.append(
        _c(
            f"  first->last task flips: +{flipped_up} solved, -{flipped_dn} regressed "
            f"(out of {summary.num_tasks} tasks)",
            "dim",
            color=color,
        )
    )
    return "\n".join(lines)


_SERIES_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-2: #f1f5f9;
      --text: #0f172a;
      --muted: #475569;
      --border: #e2e8f0;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
      --pass: #059669;
      --pass-bg: #dcfce7;
      --warn: #d97706;
      --warn-bg: #fef3c7;
      --fail: #dc2626;
      --fail-bg: #fee2e2;
      --shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    @media (prefers-color-scheme: dark) {
      :root:not([data-theme="light"]) {
        --bg: #0b1120;
        --surface: #111827;
        --surface-2: #1e293b;
        --text: #f1f5f9;
        --muted: #94a3b8;
        --border: #1f2937;
        --accent: #60a5fa;
        --accent-soft: #172554;
        --pass: #34d399;
        --pass-bg: #064e3b;
        --warn: #fbbf24;
        --warn-bg: #451a03;
        --fail: #f87171;
        --fail-bg: #450a0a;
        --shadow: 0 1px 3px rgba(0, 0, 0, 0.45);
      }
    }
    :root[data-theme="dark"] {
      --bg: #0b1120;
      --surface: #111827;
      --surface-2: #1e293b;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --border: #1f2937;
      --accent: #60a5fa;
      --accent-soft: #172554;
      --pass: #34d399;
      --pass-bg: #064e3b;
      --warn: #fbbf24;
      --warn-bg: #451a03;
      --fail: #f87171;
      --fail-bg: #450a0a;
      --shadow: 0 1px 3px rgba(0, 0, 0, 0.45);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 1.75rem 1.25rem 3rem; }
    header {
      display: flex; flex-wrap: wrap; align-items: flex-start;
      justify-content: space-between; gap: 1rem; margin-bottom: 1.5rem;
    }
    h1 { margin: 0; font-size: 1.55rem; font-weight: 800; letter-spacing: -0.02em; }
    .sub { margin: 0.25rem 0 0; color: var(--muted); font-size: 0.9rem; }
    .btn {
      padding: 0.45rem 0.85rem; border-radius: 8px; border: 1px solid var(--border);
      background: var(--surface); color: var(--text); font-size: 0.85rem; font-weight: 600;
      cursor: pointer; box-shadow: var(--shadow);
    }
    .btn:hover, .btn.active { border-color: var(--accent); color: var(--accent); }
    .btn.active { background: var(--accent-soft); }
    .kpis {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 0.9rem; margin-bottom: 1.5rem;
    }
    .kpi {
      background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
      padding: 0.95rem 1.05rem; box-shadow: var(--shadow);
    }
    .kpi-label {
      font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.04em;
      color: var(--muted); font-weight: 700;
    }
    .kpi-val { font-size: 1.45rem; font-weight: 800; margin-top: 0.2rem; }
    .kpi-sub { font-size: 0.8rem; color: var(--muted); margin-top: 0.15rem; }
    .charts {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 1rem; margin-bottom: 1.5rem;
    }
    .card {
      background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
      padding: 1.1rem 1.2rem; box-shadow: var(--shadow); margin-bottom: 1.5rem;
    }
    .charts .card { margin-bottom: 0; }
    .card h2 { margin: 0 0 0.75rem; font-size: 1.05rem; font-weight: 700; }
    .svg-box { width: 100%; overflow-x: auto; }
    .svg-box svg { display: block; width: 100%; height: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
    th, td {
      padding: 0.55rem 0.65rem; border-bottom: 1px solid var(--border);
      text-align: left; vertical-align: middle;
    }
    th {
      font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.03em;
      color: var(--muted); font-weight: 700; background: var(--surface-2);
    }
    td.num, th.num {
      text-align: right;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .badge {
      display: inline-block; padding: 0.12rem 0.45rem; border-radius: 999px;
      font-weight: 700; font-size: 0.78rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .badge.pos { background: var(--pass-bg); color: var(--pass); }
    .badge.neg { background: var(--fail-bg); color: var(--fail); }
    .badge.neu { background: var(--surface-2); color: var(--muted); }
    .controls {
      display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-bottom: 0.85rem;
    }
    .search {
      flex: 1 1 220px; padding: 0.45rem 0.7rem; border-radius: 8px;
      border: 1px solid var(--border); background: var(--bg);
      color: var(--text); font-size: 0.86rem;
    }
    .cell {
      display: inline-block; min-width: 44px; text-align: center; padding: 0.16rem 0.35rem;
      border-radius: 6px; font-weight: 700; font-size: 0.78rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .task-row { cursor: pointer; }
    .task-row:hover { background: var(--surface-2); }
    .prompt-preview {
      max-width: 420px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      color: var(--muted); font-size: 0.84rem;
    }
    .prompt-expand {
      background: var(--surface-2); padding: 0.75rem 0.9rem; border-radius: 8px;
      white-space: pre-wrap; font-size: 0.84rem; margin: 0.25rem 0;
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1 id="page-title">Eval Series</h1>
        <p class="sub" id="page-sub"></p>
      </div>
      <button class="btn" id="theme-toggle" type="button">Theme</button>
    </header>

    <div class="kpis" id="kpi-strip"></div>

    <div class="charts">
      <div class="card">
        <h2>Progression Across Checkpoints (95% Bootstrap CI)</h2>
        <div class="svg-box" id="chart-progress"></div>
      </div>
      <div class="card">
        <h2 id="pareto-title">Accuracy vs Efficiency Pareto</h2>
        <div class="svg-box" id="chart-pareto"></div>
      </div>
    </div>

    <div class="card">
      <h2>Progression Metrics Table</h2>
      <div style="overflow-x:auto">
        <table id="metrics-table"></table>
      </div>
    </div>

    <div class="card">
      <h2>Per-Task Trajectory Heatmap</h2>
      <div class="controls">
        <input
          class="search"
          id="task-search"
          type="search"
          placeholder="Filter by task id or prompt text..."
        />
        <button class="btn active" data-filter="all" type="button">All</button>
        <button class="btn" data-filter="solved" type="button">Flipped Solved</button>
        <button class="btn" data-filter="regressed" type="button">Regressed</button>
        <button class="btn" data-filter="hard" type="button">Always Hard</button>
        <button class="btn" data-filter="always_solved" type="button">Always Solved</button>
      </div>
      <div style="overflow-x:auto">
        <table id="heatmap-table"></table>
      </div>
    </div>
  </div>

  <script id="tinker-series-data" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    (function () {
      var dataEl = document.getElementById("tinker-series-data");
      var S = JSON.parse(dataEl.textContent || "{}");
      var pts = S.points || [];
      var tasks = S.tasks || [];

      var root = document.documentElement;
      var themeBtn = document.getElementById("theme-toggle");
      try {
        var saved = localStorage.getItem("tinker-inspect-theme");
        if (saved === "dark" || saved === "light") root.setAttribute("data-theme", saved);
      } catch (e) {}
      function isDark() {
        var attr = root.getAttribute("data-theme");
        if (attr === "dark") return true;
        if (attr === "light") return false;
        var mq = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
        return !!(mq && mq.matches);
      }
      function syncThemeBtn() {
        themeBtn.textContent = isDark() ? "Light mode" : "Dark mode";
      }
      syncThemeBtn();
      themeBtn.addEventListener("click", function () {
        var next = isDark() ? "light" : "dark";
        root.setAttribute("data-theme", next);
        try { localStorage.setItem("tinker-inspect-theme", next); } catch (e) {}
        syncThemeBtn();
        renderCharts();
      });

      function esc(s) {
        return String(s == null ? "" : s)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;");
      }
      function fmtPct(v) { return (v * 100).toFixed(1) + "%"; }
      function fmtDelta(v) { return (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + " pp"; }

      document.getElementById("page-title").textContent = S.title || "Eval Series";
      document.getElementById("page-sub").textContent =
        S.num_tasks + " paired tasks across " + pts.length +
        " checkpoints · primary metric: " + S.primary_metric;

      var basePt = pts[0];
      var bestPt = pts.reduce(function (a, b) {
        return b.primary_mean > a.primary_mean ? b : a;
      }, basePt);
      var lastPt = pts[pts.length - 1];
      var flippedUp = tasks.filter(function (t) { return t.flipped_solved; }).length;
      var flippedDn = tasks.filter(function (t) { return t.flipped_regressed; }).length;

      var turnDelta = (bestPt.mean_turns - basePt.mean_turns) / Math.max(basePt.mean_turns, 1e-6);
      var tokSub = "turns: " + basePt.mean_turns.toFixed(2) + " -> " + bestPt.mean_turns.toFixed(2);
      var tokVal = (turnDelta * 100).toFixed(1) + "% turns";
      if (basePt.mean_output_tokens != null && bestPt.mean_output_tokens != null) {
        var denomT = Math.max(basePt.mean_output_tokens, 1e-6);
        var pctT = ((bestPt.mean_output_tokens - basePt.mean_output_tokens) / denomT) * 100;
        tokVal = (pctT >= 0 ? "+" : "") + pctT.toFixed(1) + "%";
        tokSub =
          basePt.mean_output_tokens.toFixed(0) + " -> " +
          bestPt.mean_output_tokens.toFixed(0) + " out tokens";
      }

      var kpiItems = [
        {
          label: "Paired Tasks",
          val: String(S.num_tasks),
          sub: pts.length + " checkpoints · metric: " + S.primary_metric
        },
        {
          label: "Baseline -> Best (" + bestPt.label + ")",
          val: fmtPct(basePt.primary_mean) + " -> " + fmtPct(bestPt.primary_mean),
          sub: fmtDelta(bestPt.delta_vs_base) +
            " [" + fmtDelta(bestPt.ci_low) + ", " + fmtDelta(bestPt.ci_high) + "]"
        },
        {
          label: "Output Efficiency (Base -> Best)",
          val: tokVal,
          sub: tokSub
        },
        {
          label: "Task Flips (First -> Last)",
          val: "+" + flippedUp + " / -" + flippedDn,
          sub: lastPt.label + " vs " + basePt.label
        }
      ];
      document.getElementById("kpi-strip").innerHTML = kpiItems.map(function (k) {
        return '<div class="kpi"><div class="kpi-label">' + esc(k.label) +
          '</div><div class="kpi-val">' + esc(k.val) +
          '</div><div class="kpi-sub">' + esc(k.sub) + '</div></div>';
      }).join("");

      function renderCharts() {
        var fg = isDark() ? "#cbd5e1" : "#334155";
        var grid = isDark() ? "#1f2937" : "#e2e8f0";
        var bg = isDark() ? "#0f172a" : "#f8fafc";
        var accent = isDark() ? "#60a5fa" : "#2563eb";
        var good = isDark() ? "#34d399" : "#059669";

        var W = 540, H = 280, pL = 52, pR = 32, pT = 28, pB = 54;
        var ys = [];
        pts.forEach(function (p, i) {
          ys.push(p.primary_mean);
          if (i > 0) {
            ys.push(basePt.primary_mean + p.ci_low);
            ys.push(basePt.primary_mean + p.ci_high);
          }
        });
        var yMin = Math.max(0, Math.floor((Math.min.apply(null, ys) - 0.05) * 10) / 10);
        var yMax = Math.min(1.0, Math.ceil((Math.max.apply(null, ys) + 0.06) * 10) / 10);
        if (yMax - yMin < 0.25) {
          yMin = Math.max(0, yMin - 0.1);
          yMax = Math.min(1.0, yMax + 0.1);
        }
        var xPad = 34;
        function xp(i) {
          if (pts.length <= 1) return (pL + W - pR) / 2;
          return (pL + xPad) + (W - pL - pR - 2 * xPad) * (i / (pts.length - 1));
        }
        function yp(v) {
          return pT + (H - pT - pB) * (1 - (v - yMin) / Math.max(yMax - yMin, 1e-6));
        }
        var halo = (
          'paint-order="stroke" stroke="' + bg +
          '" stroke-width="4" stroke-linejoin="round"'
        );
        var svg1 = ['<svg viewBox="0 0 ' + W + ' ' + H + '">'];
        svg1.push(
          '<rect x="' + pL + '" y="' + pT + '" width="' + (W - pL - pR) +
          '" height="' + (H - pT - pB) + '" rx="4" fill="' + bg + '" stroke="' + grid + '"/>'
        );
        for (var g = 0; g <= 4; g++) {
          var v = yMin + (yMax - yMin) * (g / 4);
          var yy = yp(v);
          svg1.push(
            '<line x1="' + pL + '" y1="' + yy.toFixed(1) + '" x2="' + (W - pR) +
            '" y2="' + yy.toFixed(1) + '" stroke="' + grid + '" stroke-dasharray="3,3"/>'
          );
          svg1.push(
            '<text x="' + (pL - 7) + '" y="' + (yy + 4).toFixed(1) +
            '" text-anchor="end" font-size="10.5" fill="' + fg + '">' +
            (v * 100).toFixed(0) + '%</text>'
          );
        }
        var by = yp(basePt.primary_mean);
        svg1.push(
          '<line x1="' + pL + '" y1="' + by.toFixed(1) + '" x2="' + (W - pR) +
          '" y2="' + by.toFixed(1) + '" stroke="#94a3b8" stroke-dasharray="5,4" opacity="0.7"/>'
        );
        // Subtle translucent 95% CI range bands behind line & dots
        pts.forEach(function (p, i) {
          if (i === 0) return;
          var cx = xp(i);
          var lo = yp(Math.max(yMin, basePt.primary_mean + p.ci_low));
          var hi = yp(Math.min(yMax, basePt.primary_mean + p.ci_high));
          var bandCol = p === bestPt ? good : accent;
          svg1.push(
            '<line x1="' + cx.toFixed(1) + '" y1="' + lo.toFixed(1) +
            '" x2="' + cx.toFixed(1) + '" y2="' + hi.toFixed(1) +
            '" stroke="' + bandCol + '" stroke-width="5" stroke-linecap="round" opacity="0.22"/>'
          );
        });
        var linePts = pts.map(function (p, i) {
          return xp(i).toFixed(1) + "," + yp(p.primary_mean).toFixed(1);
        }).join(" ");
        svg1.push(
          '<polyline points="' + linePts + '" fill="none" stroke="' + accent +
          '" stroke-width="2.2" stroke-linejoin="round"/>'
        );
        pts.forEach(function (p, i) {
          var cx = xp(i), cy = yp(p.primary_mean);
          var dotCol = p === bestPt ? good : accent;
          svg1.push(
            '<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) +
            '" r="4.8" fill="' + dotCol + '" stroke="#ffffff" stroke-width="1.5"/>'
          );
          var lblAnchor = i === pts.length - 1 ? "end" : "start";
          var lblDx = i === pts.length - 1 ? -8 : 8;
          var lblDy = cy < pT + 26 ? 15 : -8;
          svg1.push(
            '<text x="' + (cx + lblDx).toFixed(1) + '" y="' + (cy + lblDy).toFixed(1) +
            '" text-anchor="' + lblAnchor + '" font-size="10.5" font-weight="700" fill="' +
            fg + '" ' + halo + '>' + fmtPct(p.primary_mean) + '</text>'
          );
          var shortLbl = p.label.length > 13 ? p.label.slice(0, 12) + "…" : p.label;
          var xLblY = pts.length >= 5 && i % 2 === 1 ? H - 10 : H - 24;
          svg1.push(
            '<text x="' + cx.toFixed(1) + '" y="' + xLblY +
            '" text-anchor="middle" font-size="9.8" fill="' + fg + '">' +
            esc(shortLbl) + '</text>'
          );
        });
        svg1.push(
          '<text x="' + (W - pR - 4) + '" y="' + (pT + 13) +
          '" text-anchor="end" font-size="9.5" fill="' + fg + '" opacity="0.7">' +
          'Shaded bars: 95% paired CI vs base</text>'
        );
        svg1.push("</svg>");
        document.getElementById("chart-progress").innerHTML = svg1.join("");

        var tokPts = pts.filter(function (p) { return p.mean_output_tokens != null; });
        var useTok = tokPts.length >= 2;
        document.getElementById("pareto-title").textContent = useTok
          ? "Accuracy vs Mean Output Tokens (Pareto)"
          : "Accuracy vs Mean Turns (Pareto)";
        var pPts = useTok ? tokPts : pts;
        var xs = pPts.map(function (p) {
          return useTok ? p.mean_output_tokens : p.mean_turns;
        });
        var xMin = Math.min.apply(null, xs), xMax = Math.max.apply(null, xs);
        var xSpan = Math.max(xMax - xMin, 1);
        xMin = Math.max(0, xMin - 0.16 * xSpan);
        xMax = xMax + 0.16 * xSpan;
        function xq(val) {
          return pL + (W - pL - pR) * ((val - xMin) / Math.max(xMax - xMin, 1e-6));
        }

        var svg2 = ['<svg viewBox="0 0 ' + W + ' ' + H + '">'];
        svg2.push(
          '<rect x="' + pL + '" y="' + pT + '" width="' + (W - pL - pR) +
          '" height="' + (H - pT - pB) + '" rx="4" fill="' + bg + '" stroke="' + grid + '"/>'
        );
        for (var g2 = 0; g2 <= 4; g2++) {
          var v2 = yMin + (yMax - yMin) * (g2 / 4);
          var yy2 = yp(v2);
          svg2.push(
            '<line x1="' + pL + '" y1="' + yy2.toFixed(1) + '" x2="' + (W - pR) +
            '" y2="' + yy2.toFixed(1) + '" stroke="' + grid + '" stroke-dasharray="3,3"/>'
          );
          svg2.push(
            '<text x="' + (pL - 7) + '" y="' + (yy2 + 4).toFixed(1) +
            '" text-anchor="end" font-size="10.5" fill="' + fg + '">' +
            (v2 * 100).toFixed(0) + '%</text>'
          );
          var xv = xMin + (xMax - xMin) * (g2 / 4);
          var xx = xq(xv);
          svg2.push(
            '<line x1="' + xx.toFixed(1) + '" y1="' + pT + '" x2="' + xx.toFixed(1) +
            '" y2="' + (H - pB) + '" stroke="' + grid + '" stroke-dasharray="3,3"/>'
          );
          svg2.push(
            '<text x="' + xx.toFixed(1) + '" y="' + (H - pB + 16) +
            '" text-anchor="middle" font-size="10.5" fill="' + fg + '">' +
            (useTok ? xv.toFixed(0) : xv.toFixed(1)) + '</text>'
          );
        }
        var sortedP = pPts.slice().sort(function (a, b) {
          var xa = useTok ? a.mean_output_tokens : a.mean_turns;
          var xb = useTok ? b.mean_output_tokens : b.mean_turns;
          return xa !== xb ? xa - xb : b.primary_mean - a.primary_mean;
        });
        var frontier = [], maxY = -1;
        sortedP.forEach(function (p) {
          if (p.primary_mean > maxY + 1e-9) { frontier.push(p); maxY = p.primary_mean; }
        });
        if (frontier.length >= 2) {
          var fStr = frontier.map(function (p) {
            var xval = useTok ? p.mean_output_tokens : p.mean_turns;
            return xq(xval).toFixed(1) + "," + yp(p.primary_mean).toFixed(1);
          }).join(" ");
          svg2.push(
            '<polyline points="' + fStr + '" fill="none" stroke="' + good +
            '" stroke-width="2" stroke-dasharray="5,3" opacity="0.8"/>'
          );
        }
        var colors = ["#64748b", "#2563eb", "#7c3aed", "#059669", "#d97706", "#db2777"];
        var placedBoxes = [];
        function boxesOverlap(a, b) {
          return !(a.x2 < b.x1 || a.x1 > b.x2 || a.y2 < b.y1 || a.y1 > b.y2);
        }
        function inPlot(box) {
          return (
            box.x1 >= pL + 2 && box.x2 <= W - pR - 2 &&
            box.y1 >= pT + 2 && box.y2 <= H - pB - 2
          );
        }
        // Reserve small boxes around every dot so labels never cover dots
        pPts.forEach(function (p) {
          var xval = useTok ? p.mean_output_tokens : p.mean_turns;
          var dx0 = xq(xval), dy0 = yp(p.primary_mean);
          placedBoxes.push({ x1: dx0 - 7, y1: dy0 - 7, x2: dx0 + 7, y2: dy0 + 7 });
        });
        var candidateOffsets = [
          [-10, -4, "end"],
          [10, -4, "start"],
          [-10, 14, "end"],
          [10, 14, "start"],
          [-10, -18, "end"],
          [10, -18, "start"],
          [-10, 26, "end"],
          [10, 26, "start"],
          [0, -14, "middle"],
          [0, 20, "middle"],
          [-14, -30, "end"],
          [-14, 34, "end"],
        ];
        pPts.forEach(function (p, i) {
          var xval = useTok ? p.mean_output_tokens : p.mean_turns;
          var cx = xq(xval), cy = yp(p.primary_mean);
          var col = colors[i % colors.length];
          svg2.push(
            '<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) +
            '" r="5" fill="' + col + '" stroke="#ffffff" stroke-width="1.5"/>'
          );
          var tw = p.label.length * 6.2 + 6;
          var th = 14;
          var chosen = null;
          for (var c = 0; c < candidateOffsets.length; c++) {
            var cand = candidateOffsets[c];
            var lx = cx + cand[0], ly = cy + cand[1], anc = cand[2];
            var x1 = anc === "start" ? lx : (anc === "end" ? lx - tw : lx - tw / 2);
            var box = { x1: x1 - 2, y1: ly - th + 2, x2: x1 + tw + 2, y2: ly + 3 };
            if (!inPlot(box)) continue;
            var hit = false;
            for (var k = 0; k < placedBoxes.length; k++) {
              if (boxesOverlap(box, placedBoxes[k])) { hit = true; break; }
            }
            if (!hit) { chosen = { lx: lx, ly: ly, anc: anc, box: box }; break; }
          }
          if (!chosen) {
            var fallAnc = cx > W - pR - 110 ? "end" : "start";
            var fallLx = cx + (fallAnc === "end" ? -10 : 10);
            var fallLy = Math.max(pT + 14, Math.min(H - pB - 6, cy + (i % 2 === 0 ? -12 : 16)));
            chosen = {
              lx: fallLx,
              ly: fallLy,
              anc: fallAnc,
              box: { x1: fallLx - tw, y1: fallLy - th, x2: fallLx + tw, y2: fallLy + 3 },
            };
          }
          placedBoxes.push(chosen.box);
          var dist = Math.hypot(chosen.lx - cx, (chosen.ly - 4) - cy);
          if (dist > 18) {
            svg2.push(
              '<line x1="' + cx.toFixed(1) + '" y1="' + cy.toFixed(1) +
              '" x2="' + chosen.lx.toFixed(1) + '" y2="' + (chosen.ly - 3).toFixed(1) +
              '" stroke="' + col + '" stroke-width="1" opacity="0.45"/>'
            );
          }
          svg2.push(
            '<text x="' + chosen.lx.toFixed(1) + '" y="' + chosen.ly.toFixed(1) +
            '" text-anchor="' + chosen.anc + '" font-size="10.2" font-weight="600" fill="' +
            col + '" ' + halo + '>' + esc(p.label) + '</text>'
          );
        });
        var xAxisLbl = useTok ? "Mean Output Tokens (lower is faster)" : "Mean Turns";
        svg2.push(
          '<text x="' + ((pL + W - pR) / 2).toFixed(1) + '" y="' + (H - 8) +
          '" text-anchor="middle" font-size="10.5" fill="' + fg + '">' + xAxisLbl + '</text>'
        );
        svg2.push("</svg>");
        document.getElementById("chart-pareto").innerHTML = svg2.join("");
      }
      renderCharts();

      var hasFmt = pts.some(function (p) {
        return p.metric_means && p.metric_means.format_ok != null;
      });
      var hasTok = pts.some(function (p) { return p.mean_output_tokens != null; });
      var hasTool = pts.some(function (p) { return p.used_tool_rate != null; });
      var hasErr = pts.some(function (p) { return p.mean_tool_errors != null; });

      var thead =
        "<thead><tr><th>#</th><th>Checkpoint / Method</th><th class='num'>" +
        esc(S.primary_metric) +
        "</th><th class='num'>Delta vs Base</th><th class='num'>95% CI</th>" +
        "<th class='num'>W/R/T</th>" +
        (hasFmt ? "<th class='num'>Format OK</th>" : "") +
        (hasTok ? "<th class='num'>Out Tokens</th>" : "") +
        (hasTool ? "<th class='num'>Tool Rate</th>" : "") +
        (hasErr ? "<th class='num'>Tool Err</th>" : "") +
        "<th class='num'>Turns</th></tr></thead>";

      var tbody = "<tbody>" + pts.map(function (p, i) {
        var deltaCell = "<span class='badge neu'>base</span>";
        var ciCell = "—";
        var wrtCell = "—";
        if (i > 0) {
          var cls = p.ci_low > 0 ? "pos" : (p.ci_high < 0 ? "neg" : "neu");
          deltaCell = "<span class='badge " + cls + "'>" + fmtDelta(p.delta_vs_base) + "</span>";
          ciCell = "[" + fmtDelta(p.ci_low) + ", " + fmtDelta(p.ci_high) + "]";
          wrtCell = p.wins_vs_base + "/" + p.regressions_vs_base + "/" + p.ties_vs_base;
        }
        var fmtVal = p.metric_means && p.metric_means.format_ok != null
          ? fmtPct(p.metric_means.format_ok) : "—";
        var tokCell = p.mean_output_tokens != null ? p.mean_output_tokens.toFixed(1) : "—";
        var toolCell = p.used_tool_rate != null ? fmtPct(p.used_tool_rate) : "—";
        var errCell = p.mean_tool_errors != null ? p.mean_tool_errors.toFixed(2) : "—";
        return "<tr><td class='num'>" + (i + 1) + "</td><td><strong>" + esc(p.label) +
          "</strong></td>" +
          "<td class='num'>" + fmtPct(p.primary_mean) + "</td>" +
          "<td class='num'>" + deltaCell + "</td>" +
          "<td class='num'>" + ciCell + "</td>" +
          "<td class='num'>" + wrtCell + "</td>" +
          (hasFmt ? "<td class='num'>" + fmtVal + "</td>" : "") +
          (hasTok ? "<td class='num'>" + tokCell + "</td>" : "") +
          (hasTool ? "<td class='num'>" + toolCell + "</td>" : "") +
          (hasErr ? "<td class='num'>" + errCell + "</td>" : "") +
          "<td class='num'>" + p.mean_turns.toFixed(2) + "</td></tr>";
      }).join("") + "</tbody>";
      document.getElementById("metrics-table").innerHTML = thead + tbody;

      var activeFilter = "all";
      var expanded = {};
      var searchInput = document.getElementById("task-search");

      function scoreStyle(v) {
        if (v >= 0.99) return "background:var(--pass-bg);color:var(--pass);";
        if (v <= 0.01) return "background:var(--fail-bg);color:var(--fail);";
        return "background:var(--warn-bg);color:var(--warn);";
      }

      function renderHeatmap() {
        var q = (searchInput.value || "").trim().toLowerCase();
        var filtered = tasks.filter(function (t) {
          if (activeFilter === "solved" && !t.flipped_solved) return false;
          if (activeFilter === "regressed" && !t.flipped_regressed) return false;
          if (activeFilter === "hard") {
            if (!t.scores.every(function (s) { return s < 0.5; })) return false;
          }
          if (activeFilter === "always_solved") {
            if (!t.scores.every(function (s) { return s >= 0.5; })) return false;
          }
          if (q) {
            var hay = (String(t.task_id || t.task_key) + " " + (t.prompt || "")).toLowerCase();
            if (hay.indexOf(q) === -1) return false;
          }
          return true;
        });

        var hHead = "<thead><tr><th>Task</th><th>Prompt</th>" +
          pts.map(function (p) { return "<th class='num'>" + esc(p.label) + "</th>"; }).join("") +
          "<th class='num'>Final &Delta;</th></tr></thead>";

        var colSpan = pts.length + 3;
        var hRows = filtered.map(function (t) {
          var key = t.task_key;
          var dCls = t.final_delta > 1e-6 ? "pos" : (t.final_delta < -1e-6 ? "neg" : "neu");
          var dTxt = (t.final_delta >= 0 ? "+" : "") + t.final_delta.toFixed(2);
          var scoreCells = t.scores.map(function (s) {
            return "<td class='num'><span class='cell' style='" + scoreStyle(s) + "'>" +
              s.toFixed(2) + "</span></td>";
          }).join("");
          var mainRow = "<tr class='task-row' data-tk='" + esc(key) + "'>" +
            "<td class='mono'>" + esc(t.task_id != null ? t.task_id : t.task_key) + "</td>" +
            "<td><div class='prompt-preview'>" + esc(t.prompt) + "</div></td>" +
            scoreCells +
            "<td class='num'><span class='badge " + dCls + "'>" + dTxt + "</span></td></tr>";
          if (!expanded[key]) return mainRow;
          return mainRow + "<tr><td colspan='" + colSpan + "'><div class='prompt-expand'>" +
            esc(t.prompt) + "</div></td></tr>";
        }).join("");

        var emptyRow = "<tr><td colspan='" + colSpan +
          "' style='text-align:center;color:var(--muted)'>No matching tasks</td></tr>";
        var tbl = document.getElementById("heatmap-table");
        tbl.innerHTML = hHead + "<tbody>" + (hRows || emptyRow) + "</tbody>";
        tbl.querySelectorAll("tr.task-row").forEach(function (tr) {
          tr.addEventListener("click", function () {
            var tk = tr.getAttribute("data-tk");
            expanded[tk] = !expanded[tk];
            renderHeatmap();
          });
        });
      }

      document.querySelectorAll("[data-filter]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          activeFilter = btn.getAttribute("data-filter");
          document.querySelectorAll("[data-filter]").forEach(function (b) {
            b.classList.remove("active");
          });
          btn.classList.add("active");
          renderHeatmap();
        });
      });
      searchInput.addEventListener("input", renderHeatmap);
      renderHeatmap();
    })();
  </script>
</body>
</html>
"""


def build_series_html(summary: EvalSeriesSummary, *, sanitize_paths: bool = False) -> str:
    """Build a self-contained, zero-CDN HTML report for an evaluation progression series."""
    payload = asdict(summary)
    if sanitize_paths:
        payload = _sanitize_value(payload)
    safe_title = (
        str(payload.get("title", "Eval Series"))
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return _SERIES_HTML_TEMPLATE.replace(
        "__TITLE__", f"tinker-inspect series — {safe_title}"
    ).replace("__PAYLOAD_JSON__", _safe_json_for_script(payload))
