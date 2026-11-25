"""Clean ANSI terminal summaries and machine-readable JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from tinker_inspect.schema import EvalDiffSummary, RunData, RunDiagnostics, Severity

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
}

_SEV_STYLE: dict[Severity, tuple[str, str]] = {
    "pass": ("PASS", "green"),
    "info": ("INFO", "cyan"),
    "warn": ("WARN", "yellow"),
    "fail": ("FAIL", "red"),
}


def _c(text: str, style: str, *, color: bool) -> str:
    if not color:
        return text
    return f"{_ANSI.get(style, '')}{text}{_ANSI['reset']}"


def _fmt_int(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


def render_run_terminal(
    run: RunData,
    diag: RunDiagnostics,
    *,
    eval_diff: EvalDiffSummary | None = None,
    color: bool = True,
) -> str:
    lines: list[str] = []
    header = f"tinker-inspect :: {diag.run_name}"
    if diag.model_name:
        header += f"  ({diag.model_name})"
    lines.append(_c(header, "bold", color=color))
    meta_parts = [f"steps={diag.num_steps}"]
    if diag.total_wall_seconds > 0:
        meta_parts.append(f"wall={_fmt_duration(diag.total_wall_seconds)}")
    if run.config.get("learning_rate") is not None:
        meta_parts.append(f"lr={run.config['learning_rate']}")
    if run.config.get("lora_rank") is not None:
        meta_parts.append(f"lora_rank={run.config['lora_rank']}")
    if run.checkpoints:
        meta_parts.append(f"checkpoints={len(run.checkpoints)}")
    lines.append(_c("  " + "  |  ".join(meta_parts), "dim", color=color))
    lines.append("")

    # Health checks
    lines.append(_c("Health Checks", "bold", color=color))
    for f in diag.findings:
        tag, sty = _SEV_STYLE[f.severity]
        badge = _c(f"[{tag:4s}]", sty, color=color)
        lines.append(f"  {badge} {f.title}: {f.headline}")
    lines.append("")

    # Tokens & Cost
    t = diag.tokens
    lines.append(_c(f"Tokens & Cost ({t.source})", "bold", color=color))
    prefill_str = f"prefill={_fmt_int(t.prefill_total)}"
    if t.prefill_multiplier is not None:
        prefill_str += (
            f" ({t.prefill_multiplier:.2f}x unique {_fmt_int(t.unique_trajectory_tokens)})"
        )
    if t.cacheable_fraction is not None:
        prefill_str += f"  cacheable={t.cacheable_fraction:.0%}"
    lines.append(f"  {prefill_str}")
    lines.append(
        f"  sampled={_fmt_int(t.sampled_total)}  trained={_fmt_int(t.trained_total)}"
        + (
            f"  discarded_sampling={t.discarded_token_fraction:.0%}"
            if t.discarded_token_fraction is not None
            else ""
        )
    )
    if t.estimated_cost_usd is not None:
        cost_line = f"  estimated_usd=${t.estimated_cost_usd:.2f}"
        if t.estimated_discarded_cost_usd is not None:
            cost_line += f"  (discarded=${t.estimated_discarded_cost_usd:.2f})"
        if t.pricing_source:
            cost_line += f"  [{t.pricing_source}]"
        lines.append(cost_line)

    # Multi-turn tool & cache breakdown
    if diag.tools is not None:
        td = diag.tools
        tool_parts: list[str] = []
        if td.mean_turns_per_episode is not None:
            tool_parts.append(f"turns/ep={td.mean_turns_per_episode:.2f}")
        if td.used_tool_rate is not None:
            tool_parts.append(f"used_tool={td.used_tool_rate:.0%}")
        if td.mean_tool_calls is not None:
            tool_parts.append(f"calls/ep={td.mean_tool_calls:.2f}")
        if td.mean_tool_errors is not None:
            tool_parts.append(f"errors/ep={td.mean_tool_errors:.2f}")
        if td.hit_max_turns_rate is not None:
            tool_parts.append(f"max_turns={td.hit_max_turns_rate:.1%}")
        if td.hit_max_tokens_rate is not None:
            tool_parts.append(f"max_tokens={td.hit_max_tokens_rate:.1%}")
        cache_pts = [
            r["cache_hit_frac"]
            for r in td.per_step_cache_hit
            if r.get("cache_hit_frac") is not None
        ]
        if cache_pts:
            tool_parts.append(f"cache_hit={cache_pts[0]:.0%}->{cache_pts[-1]:.0%}")
        if tool_parts:
            lines.append("")
            lines.append(_c("Multi-Turn Tool & Cache", "bold", color=color))
            lines.append("  " + "  |  ".join(tool_parts))

    # First vs Last step summary
    if run.steps:
        first_s, last_s = run.steps[0], run.steps[-1]
        lines.append("")
        lines.append(_c("Training Progress (step 0 -> final)", "bold", color=color))
        if first_s.reward_total is not None and last_s.reward_total is not None:
            lines.append(
                f"  reward/total      {first_s.reward_total:.3f} -> {last_s.reward_total:.3f} "
                f"({last_s.reward_total - first_s.reward_total:+.3f})"
            )
        for key in ("outcome", "outcome_facts", "success", "success_facts"):
            if key in first_s.env_metrics and key in last_s.env_metrics:
                b, e = first_s.env_metrics[key], last_s.env_metrics[key]
                lines.append(f"  {key:<17s} {b:.3f} -> {e:.3f} ({e - b:+.3f})")
        if first_s.ac_tokens_per_turn is not None and last_s.ac_tokens_per_turn is not None:
            lines.append(
                f"  ac_tokens/turn    {first_s.ac_tokens_per_turn:.1f} -> "
                f"{last_s.ac_tokens_per_turn:.1f}"
            )

    if eval_diff is not None:
        lines.append("")
        lines.append(render_eval_diff_terminal(eval_diff, color=color))

    if run.warnings:
        lines.append("")
        lines.append(_c(f"Loader Warnings ({len(run.warnings)})", "yellow", color=color))
        for w in run.warnings[:5]:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def render_eval_diff_terminal(diff: EvalDiffSummary, *, color: bool = True) -> str:
    lines: list[str] = []
    lines.append(
        _c(
            f"Eval Diff :: {diff.base_label} -> {diff.final_label} ({diff.num_tasks} paired tasks)",
            "bold",
            color=color,
        )
    )
    lines.append(
        f"  {'metric':<18s} {'base':>7s} {'final':>7s} "
        f"{'delta':>8s}  {'95% CI':>18s}  {'W/R/T':>11s}"
    )
    for md in diff.metric_deltas:
        ci_str = f"[{md.ci_low:+.3f}, {md.ci_high:+.3f}]"
        wrt = f"{md.wins}/{md.regressions}/{md.ties}"
        delta_str = f"{md.delta:+.3f}"
        if md.ci_low > 0:
            delta_str = _c(delta_str, "green", color=color)
        elif md.ci_high < 0:
            delta_str = _c(delta_str, "red", color=color)
        lines.append(
            f"  {md.metric:<18s} {md.base_mean:7.3f} {md.final_mean:7.3f} "
            f"{delta_str:>8s}  {ci_str:>18s}  {wrt:>11s}"
        )
    extras: list[str] = []
    if diff.answer_chars_change_pct is not None:
        extras.append(f"answer_chars={diff.answer_chars_change_pct:+.1f}%")
    if diff.turns_change_pct is not None:
        extras.append(f"turns={diff.turns_change_pct:+.1f}%")
    if diff.base_failed or diff.final_failed:
        extras.append(f"failed={diff.base_failed}->{diff.final_failed}")
    if extras:
        lines.append("  " + "  |  ".join(extras))
    return "\n".join(lines)


def render_compare_terminal(
    run_a: RunData,
    diag_a: RunDiagnostics,
    run_b: RunData,
    diag_b: RunDiagnostics,
    *,
    color: bool = True,
) -> str:
    def _row(label: str, va: str, vb: str) -> str:
        return f"  {label:<26s} {va:>18s} {vb:>18s}"

    def _opt(val: float | None, fmt: str) -> str:
        return format(val, fmt) if val is not None else "n/a"

    a_last = run_a.steps[-1].reward_total if run_a.steps else None
    b_last = run_b.steps[-1].reward_total if run_b.steps else None
    a_strag = (
        f"{diag_a.median_straggler_ratio:.2f}x"
        if diag_a.median_straggler_ratio is not None
        else "n/a"
    )
    b_strag = (
        f"{diag_b.median_straggler_ratio:.2f}x"
        if diag_b.median_straggler_ratio is not None
        else "n/a"
    )
    lines = [
        _c(f"Run Comparison :: {diag_a.run_name}  vs  {diag_b.run_name}", "bold", color=color),
        _row("metric", diag_a.run_name[:18], diag_b.run_name[:18]),
        _row("steps", str(diag_a.num_steps), str(diag_b.num_steps)),
        _row(
            "wall time",
            _fmt_duration(diag_a.total_wall_seconds),
            _fmt_duration(diag_b.total_wall_seconds),
        ),
        _row("max |KL v1|", _opt(diag_a.max_abs_kl_v1, ".2e"), _opt(diag_b.max_abs_kl_v1, ".2e")),
        _row(
            "dropped groups",
            _opt(diag_a.discarded_group_fraction, ".0%"),
            _opt(diag_b.discarded_group_fraction, ".0%"),
        ),
        _row(
            "prefill tokens",
            _fmt_int(diag_a.tokens.prefill_total),
            _fmt_int(diag_b.tokens.prefill_total),
        ),
        _row(
            "sampled tokens",
            _fmt_int(diag_a.tokens.sampled_total),
            _fmt_int(diag_b.tokens.sampled_total),
        ),
        _row("median straggler ratio", a_strag, b_strag),
        _row("final reward/total", _opt(a_last, ".3f"), _opt(b_last, ".3f")),
    ]
    return "\n".join(lines)


def diagnostics_to_json(
    diag: RunDiagnostics,
    *,
    eval_diff: EvalDiffSummary | None = None,
) -> str:
    payload: dict[str, Any] = {"diagnostics": asdict(diag)}
    if eval_diff is not None:
        payload["eval_diff"] = {
            "base_label": eval_diff.base_label,
            "final_label": eval_diff.final_label,
            "num_tasks": eval_diff.num_tasks,
            "base_total_episodes": eval_diff.base_total_episodes,
            "final_total_episodes": eval_diff.final_total_episodes,
            "base_failed": eval_diff.base_failed,
            "final_failed": eval_diff.final_failed,
            "primary_metric": eval_diff.primary_metric,
            "answer_chars_change_pct": eval_diff.answer_chars_change_pct,
            "turns_change_pct": eval_diff.turns_change_pct,
            "metric_deltas": [asdict(m) for m in eval_diff.metric_deltas],
        }
    return json.dumps(payload, indent=2)


def has_failures(diag: RunDiagnostics, *, fail_on_warn: bool = False) -> bool:
    for f in diag.findings:
        if f.severity == "fail":
            return True
        if fail_on_warn and f.severity == "warn":
            return True
    return False
