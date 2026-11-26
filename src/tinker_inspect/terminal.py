"""Clean ANSI terminal summaries and machine-readable JSON serialization."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from typing import Any

from tinker_inspect.diagnostics import is_outcome_metric
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

_END_STATE_ORDER = ("completed", "token_limit", "turn_limit", "parse_error", "error")


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


def _fmt_end_states(
    counts: dict[str, int],
    success_by_state: dict[str, tuple[float, int]] | None = None,
) -> str:
    if not counts:
        return "n/a"
    total = sum(counts.values())
    if total <= 0:
        return "n/a"
    order = ("completed", "token_limit", "turn_limit", "parse_error", "ungraded", "error")
    ordered = [k for k in order if counts.get(k, 0) > 0]
    ordered += sorted(k for k in counts if k not in order and counts[k] > 0)
    parts: list[str] = []
    for k in ordered:
        n = counts[k]
        pct = n / total
        if success_by_state and k in success_by_state:
            s_rate, _ = success_by_state[k]
            parts.append(f"{k}={n} ({pct:.0%}, succ={s_rate:.0%})")
        else:
            parts.append(f"{k}={n} ({pct:.0%})")
    return ", ".join(parts)


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
        prefill_str += f"  {t.cache_metric_label}={t.cacheable_fraction:.0%}"
    lines.append(f"  {prefill_str}")
    discard_parts: list[str] = []
    if t.discarded_token_fraction is not None:
        discard_parts.append(f"discarded_sampling={t.discarded_token_fraction:.0%}")
    if t.discarded_sampled_only_fraction is not None:
        discard_parts.append(f"discarded_action_only={t.discarded_sampled_only_fraction:.0%}")
    discard_suffix = ("  " + "  ".join(discard_parts)) if discard_parts else ""
    lines.append(
        f"  sampled={_fmt_int(t.sampled_total)}  trained={_fmt_int(t.trained_total)}"
        f"{discard_suffix}"
    )
    if t.estimated_cost_usd is not None:
        cost_line = f"  estimated_usd=${t.estimated_cost_usd:.2f}"
        if t.estimated_discarded_cost_usd is not None:
            cost_line += f"  (discarded=${t.estimated_discarded_cost_usd:.2f})"
        if t.pricing_source:
            cost_line += f"  [{t.pricing_source}]"
        lines.append(cost_line)

    # Timing & Straggler Idle Breakdown
    timing_parts: list[str] = []
    if diag.sampling_idle_fraction is not None:
        timing_parts.append(f"idle_wait={diag.sampling_idle_fraction:.0%}")
    if diag.median_straggler_ratio is not None:
        timing_parts.append(f"median_straggler={diag.median_straggler_ratio:.2f}x")
    if diag.policy_sample_mean_s is not None:
        timing_parts.append(f"policy_sample={diag.policy_sample_mean_s:.2f}s")
    if diag.env_step_mean_s is not None:
        timing_parts.append(f"env_step={diag.env_step_mean_s:.2f}s")
    if timing_parts:
        lines.append("")
        lines.append(_c("Sampling Timing & Stragglers", "bold", color=color))
        lines.append("  " + "  |  ".join(timing_parts))

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
        if td.trunc_episode_rate is not None:
            tool_parts.append(f"trunc_ep={td.trunc_episode_rate:.1%}")
        elif td.hit_max_tokens_rate is not None:
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
            lines.append(_c(f"Multi-Turn Tool & Cache ({td.source})", "bold", color=color))
            lines.append("  " + "  |  ".join(tool_parts))
            if td.trunc_reward_mean is not None and td.untrunc_reward_mean is not None:
                lines.append(
                    f"  trunc_reward={td.trunc_reward_mean:.3f} vs "
                    f"untrunc_reward={td.untrunc_reward_mean:.3f}"
                )

    # Windowed Training Progress (pre-filter preferred)
    if diag.reward_early_mean is not None and diag.reward_late_mean is not None:
        w = max(1, diag.reward_window_size)
        src_desc = (
            "pre-filter all sampled" if diag.reward_source == "prefilter" else "post-filter kept"
        )
        lines.append("")
        lines.append(
            _c(f"Training Progress (first {w} vs last {w} steps, {src_desc})", "bold", color=color)
        )
        delta_rew = diag.reward_late_mean - diag.reward_early_mean
        lines.append(
            f"  reward (mean)     {diag.reward_early_mean:.3f} -> {diag.reward_late_mean:.3f} "
            f"({delta_rew:+.3f})"
        )
        if diag.reward_source == "prefilter" and diag.prefilter_steps:
            early_pf = diag.prefilter_steps[:w]
            late_pf = diag.prefilter_steps[-w:]
            ek_num = sum(
                p.kept_mean_reward * p.episodes_kept
                for p in early_pf
                if p.kept_mean_reward is not None and p.episodes_kept > 0
            )
            ek_den = sum(
                p.episodes_kept
                for p in early_pf
                if p.kept_mean_reward is not None and p.episodes_kept > 0
            )
            lk_num = sum(
                p.kept_mean_reward * p.episodes_kept
                for p in late_pf
                if p.kept_mean_reward is not None and p.episodes_kept > 0
            )
            lk_den = sum(
                p.episodes_kept
                for p in late_pf
                if p.kept_mean_reward is not None and p.episodes_kept > 0
            )
            if ek_den > 0 and lk_den > 0:
                ek_mean = ek_num / ek_den
                lk_mean = lk_num / lk_den
                lines.append(
                    f"  reward (trained)  {ek_mean:.3f} -> {lk_mean:.3f} "
                    f"({lk_mean - ek_mean:+.3f})  "
                    "[post-filter kept, pooled]"
                )
        if diag.prefilter_steps:
            early_pf = diag.prefilter_steps[:w]
            late_pf = diag.prefilter_steps[-w:]
            early_ac = sum(p.ac_tokens_per_turn for p in early_pf) / len(early_pf)
            late_ac = sum(p.ac_tokens_per_turn for p in late_pf) / len(late_pf)
            lines.append(f"  ac_tokens/turn    {early_ac:.1f} -> {late_ac:.1f}  (pre-filter)")
        elif run.steps:
            first_s, last_s = run.steps[0], run.steps[-1]
            if first_s.ac_tokens_per_turn is not None and last_s.ac_tokens_per_turn is not None:
                lines.append(
                    f"  ac_tokens/turn    {first_s.ac_tokens_per_turn:.1f} -> "
                    f"{last_s.ac_tokens_per_turn:.1f}"
                )
    elif run.steps:
        first_s, last_s = run.steps[0], run.steps[-1]
        lines.append("")
        lines.append(_c("Training Progress (step 0 -> final, post-filter)", "bold", color=color))
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

    # logs.log summary if non-zero
    ls = run.log_summary
    if ls is not None and (ls.error_lines or ls.warn_lines or ls.retry_or_rate_limit_lines):
        lines.append("")
        lines.append(_c("logs.log Signals", "bold", color=color))
        lines.append(
            f"  errors={ls.error_lines}  |  warnings={ls.warn_lines}  |  "
            f"retries/rate_limits={ls.retry_or_rate_limit_lines}"
        )

    if eval_diff is not None:
        lines.append("")
        lines.append(render_eval_diff_terminal(eval_diff, color=color))

    if run.warnings:
        lines.append("")
        lines.append(_c(f"Loader Warnings ({len(run.warnings)})", "yellow", color=color))
        for w_msg in run.warnings[:5]:
            lines.append(f"  - {w_msg}")

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
    if diff.base_unmatched_tasks or diff.final_unmatched_tasks:
        lines.append(
            _c(
                f"  unmatched tasks ignored: {diff.base_unmatched_tasks} base-only, "
                f"{diff.final_unmatched_tasks} final-only",
                "yellow",
                color=color,
            )
        )
    lines.append(
        f"  {'metric':<18s} {'base':>7s} {'final':>7s} "
        f"{'delta':>8s}  {'95% CI':>18s}  {'W/R/T':>11s}"
    )
    primary_md = next((m for m in diff.metric_deltas if m.metric == diff.primary_metric), None)
    collapsed_aliases: list[str] = []
    for md in diff.metric_deltas:
        if (
            primary_md is not None
            and md.metric != primary_md.metric
            and is_outcome_metric(md.metric)
            and abs(md.base_mean - primary_md.base_mean) < 1e-6
            and abs(md.final_mean - primary_md.final_mean) < 1e-6
            and abs(md.ci_low - primary_md.ci_low) < 1e-6
            and abs(md.ci_high - primary_md.ci_high) < 1e-6
            and (md.wins, md.regressions, md.ties)
            == (primary_md.wins, primary_md.regressions, primary_md.ties)
        ):
            collapsed_aliases.append(md.metric)
            continue
        ci_str = f"[{md.ci_low:+.3f}, {md.ci_high:+.3f}]"
        wrt = f"{md.wins}/{md.regressions}/{md.ties}"
        delta_str = f"{md.delta:+.3f}"
        if is_outcome_metric(md.metric):
            if md.ci_low > 0:
                delta_str = _c(delta_str, "green", color=color)
            elif md.ci_high < 0:
                delta_str = _c(delta_str, "red", color=color)
        lines.append(
            f"  {md.metric:<18s} {md.base_mean:7.3f} {md.final_mean:7.3f} "
            f"{delta_str:>8s}  {ci_str:>18s}  {wrt:>11s}"
        )
    if collapsed_aliases:
        lines.append(
            _c(
                f"  (identical to {diff.primary_metric}: {', '.join(collapsed_aliases)})",
                "dim",
                color=color,
            )
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
    if diff.answered_base_rate is not None and diff.answered_final_rate is not None:
        split_parts = [
            f"completed={diff.answered_base_rate:.1%}->{diff.answered_final_rate:.1%} "
            f"({(diff.answered_final_rate - diff.answered_base_rate) * 100:+.1f}pp)"
        ]
        if (
            diff.correct_given_answered_base is not None
            and diff.correct_given_answered_final is not None
        ):
            acc_pp = (diff.correct_given_answered_final - diff.correct_given_answered_base) * 100
            split_parts.append(
                f"success|completed={diff.correct_given_answered_base:.1%}->"
                f"{diff.correct_given_answered_final:.1%} ({acc_pp:+.1f}pp)"
            )
        lines.append("  " + "  |  ".join(split_parts))
    if diff.base_end_states or diff.final_end_states:
        lines.append(
            "  ended_how (base)  : "
            f"{_fmt_end_states(diff.base_end_states, diff.base_end_state_success)}"
        )
        lines.append(
            "  ended_how (final) : "
            f"{_fmt_end_states(diff.final_end_states, diff.final_end_state_success)}"
        )
    boot_info = f"  bootstrap: n={diff.n_boot}, seed={diff.seed}"
    if diff.cluster_by:
        c_suffix = f" ({diff.num_clusters} clusters)" if diff.num_clusters is not None else ""
        boot_info += f", cluster_by={diff.cluster_by}{c_suffix}"
    lines.append(_c(boot_info, "dim", color=color))
    for w_msg in diff.warnings:
        lines.append(_c(f"  WARN: {w_msg}", "yellow", color=color))
    return "\n".join(lines)


def render_compare_terminal(
    run_a: RunData,
    diag_a: RunDiagnostics,
    run_b: RunData,
    diag_b: RunDiagnostics,
    *,
    color: bool = True,
) -> str:
    col_w = max(len(diag_a.run_name), len(diag_b.run_name), 18)

    def _row(label: str, va: str, vb: str) -> str:
        return f"  {label:<26s} {va:>{col_w}s}  {vb:>{col_w}s}"

    def _opt(val: float | None, fmt: str) -> str:
        return format(val, fmt) if val is not None else "n/a"

    def _reward_progress(run: RunData, diag: RunDiagnostics) -> str:
        if diag.reward_early_mean is not None and diag.reward_late_mean is not None:
            tag = "pre" if diag.reward_source == "prefilter" else "post"
            return f"{diag.reward_early_mean:.3f}->{diag.reward_late_mean:.3f} ({tag})"
        last = run.steps[-1].reward_total if run.steps else None
        return _opt(last, ".3f")

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
    a_pref = (
        f"{_fmt_int(diag_a.tokens.prefill_total)} ({diag_a.tokens.prefill_multiplier:.2f}x)"
        if diag_a.tokens.prefill_multiplier is not None
        else _fmt_int(diag_a.tokens.prefill_total)
    )
    b_pref = (
        f"{_fmt_int(diag_b.tokens.prefill_total)} ({diag_b.tokens.prefill_multiplier:.2f}x)"
        if diag_b.tokens.prefill_multiplier is not None
        else _fmt_int(diag_b.tokens.prefill_total)
    )
    lines = [
        _c(f"Run Comparison :: {diag_a.run_name}  vs  {diag_b.run_name}", "bold", color=color),
        _row("metric", diag_a.run_name, diag_b.run_name),
        _row("steps", str(diag_a.num_steps), str(diag_b.num_steps)),
        _row(
            "wall time",
            _fmt_duration(diag_a.total_wall_seconds),
            _fmt_duration(diag_b.total_wall_seconds),
        ),
        _row(
            "reward (early->late)",
            _reward_progress(run_a, diag_a),
            _reward_progress(run_b, diag_b),
        ),
        _row("max |KL v1|", _opt(diag_a.max_abs_kl_v1, ".2e"), _opt(diag_b.max_abs_kl_v1, ".2e")),
        _row(
            "dropped groups",
            _opt(diag_a.discarded_group_fraction, ".0%"),
            _opt(diag_b.discarded_group_fraction, ".0%"),
        ),
        _row("prefill tokens", a_pref, b_pref),
        _row(
            "sampled tokens",
            _fmt_int(diag_a.tokens.sampled_total),
            _fmt_int(diag_b.tokens.sampled_total),
        ),
        _row(
            "sampling idle wait",
            _opt(diag_a.sampling_idle_fraction, ".0%"),
            _opt(diag_b.sampling_idle_fraction, ".0%"),
        ),
        _row("median straggler ratio", a_strag, b_strag),
    ]
    return "\n".join(lines)


def eval_diff_to_dict(diff: EvalDiffSummary) -> dict[str, Any]:
    """Build a structured, JSON-serializable dict for an EvalDiffSummary."""
    return {
        "base_label": diff.base_label,
        "final_label": diff.final_label,
        "num_tasks": diff.num_tasks,
        "base_total_episodes": diff.base_total_episodes,
        "final_total_episodes": diff.final_total_episodes,
        "base_failed": diff.base_failed,
        "final_failed": diff.final_failed,
        "base_unmatched_tasks": diff.base_unmatched_tasks,
        "final_unmatched_tasks": diff.final_unmatched_tasks,
        "primary_metric": diff.primary_metric,
        "answer_chars_change_pct": diff.answer_chars_change_pct,
        "turns_change_pct": diff.turns_change_pct,
        "base_end_states": diff.base_end_states,
        "final_end_states": diff.final_end_states,
        "answered_base_rate": diff.answered_base_rate,
        "answered_final_rate": diff.answered_final_rate,
        "correct_given_answered_base": diff.correct_given_answered_base,
        "correct_given_answered_final": diff.correct_given_answered_final,
        "cluster_by": diff.cluster_by,
        "n_boot": diff.n_boot,
        "seed": diff.seed,
        "warnings": diff.warnings,
        "metric_deltas": [asdict(m) for m in diff.metric_deltas],
        "tasks": [
            {
                "task_key": t.task_key,
                "task_id": t.task_id,
                "cluster_id": t.cluster_id,
                "gold_answer": t.gold_answer,
                "base_count": t.base_count,
                "final_count": t.final_count,
                "base_metrics": {k: round(v, 4) for k, v in t.base_metrics.items()},
                "final_metrics": {k: round(v, 4) for k, v in t.final_metrics.items()},
                "deltas": {k: round(v, 4) for k, v in t.deltas.items()},
                "base_mean_turns": round(t.base_mean_turns, 2),
                "final_mean_turns": round(t.final_mean_turns, 2),
                "base_end_states": dict(Counter(ep.end_state for ep in t.base_episodes)),
                "final_end_states": dict(Counter(ep.end_state for ep in t.final_episodes)),
            }
            for t in diff.tasks
        ],
    }


def eval_diff_to_json(diff: EvalDiffSummary) -> str:
    """Serialize an EvalDiffSummary to formatted JSON."""
    return json.dumps(eval_diff_to_dict(diff), indent=2)


def diagnostics_to_json(
    diag: RunDiagnostics,
    *,
    eval_diff: EvalDiffSummary | None = None,
) -> str:
    payload: dict[str, Any] = {"diagnostics": asdict(diag)}
    if eval_diff is not None:
        payload["eval_diff"] = eval_diff_to_dict(eval_diff)
    return json.dumps(payload, indent=2)


def compare_to_json(
    run_a: RunData,
    diag_a: RunDiagnostics,
    run_b: RunData,
    diag_b: RunDiagnostics,
) -> str:
    """Serialize a two-run comparison to formatted JSON."""
    payload: dict[str, Any] = {
        "run_a": {
            "name": run_a.name,
            "log_dir": run_a.log_dir,
            "diagnostics": asdict(diag_a),
        },
        "run_b": {
            "name": run_b.name,
            "log_dir": run_b.log_dir,
            "diagnostics": asdict(diag_b),
        },
    }
    return json.dumps(payload, indent=2)


def has_failures(diag: RunDiagnostics, *, fail_on_warn: bool = False) -> bool:
    for f in diag.findings:
        if f.severity == "fail":
            return True
        if fail_on_warn and f.severity == "warn":
            return True
    return False
