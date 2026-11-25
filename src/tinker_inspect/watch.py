"""Live terminal monitor for an active Tinker RL training run directory."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from tinker_inspect.diagnostics import analyze_run
from tinker_inspect.loader import load_run
from tinker_inspect.schema import RunData, RunDiagnostics, Severity, TokenCostRates
from tinker_inspect.terminal import has_failures

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


def _fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


def _fmt_int(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _overall_status(diag: RunDiagnostics) -> Severity:
    if any(f.severity == "fail" for f in diag.findings):
        return "fail"
    if any(f.severity == "warn" for f in diag.findings):
        return "warn"
    return "pass"


def _step_trunc_pct(run: RunData, step_idx: int) -> float | None:
    max_tokens = run.config.get("max_tokens")
    groups = run.groups_by_step.get(step_idx)
    if not isinstance(max_tokens, int) or max_tokens <= 0 or not groups:
        return None
    total_turns = 0
    trunc_turns = 0
    for g in groups:
        for tr in g.trajectories:
            for turn in tr.steps:
                total_turns += 1
                if turn.ac_len >= max_tokens:
                    trunc_turns += 1
    if total_turns == 0:
        return None
    return (trunc_turns / total_turns) * 100.0


def format_watch_line(
    run: RunData,
    diag: RunDiagnostics,
    step_idx: int,
    *,
    color: bool = True,
) -> str:
    """Format a concise single-line watch summary for a newly observed training step."""
    step_rec = next((s for s in run.steps if s.step == step_idx), None)

    rew_s = (
        f"{step_rec.reward_total:.3f}"
        if step_rec is not None and step_rec.reward_total is not None
        else "n/a"
    )
    kl_s = f"{step_rec.kl_v1:.2e}" if step_rec is not None and step_rec.kl_v1 is not None else "n/a"

    drop_row = next((r for r in diag.per_step_drop_stats if r.get("step") == step_idx), None)
    if drop_row is not None:
        kept_s = (
            f"{drop_row['groups_kept']}/{drop_row['groups_sampled']} "
            f"(all_good={drop_row['groups_all_good']}, all_bad={drop_row['groups_all_bad']})"
        )
    else:
        kept_s = "n/a"

    trunc_pct = _step_trunc_pct(run, step_idx)
    trunc_s = f"{trunc_pct:.1f}%" if trunc_pct is not None else "n/a"

    ac_s = (
        f"{step_rec.ac_tokens_per_turn:.1f}"
        if step_rec is not None and step_rec.ac_tokens_per_turn is not None
        else "n/a"
    )

    wall_val: float | None = None
    if step_rec is not None:
        wall_val = step_rec.time_metrics.get("total")
    if wall_val is None and step_idx in run.timings:
        wall_val = run.timings[step_idx].total_wall
    wall_s = f"{wall_val:.1f}s" if wall_val is not None else "n/a"

    status = _overall_status(diag)
    tag, sty = _SEV_STYLE[status]
    badge = _c(f"[{tag}]", sty, color=color)

    return (
        f"step={step_idx} | rew={rew_s} | kl_v1={kl_s} | kept={kept_s} | "
        f"trunc={trunc_s} | ac_tok={ac_s} | wall={wall_s} | status={badge}"
    )


def watch_run(
    log_dir: str | Path,
    *,
    interval_s: float = 2.0,
    fail_fast: bool = False,
    fail_on_warn: bool = False,
    rates: TokenCostRates | None = None,
    color: bool = True,
    max_polls: int | None = None,
    stream: Any = None,
) -> int:
    """Poll an active Tinker RL log_dir, printing new steps and deduplicated alerts.

    Returns exit code ``2`` when ``fail_fast`` trips (or a completed run ends in failure under
    ``fail_fast``), otherwise ``0``.
    """
    out = stream if stream is not None else sys.stdout
    target = Path(log_dir).expanduser()

    def _emit(line: str) -> None:
        print(line, file=out, flush=True)

    header_printed = False
    waiting_printed = False
    last_step: int | None = None
    seen_alerts: set[tuple[str, Severity, str]] = set()
    polls = 0

    try:
        while True:
            if max_polls is not None and polls >= max_polls:
                return 0
            polls += 1

            run: RunData | None = None
            try:
                run = load_run(target)
            except (FileNotFoundError, ValueError):
                if not waiting_printed:
                    _emit(
                        _c(
                            f"tinker-inspect watch :: waiting for metrics in {target} ...",
                            "dim",
                            color=color,
                        )
                    )
                    waiting_printed = True

            if run is not None:
                diag = analyze_run(run, rates=rates)
                if not header_printed:
                    model_str = f"  ({diag.model_name})" if diag.model_name else ""
                    _emit(_c(f"tinker-inspect watch :: {run.name}{model_str}", "bold", color=color))
                    header_printed = True

                for step_rec in run.steps:
                    if last_step is None or step_rec.step > last_step:
                        _emit(format_watch_line(run, diag, step_rec.step, color=color))
                        last_step = step_rec.step

                for f in diag.findings:
                    if f.severity not in ("warn", "fail"):
                        continue
                    key = (f.check_id, f.severity, f.headline)
                    if key not in seen_alerts:
                        seen_alerts.add(key)
                        tag, sty = _SEV_STYLE[f.severity]
                        _emit(f"  {_c(f'[{tag}]', sty, color=color)} {f.title}: {f.headline}")

                if fail_fast and has_failures(diag, fail_on_warn=fail_on_warn):
                    _emit(
                        _c(
                            "FAIL-FAST TRIGGERED: health check threshold breached; exiting 2.",
                            "red",
                            color=color,
                        )
                    )
                    return 2

                if (
                    run.summary_step is not None and run.summary_step.raw.get("step") is not None
                ) or any(c.name == "final" for c in run.checkpoints):
                    t = diag.tokens
                    cost_s = (
                        f"${t.estimated_cost_usd:.2f}"
                        if t.estimated_cost_usd is not None
                        else "n/a"
                    )
                    _emit("")
                    _emit(
                        _c(
                            f"Run complete :: {run.name} | steps={diag.num_steps} | "
                            f"wall={_fmt_duration(diag.total_wall_seconds)} | "
                            f"sampled={_fmt_int(t.sampled_total)} | "
                            f"trained={_fmt_int(t.trained_total)} | cost={cost_s}",
                            "bold",
                            color=color,
                        )
                    )
                    for f in diag.findings:
                        tag, sty = _SEV_STYLE[f.severity]
                        _emit(f"  {_c(f'[{tag:4s}]', sty, color=color)} {f.title}: {f.headline}")
                    return 2 if (fail_fast and has_failures(diag, fail_on_warn=fail_on_warn)) else 0

            if max_polls is not None and polls >= max_polls:
                return 0
            if interval_s > 0:
                time.sleep(interval_s)
    except KeyboardInterrupt:
        _emit(_c("\nWatch interrupted.", "dim", color=color))
        return 0
