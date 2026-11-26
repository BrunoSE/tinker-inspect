"""Live terminal monitor for an active Tinker RL training run directory."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from tinker_inspect.diagnostics import analyze_run
from tinker_inspect.loader import load_run
from tinker_inspect.schema import (
    PrefilterStepSummary,
    RunData,
    RunDiagnostics,
    Severity,
    TokenCostRates,
)
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


def _find_prefilter_step(diag: RunDiagnostics, step_idx: int) -> PrefilterStepSummary | None:
    return next((p for p in diag.prefilter_steps if p.step == step_idx), None)


def _step_episode_trunc_pct(
    run: RunData,
    pf: PrefilterStepSummary | None,
    step_idx: int,
) -> float | None:
    if pf is not None and pf.trunc_episode_rate is not None:
        return pf.trunc_episode_rate * 100.0
    groups = run.groups_by_step.get(step_idx)
    if not groups:
        return None
    raw_max_tokens = run.config.get("max_tokens")
    max_tokens = raw_max_tokens if isinstance(raw_max_tokens, int) and raw_max_tokens > 0 else None
    total_eps = 0
    trunc_eps = 0
    for g in groups:
        for tr in g.trajectories:
            total_eps += 1
            if tr.end_state(max_tokens) == "token_limit":
                trunc_eps += 1
    if total_eps == 0:
        return None
    return (trunc_eps / total_eps) * 100.0


def _step_status(
    run: RunData,
    diag: RunDiagnostics,
    step_idx: int,
    *,
    pf: PrefilterStepSummary | None,
    drop_row: dict[str, Any] | None,
    trunc_pct: float | None,
) -> Severity:
    """Compute a per-step status badge rather than stamping final cumulative status on all steps."""
    step_rec = next((s for s in run.steps if s.step == step_idx), None)
    status: Severity = "pass"

    def _bump(sev: Severity) -> None:
        nonlocal status
        rank = {"pass": 0, "info": 1, "warn": 2, "fail": 3}
        if rank[sev] > rank[status]:
            status = sev

    if step_rec is not None:
        on_policy = (
            run.config.get("num_substeps", 1) in (1, None)
            and not run.config.get("stream_minibatch_config")
            and not run.config.get("async_config")
        )
        kl1 = abs(step_rec.kl_v1) if step_rec.kl_v1 is not None else 0.0
        kl2 = abs(step_rec.kl_v2) if step_rec.kl_v2 is not None else 0.0
        if on_policy and max(kl1, kl2) > 1e-4:
            _bump("fail")
        elif on_policy and max(kl1, kl2) > 1e-6:
            _bump("warn")
        if step_rec.sample_train_mismatch_frac is not None:
            if step_rec.sample_train_mismatch_frac > 0.05:
                _bump("fail")
            elif step_rec.sample_train_mismatch_frac > 0.005:
                _bump("warn")

    if drop_row is not None:
        drop_frac = float(drop_row.get("drop_frac", 0.0))
        if drop_frac >= 1.0:
            _bump("fail")
        elif drop_frac >= 0.70:
            _bump("warn")

    if trunc_pct is not None and trunc_pct >= 25.0:
        _bump("warn")

    if pf is not None and pf.max_turns_rate is not None and pf.max_turns_rate >= 0.25:
        _bump("warn")

    for f in diag.findings:
        if f.step == step_idx and f.severity in ("warn", "fail"):
            _bump(f.severity)

    return status


def format_watch_line(
    run: RunData,
    diag: RunDiagnostics,
    step_idx: int,
    *,
    color: bool = True,
) -> str:
    """Format a concise single-line watch summary for a newly observed training step."""
    step_rec = next((s for s in run.steps if s.step == step_idx), None)
    pf = _find_prefilter_step(diag, step_idx)

    if pf is not None:
        rew_s = f"{pf.mean_reward:.3f}"
    elif step_rec is not None and step_rec.reward_total is not None:
        rew_s = f"{step_rec.reward_total:.3f}"
    else:
        rew_s = "n/a"

    kl_s = f"{step_rec.kl_v1:.2e}" if step_rec is not None and step_rec.kl_v1 is not None else "n/a"

    drop_row = next((r for r in diag.per_step_drop_stats if r.get("step") == step_idx), None)
    if drop_row is not None:
        kept_s = (
            f"{drop_row['groups_kept']}/{drop_row['groups_sampled']} "
            f"(all_good={drop_row['groups_all_good']}, all_bad={drop_row['groups_all_bad']})"
        )
    else:
        kept_s = "n/a"

    trunc_pct = _step_episode_trunc_pct(run, pf, step_idx)
    trunc_s = f"{trunc_pct:.1f}%" if trunc_pct is not None else "n/a"

    if pf is not None and pf.ac_tokens_per_turn > 0:
        ac_s = f"{pf.ac_tokens_per_turn:.1f}"
    elif step_rec is not None and step_rec.ac_tokens_per_turn is not None:
        ac_s = f"{step_rec.ac_tokens_per_turn:.1f}"
    else:
        ac_s = "n/a"

    wall_val: float | None = None
    if step_rec is not None:
        wall_val = step_rec.time_metrics.get("total")
    if wall_val is None and step_idx in run.timings:
        wall_val = run.timings[step_idx].total_wall
    wall_s = f"{wall_val:.1f}s" if wall_val is not None else "n/a"

    status = _step_status(
        run,
        diag,
        step_idx,
        pf=pf,
        drop_row=drop_row,
        trunc_pct=trunc_pct,
    )
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

                is_complete = (
                    run.summary_step is not None and run.summary_step.raw.get("step") is not None
                ) or any(c.name == "final" for c in run.checkpoints)

                if not is_complete:
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
                else:
                    t = diag.tokens
                    cost_s = (
                        f"${t.estimated_cost_usd:.2f}"
                        if t.estimated_cost_usd is not None
                        else "n/a"
                    )
                    cum_status = _overall_status(diag)
                    ctag, csty = _SEV_STYLE[cum_status]
                    cbadge = _c(f"[{ctag}]", csty, color=color)
                    _emit("")
                    _emit(
                        _c(
                            f"Run complete :: {run.name} | steps={diag.num_steps} | "
                            f"wall={_fmt_duration(diag.total_wall_seconds)} | "
                            f"sampled={_fmt_int(t.sampled_total)} | "
                            f"trained={_fmt_int(t.trained_total)} | cost={cost_s} | ",
                            "bold",
                            color=color,
                        )
                        + f"status={cbadge}"
                    )
                    for f in diag.findings:
                        key = (f.check_id, f.severity, f.headline)
                        if key in seen_alerts:
                            continue
                        seen_alerts.add(key)
                        tag, sty = _SEV_STYLE[f.severity]
                        _emit(f"  {_c(f'[{tag:4s}]', sty, color=color)} {f.title}: {f.headline}")
                    if fail_fast and has_failures(diag, fail_on_warn=fail_on_warn):
                        _emit(
                            _c(
                                "FAIL-FAST TRIGGERED: health check threshold breached; exiting 2.",
                                "red",
                                color=color,
                            )
                        )
                        return 2
                    return 0

            if max_polls is not None and polls >= max_polls:
                return 0
            if interval_s > 0:
                time.sleep(interval_s)
    except KeyboardInterrupt:
        _emit(_c("\nWatch interrupted.", "dim", color=color))
        return 0
