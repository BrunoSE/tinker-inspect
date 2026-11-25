"""Self-contained single-file HTML report builder for run, diff, and compare modes."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from importlib import resources
from pathlib import Path
from typing import Any

from tinker_inspect import __version__
from tinker_inspect.schema import EvalDiffSummary, RunData, RunDiagnostics

_HOME_PATH_RE = re.compile(r"(/Users/[^/\s\"']+|/home/[^/\s\"']+|[A-Za-z]:\\Users\\[^\\\s\"']+)")
_MAX_MSG_CHARS = 4000


def sanitize_value(value: Any) -> Any:
    """Recursively redact local home directory paths for safe public sharing."""
    if isinstance(value, str):
        return _HOME_PATH_RE.sub("~", value)
    if isinstance(value, list):
        return [sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize_value(v) for k, v in value.items()}
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


def _truncate(text: str | None, limit: int = _MAX_MSG_CHARS) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit:,} chars]"


def _serialize_run_block(
    run: RunData,
    diag: RunDiagnostics,
    *,
    sanitize_paths: bool = False,
) -> dict[str, Any]:
    steps_out: list[dict[str, Any]] = []
    for s in run.steps:
        steps_out.append(
            {
                "step": s.step,
                "lr": s.lr,
                "done_frac": s.done_frac,
                "reward_total": s.reward_total,
                "kl_v1": s.kl_v1,
                "kl_v2": s.kl_v2,
                "entropy": s.entropy,
                "kept_episodes": s.kept_episodes,
                "kept_turns": s.kept_turns,
                "turns_per_episode": s.turns_per_episode,
                "ac_tokens_per_turn": s.ac_tokens_per_turn,
                "ob_tokens_per_turn": s.ob_tokens_per_turn,
                "total_ac_tokens": s.total_ac_tokens,
                "total_ob_tokens": s.total_ob_tokens,
                "env_metrics": s.env_metrics,
                "tag_metrics": s.tag_metrics,
                "time_total": s.time_metrics.get("total"),
                "time_sampling": s.time_metrics.get("sampling"),
                "time_train": s.time_metrics.get("train_step"),
                "time_ckpt": s.time_metrics.get("save_checkpoint"),
            }
        )

    timings_out: dict[str, Any] = {}
    for step_idx, t in sorted(run.timings.items()):
        timings_out[str(step_idx)] = {
            "step": t.step,
            "sampling_wall": t.sampling_wall,
            "train_wall": t.train_wall,
            "checkpoint_wall": t.checkpoint_wall,
            "total_wall": t.total_wall,
            "straggler_ratio": t.straggler_ratio,
            "spans": [
                {
                    "name": sp.name,
                    "duration": round(sp.duration, 4),
                    "wall_start": round(sp.wall_start, 4),
                    "wall_end": round(sp.wall_end, 4),
                }
                for sp in t.spans
            ],
        }

    groups_out: dict[str, list[dict[str, Any]]] = {}
    for step_idx, groups in sorted(run.groups_by_step.items()):
        groups_out[str(step_idx)] = [
            {
                "group_idx": g.group_idx,
                "tags": g.tags,
                "status": g.status,
                "kept": g.kept_after_filter,
                "mean_reward": round(g.mean_reward, 4),
                "min_reward": round(g.min_reward, 4),
                "max_reward": round(g.max_reward, 4),
                "mean_turns": round(g.mean_turns, 2),
                "sum_ob_len": g.sum_ob_len,
                "sum_ac_len": g.sum_ac_len,
                "unique_final_ob_len": g.unique_final_ob_len,
                "trajectories": [
                    {
                        "traj_idx": tr.traj_idx,
                        "reward": round(tr.total_reward, 4),
                        "metrics": {k: round(v, 4) for k, v in tr.trajectory_metrics.items()},
                        "final_ob_len": tr.final_ob_len,
                        "turns": [
                            {"ob": tn.ob_len, "ac": tn.ac_len, "r": round(tn.reward, 4)}
                            for tn in tr.steps
                        ],
                    }
                    for tr in g.trajectories
                ],
            }
            for g in groups
        ]

    block: dict[str, Any] = {
        "name": run.name,
        "log_dir": run.log_dir,
        "config": run.config,
        "diagnostics": asdict(diag),
        "steps": steps_out,
        "checkpoints": [
            {
                "name": c.name,
                "batch": c.batch,
                "state_path": c.state_path,
                "sampler_path": c.sampler_path,
            }
            for c in run.checkpoints
        ],
        "timings": timings_out,
        "groups_by_step": groups_out,
        "code_diff": run.code_diff,
        "warnings": run.warnings,
    }
    return sanitize_value(block) if sanitize_paths else block


def _serialize_eval_diff(diff: EvalDiffSummary, *, sanitize_paths: bool = False) -> dict[str, Any]:
    def ser_ep(ep: Any) -> dict[str, Any]:
        return {
            "ok": ep.ok,
            "reward": round(ep.reward, 4),
            "metrics": {k: round(v, 4) for k, v in ep.metrics.items()},
            "num_turns": ep.num_turns,
            "input_tokens": ep.input_tokens,
            "output_tokens": ep.output_tokens,
            "final_answer_chars": ep.final_answer_chars,
            "errors": ep.errors,
            "messages": [
                {
                    "role": m.role,
                    "content": _truncate(m.content),
                    "name": m.name,
                    "tool_calls": [
                        {
                            "name": tc.get("name", ""),
                            "arguments": _truncate(str(tc.get("arguments", "")), 1000),
                        }
                        for tc in m.tool_calls
                    ],
                }
                for m in ep.messages
                if m.role != "system"
            ],
        }

    block: dict[str, Any] = {
        "base_label": diff.base_label,
        "final_label": diff.final_label,
        "num_tasks": diff.num_tasks,
        "base_total_episodes": diff.base_total_episodes,
        "final_total_episodes": diff.final_total_episodes,
        "base_failed": diff.base_failed,
        "final_failed": diff.final_failed,
        "primary_metric": diff.primary_metric,
        "answer_chars_change_pct": diff.answer_chars_change_pct,
        "turns_change_pct": diff.turns_change_pct,
        "metric_deltas": [asdict(m) for m in diff.metric_deltas],
        "tasks": [
            {
                "task_key": t.task_key,
                "task_id": t.task_id,
                "prompt": _truncate(t.prompt, 1200) or "",
                "base_count": t.base_count,
                "final_count": t.final_count,
                "base_metrics": {k: round(v, 4) for k, v in t.base_metrics.items()},
                "final_metrics": {k: round(v, 4) for k, v in t.final_metrics.items()},
                "deltas": {k: round(v, 4) for k, v in t.deltas.items()},
                "base_chars": round(t.base_mean_answer_chars, 1),
                "final_chars": round(t.final_mean_answer_chars, 1),
                "chars_ratio": round(t.answer_chars_ratio, 3)
                if t.answer_chars_ratio is not None
                else None,
                "base_turns": round(t.base_mean_turns, 2),
                "final_turns": round(t.final_mean_turns, 2),
                "base_episodes": [ser_ep(e) for e in t.base_episodes],
                "final_episodes": [ser_ep(e) for e in t.final_episodes],
            }
            for t in diff.tasks
        ],
    }
    return sanitize_value(block) if sanitize_paths else block


def _load_template() -> str:
    return (
        resources.files("tinker_inspect.assets")
        .joinpath("report_template.html")
        .read_text(encoding="utf-8")
    )


def _assemble_html(title: str, payload: dict[str, Any]) -> str:
    template = _load_template()
    safe_title = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )
    html = template.replace("__TITLE__", safe_title)
    html = html.replace("__PAYLOAD_JSON__", _safe_json_for_script(payload))
    return html


def build_run_report_html(
    run: RunData,
    diag: RunDiagnostics,
    *,
    eval_diff: EvalDiffSummary | None = None,
    sanitize_paths: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "mode": "run",
        "version": __version__,
        "run": _serialize_run_block(run, diag, sanitize_paths=sanitize_paths),
        "eval_diff": (
            _serialize_eval_diff(eval_diff, sanitize_paths=sanitize_paths)
            if eval_diff is not None
            else None
        ),
    }
    return _assemble_html(f"tinker-inspect — {run.name}", payload)


def build_eval_diff_html(
    diff: EvalDiffSummary,
    *,
    sanitize_paths: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "mode": "diff",
        "version": __version__,
        "eval_diff": _serialize_eval_diff(diff, sanitize_paths=sanitize_paths),
    }
    return _assemble_html(f"tinker-inspect diff — {diff.base_label} vs {diff.final_label}", payload)


def build_compare_html(
    run_a: RunData,
    diag_a: RunDiagnostics,
    run_b: RunData,
    diag_b: RunDiagnostics,
    *,
    sanitize_paths: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "mode": "compare",
        "version": __version__,
        "run_a": _serialize_run_block(run_a, diag_a, sanitize_paths=sanitize_paths),
        "run_b": _serialize_run_block(run_b, diag_b, sanitize_paths=sanitize_paths),
    }
    return _assemble_html(f"tinker-inspect compare — {run_a.name} vs {run_b.name}", payload)


def write_html(html: str, path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
