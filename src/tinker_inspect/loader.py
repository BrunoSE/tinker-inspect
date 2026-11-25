"""Resilient loaders for Tinker RL log_path directories and eval JSONL files."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from tinker_inspect.schema import (
    CheckpointRecord,
    EvalEpisode,
    EvalMessage,
    GroupStatus,
    GroupSummary,
    RunData,
    SpanRecord,
    StepRecord,
    StepTiming,
    TrajectoryRecord,
    TurnRecord,
)

_ITER_DIR_RE = re.compile(r"^iteration_(\d+)$")
_ENV_METRIC_RE = re.compile(r"^env/([^/]+)/(.+)$")
_MAX_DIFF_BYTES = 100_000


def _to_float(val: Any) -> float | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if math.isfinite(f) else None
    return None


def _to_int(val: Any) -> int | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float) and math.isfinite(val):
        return int(round(val))
    return None


def _iter_jsonl(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                warnings.append(f"Skipped malformed JSON at {path.name}:{lineno} ({e.msg})")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _parse_step_record(row: dict[str, Any], fallback_step: int) -> StepRecord:
    step = _to_int(row.get("step"))
    if step is None:
        step = _to_int(row.get("progress/batch"))
    if step is None:
        step = fallback_step

    env_metrics: dict[str, float] = {}
    tag_metrics: dict[str, dict[str, float]] = defaultdict(dict)
    native_tokens: dict[str, int] = {}
    native_discarded_tokens: dict[str, int] = {}
    rollout_errors: dict[str, float] = {}
    time_metrics: dict[str, float] = {}

    for k, v in row.items():
        fv = _to_float(v)
        if fv is None:
            continue
        m = _ENV_METRIC_RE.match(k)
        if m:
            scope, metric_name = m.group(1), m.group(2)
            if scope == "all":
                env_metrics[metric_name] = fv
            else:
                tag_metrics[scope][metric_name] = fv
            continue
        if k.startswith("tokens/discarded/"):
            iv = _to_int(v)
            if iv is not None:
                native_discarded_tokens[k.removeprefix("tokens/discarded/")] = iv
        elif k.startswith("tokens/") and not k.startswith("tokens/cumulative/"):
            iv = _to_int(v)
            if iv is not None:
                native_tokens[k.removeprefix("tokens/")] = iv
        elif k.startswith("rollout_errors/"):
            rollout_errors[k.removeprefix("rollout_errors/")] = fv
        elif k.startswith("time/"):
            time_metrics[k.removeprefix("time/")] = fv

    return StepRecord(
        step=step,
        raw=row,
        lr=_to_float(row.get("optim/lr")),
        done_frac=_to_float(row.get("progress/done_frac")),
        reward_total=_to_float(row.get("env/all/reward/total")),
        kl_v1=_to_float(row.get("optim/kl_sample_train_v1")),
        kl_v2=_to_float(row.get("optim/kl_sample_train_v2")),
        entropy=_to_float(row.get("optim/entropy")),
        kept_episodes=_to_int(row.get("env/all/total_episodes")),
        kept_turns=_to_int(row.get("env/all/total_turns")),
        turns_per_episode=_to_float(row.get("env/all/turns_per_episode")),
        ac_tokens_per_turn=_to_float(row.get("env/all/ac_tokens_per_turn")),
        ob_tokens_per_turn=_to_float(row.get("env/all/ob_tokens_per_turn")),
        total_ac_tokens=_to_int(row.get("env/all/total_ac_tokens")),
        total_ob_tokens=_to_int(row.get("env/all/total_ob_tokens")),
        env_metrics=env_metrics,
        tag_metrics=dict(tag_metrics),
        native_tokens=native_tokens,
        native_discarded_tokens=native_discarded_tokens,
        native_cost_usd=_to_float(row.get("cost/step_usd")),
        native_discarded_cost_usd=_to_float(row.get("cost/discarded_usd")),
        native_cumulative_cost_usd=_to_float(row.get("cost/cumulative_usd")),
        rollout_errors=rollout_errors,
        time_metrics=time_metrics,
    )


def _classify_group(rewards: list[float]) -> GroupStatus:
    if not rewards:
        return "unknown"
    mn, mx = min(rewards), max(rewards)
    if mx - mn > 1e-9:
        return "mixed"
    if mn >= 1.0 - 1e-9:
        return "all_good"
    if mx <= 1e-9:
        return "all_bad"
    return "constant"


def _load_iteration_summaries(
    log_dir: Path,
    remove_constant: bool,
    warnings: list[str],
) -> dict[int, list[GroupSummary]]:
    groups_by_step: dict[int, list[GroupSummary]] = {}
    iter_dirs: list[tuple[int, Path]] = []
    for child in log_dir.iterdir():
        if child.is_dir():
            m = _ITER_DIR_RE.match(child.name)
            if m:
                iter_dirs.append((int(m.group(1)), child))
    iter_dirs.sort(key=lambda pair: pair[0])

    for iter_idx, iter_path in iter_dirs:
        summary_file = iter_path / "train_rollout_summaries.jsonl"
        if not summary_file.is_file():
            continue
        rows = _iter_jsonl(summary_file, warnings)
        by_group: dict[int, list[TrajectoryRecord]] = defaultdict(list)
        for row in rows:
            gidx = _to_int(row.get("group_idx"))
            tidx = _to_int(row.get("traj_idx"))
            if gidx is None:
                continue
            steps_raw = row.get("steps") or []
            turns: list[TurnRecord] = []
            for s in steps_raw:
                if not isinstance(s, dict):
                    continue
                turns.append(
                    TurnRecord(
                        step_idx=_to_int(s.get("step_idx")) or len(turns),
                        ob_len=_to_int(s.get("ob_len")) or 0,
                        ac_len=_to_int(s.get("ac_len")) or 0,
                        reward=_to_float(s.get("reward")) or 0.0,
                        episode_done=bool(s.get("episode_done", False)),
                    )
                )
            t_metrics = {
                str(k): fv
                for k, v in (row.get("trajectory_metrics") or {}).items()
                if (fv := _to_float(v)) is not None
            }
            traj = TrajectoryRecord(
                iteration=_to_int(row.get("iteration"))
                if row.get("iteration") is not None
                else iter_idx,
                group_idx=gidx,
                traj_idx=tidx if tidx is not None else len(by_group[gidx]),
                tags=[str(t) for t in (row.get("tags") or [])],
                total_reward=_to_float(row.get("total_reward")) or 0.0,
                final_reward=_to_float(row.get("final_reward")) or 0.0,
                trajectory_metrics=t_metrics,
                steps=turns,
                final_ob_len=_to_int(row.get("final_ob_len")) or 0,
                sampling_client_step=_to_int(row.get("sampling_client_step")),
            )
            by_group[gidx].append(traj)

        if not by_group:
            continue

        group_list: list[GroupSummary] = []
        any_mixed = False
        for gidx in sorted(by_group):
            trajs = sorted(by_group[gidx], key=lambda t: t.traj_idx)
            rewards = [t.total_reward for t in trajs]
            status = _classify_group(rewards)
            if status == "mixed":
                any_mixed = True
            group_list.append(
                GroupSummary(
                    iteration=iter_idx,
                    group_idx=gidx,
                    tags=list(trajs[0].tags) if trajs else [],
                    trajectories=trajs,
                    status=status,
                    kept_after_filter=(status == "mixed") if remove_constant else True,
                    mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
                    min_reward=min(rewards) if rewards else 0.0,
                    max_reward=max(rewards) if rewards else 0.0,
                    mean_turns=sum(t.num_turns for t in trajs) / len(trajs) if trajs else 0.0,
                    sum_ob_len=sum(t.sum_ob_len for t in trajs),
                    sum_ac_len=sum(t.sum_ac_len for t in trajs),
                    unique_final_ob_len=sum(t.unique_tokens for t in trajs),
                    estimated_prefill_cached=sum(t.estimated_prefill_cached for t in trajs),
                    estimated_prefill_uncached=sum(t.estimated_prefill_uncached for t in trajs),
                )
            )
        # Match tinker_cookbook's remove_constant_reward_groups all-constant [0:1] fallback.
        if remove_constant and not any_mixed and group_list:
            group_list[0].kept_after_filter = True
        groups_by_step[iter_idx] = group_list

    return groups_by_step


def _load_timings(log_dir: Path, warnings: list[str]) -> dict[int, StepTiming]:
    timings: dict[int, StepTiming] = {}
    rows = _iter_jsonl(log_dir / "timing_spans.jsonl", warnings)
    for idx, row in enumerate(rows):
        step = _to_int(row.get("step"))
        if step is None:
            step = idx
        spans: list[SpanRecord] = []
        rollout_durations: list[float] = []
        sampling_wall: float | None = None
        train_wall: float | None = None
        checkpoint_wall: float | None = None
        max_wall_end = 0.0

        for s in row.get("spans") or []:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name", ""))
            dur = _to_float(s.get("duration")) or 0.0
            ws = _to_float(s.get("wall_start")) or 0.0
            we = _to_float(s.get("wall_end")) or (ws + dur)
            max_wall_end = max(max_wall_end, we)
            spans.append(SpanRecord(name=name, duration=dur, wall_start=ws, wall_end=we))
            if name in ("do_group_rollout_and_filter_constant_reward", "do_group_rollout"):
                rollout_durations.append(dur)
            elif name in ("sampling", "sample"):
                sampling_wall = dur
            elif name in ("train_step", "train"):
                train_wall = (train_wall or 0.0) + dur
            elif name == "save_checkpoint":
                checkpoint_wall = (checkpoint_wall or 0.0) + dur

        straggler: float | None = None
        if rollout_durations:
            mean_d = sum(rollout_durations) / len(rollout_durations)
            if mean_d > 1e-9:
                straggler = max(rollout_durations) / mean_d

        timings[step] = StepTiming(
            step=step,
            spans=spans,
            rollout_durations=rollout_durations,
            sampling_wall=sampling_wall,
            train_wall=train_wall,
            checkpoint_wall=checkpoint_wall,
            total_wall=max_wall_end if max_wall_end > 0 else None,
            straggler_ratio=straggler,
        )
    return timings


def _load_checkpoints(log_dir: Path, warnings: list[str]) -> list[CheckpointRecord]:
    out: list[CheckpointRecord] = []
    for row in _iter_jsonl(log_dir / "checkpoints.jsonl", warnings):
        batch = _to_int(row.get("batch"))
        if batch is None:
            continue
        cum_raw = row.get("cumulative_tokens")
        cum_tokens: dict[str, int] = {}
        if isinstance(cum_raw, dict):
            for k, v in cum_raw.items():
                if (iv := _to_int(v)) is not None:
                    cum_tokens[str(k)] = iv
        out.append(
            CheckpointRecord(
                name=str(row.get("name", f"batch_{batch}")),
                batch=batch,
                state_path=row.get("state_path"),
                sampler_path=row.get("sampler_path"),
                cumulative_tokens=cum_tokens,
            )
        )
    out.sort(key=lambda c: c.batch)
    return out


def load_run(log_dir: str | Path) -> RunData:
    """Load a Tinker RL training run directory (`log_path`)."""
    root = Path(log_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {root}")

    warnings: list[str] = []
    config: dict[str, Any] = {}
    config_path = root / "config.json"
    if config_path.is_file():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(loaded, dict):
                config = loaded
        except json.JSONDecodeError as e:
            warnings.append(f"Could not parse config.json ({e.msg})")

    step_map: dict[int, StepRecord] = {}
    summary_step: StepRecord | None = None
    for idx, row in enumerate(_iter_jsonl(root / "metrics.jsonl", warnings)):
        rec = _parse_step_record(row, fallback_step=idx)
        # tinker-cookbook main emits a final post-loop summary row with only
        # cumulative/discarded token & cost counters and no env/optim/time keys.
        is_post_loop_summary = (
            not rec.env_metrics
            and not rec.tag_metrics
            and not rec.time_metrics
            and rec.kl_v1 is None
            and rec.entropy is None
            and rec.lr is None
            and (
                bool(rec.native_discarded_tokens)
                or rec.native_cumulative_cost_usd is not None
                or any(k.startswith("tokens/cumulative/") for k in row)
            )
        )
        if is_post_loop_summary:
            summary_step = rec
        else:
            step_map[rec.step] = rec
    steps = [step_map[s] for s in sorted(step_map)]

    remove_constant = bool(config.get("remove_constant_reward_groups", False))
    groups_by_step = _load_iteration_summaries(root, remove_constant, warnings)
    timings = _load_timings(root, warnings)
    checkpoints = _load_checkpoints(root, warnings)

    if not steps and not groups_by_step:
        raise ValueError(
            f"No metrics.jsonl or iteration_*/train_rollout_summaries.jsonl found in {root}"
        )

    code_diff: str | None = None
    diff_path = root / "code.diff"
    if diff_path.is_file():
        raw_diff = diff_path.read_bytes()
        if len(raw_diff) > _MAX_DIFF_BYTES:
            code_diff = (
                raw_diff[:_MAX_DIFF_BYTES].decode("utf-8", errors="replace")
                + f"\n... [truncated at {_MAX_DIFF_BYTES} bytes]"
            )
        else:
            text = raw_diff.decode("utf-8", errors="replace").strip()
            code_diff = text if text else None

    return RunData(
        log_dir=str(root),
        name=root.name,
        config=config,
        steps=steps,
        checkpoints=checkpoints,
        timings=timings,
        groups_by_step=groups_by_step,
        summary_step=summary_step,
        code_diff=code_diff,
        warnings=warnings,
    )


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part and part["text"] is not None:
                parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def _extract_trace_reward(trace: dict[str, Any]) -> float:
    if (fv := _to_float(trace.get("reward"))) is not None:
        return fv
    rewards = trace.get("rewards")
    if isinstance(rewards, dict):
        total = 0.0
        for val in rewards.values():
            if isinstance(val, dict) and (score := _to_float(val.get("score"))) is not None:
                weight = _to_float(val.get("weight"))
                total += score * (1.0 if weight is None else weight)
            elif (fv := _to_float(val)) is not None:
                total += fv
        return total
    if isinstance(rewards, list):
        return sum(fv for v in rewards if (fv := _to_float(v)) is not None)
    return 0.0


def _parse_messages(raw_messages: Any) -> list[EvalMessage]:
    msgs: list[EvalMessage] = []
    if not isinstance(raw_messages, list):
        return msgs
    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        tool_calls_raw = m.get("tool_calls") or []
        tool_calls: list[dict[str, Any]] = []
        if isinstance(tool_calls_raw, list):
            for tc in tool_calls_raw:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                tool_calls.append(
                    {
                        "name": str(fn.get("name", "")),
                        "arguments": fn.get("arguments", ""),
                    }
                )
        msgs.append(
            EvalMessage(
                role=str(m.get("role", "unknown")),
                content=_content_to_text(m.get("content"))
                if m.get("content") is not None
                else None,
                name=str(m["name"]) if m.get("name") is not None else None,
                tool_calls=tool_calls,
            )
        )
    return msgs


def _final_answer_chars(messages: list[EvalMessage]) -> int:
    for msg in reversed(messages):
        if msg.role == "assistant" and msg.content:
            return len(msg.content)
    return 0


def _prompt_from_trace(trace: dict[str, Any], messages: list[EvalMessage]) -> str:
    raw_prompt = trace.get("prompt")
    if isinstance(raw_prompt, str) and raw_prompt.strip():
        return raw_prompt.strip()
    if isinstance(raw_prompt, list):
        for m in raw_prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                txt = _content_to_text(m.get("content")).strip()
                if txt:
                    return txt
    for m in messages:
        if m.role == "user" and m.content and m.content.strip():
            return m.content.strip()
    return ""


def load_eval_jsonl(path: str | Path) -> list[EvalEpisode]:
    """Load evaluation rollouts from verifiers v1, legacy vf-eval, or rollout summary JSONL."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Eval JSONL not found: {p}")

    warnings: list[str] = []
    rows = _iter_jsonl(p, warnings)
    episodes: list[EvalEpisode] = []

    for idx, row in enumerate(rows):
        # Format 1: verifiers v1 / tinker_eval.py episode JSONL with "traces"
        if isinstance(row.get("traces"), list) and row["traces"]:
            split = str(row["split"]) if row.get("split") is not None else None
            ok = bool(row.get("ok", True))
            top_errors = [str(e) for e in (row.get("errors") or [])]
            for trace in row["traces"]:
                if not isinstance(trace, dict):
                    continue
                messages = _parse_messages(trace.get("messages"))
                prompt = _prompt_from_trace(trace, messages)
                task_id = trace.get("task_id")
                task_key = f"task:{task_id}" if task_id is not None else (prompt or f"row:{idx}")
                metrics = {
                    str(k): fv
                    for k, v in (trace.get("metrics") or {}).items()
                    if (fv := _to_float(v)) is not None
                }
                reward = _extract_trace_reward(trace)
                turns = _to_int(trace.get("sampled_turns"))
                if turns is None:
                    turns = sum(1 for m in messages if m.role == "assistant")
                trace_errors = [str(e) for e in (trace.get("errors") or [])]
                episodes.append(
                    EvalEpisode(
                        task_key=task_key,
                        task_id=task_id,
                        prompt=prompt,
                        split=split,
                        ok=ok and not trace_errors,
                        reward=reward,
                        metrics=metrics,
                        num_turns=turns,
                        input_tokens=_to_int(trace.get("num_input_tokens")),
                        output_tokens=_to_int(trace.get("num_output_tokens")),
                        final_answer_chars=_final_answer_chars(messages),
                        messages=messages,
                        errors=top_errors + trace_errors,
                    )
                )
            continue

        # Format 2: rollout_summaries.jsonl
        if "group_idx" in row and "total_reward" in row:
            gidx = row["group_idx"]
            metrics = {
                str(k): fv
                for k, v in (row.get("trajectory_metrics") or {}).items()
                if (fv := _to_float(v)) is not None
            }
            steps = row.get("steps") or []
            episodes.append(
                EvalEpisode(
                    task_key=f"group:{gidx}",
                    task_id=gidx,
                    prompt=f"Group {gidx}",
                    split=str(row["split"]) if row.get("split") is not None else None,
                    ok=True,
                    reward=_to_float(row.get("total_reward")) or 0.0,
                    metrics=metrics,
                    num_turns=len(steps),
                    input_tokens=sum(
                        (_to_int(s.get("ob_len")) or 0) for s in steps if isinstance(s, dict)
                    ),
                    output_tokens=sum(
                        (_to_int(s.get("ac_len")) or 0) for s in steps if isinstance(s, dict)
                    ),
                    final_answer_chars=0,
                    messages=[],
                )
            )
            continue

        # Format 3: tinker-cookbook git_tool_rl / native eval JSONL with "scores" dict
        if isinstance(row.get("scores"), dict) and "messages" in row:
            messages = _parse_messages(row.get("messages"))
            prompt = _prompt_from_trace(row, messages)
            task_id = row.get("task_id")
            task_key = f"task:{task_id}" if task_id is not None else (prompt or f"row:{idx}")
            scores = row["scores"]
            metrics = {str(k): fv for k, v in scores.items() if (fv := _to_float(v)) is not None}
            reward = _to_float(scores.get("shaped_reward"))
            if reward is None:
                reward = _to_float(scores.get("outcome"))
            if reward is None:
                reward = _to_float(row.get("reward")) or 0.0
            err = row.get("error")
            ok = bool(row.get("messages") is not None or row.get("graded_by_env")) and not bool(err)
            episodes.append(
                EvalEpisode(
                    task_key=task_key,
                    task_id=task_id,
                    prompt=prompt,
                    split=str(row["split"]) if row.get("split") is not None else None,
                    ok=ok,
                    reward=reward,
                    metrics=metrics,
                    num_turns=sum(1 for m in messages if m.role == "assistant"),
                    input_tokens=_to_int(
                        row.get("num_input_tokens")
                        if row.get("num_input_tokens") is not None
                        else row.get("input_tokens", row.get("sum_ob_tokens"))
                    ),
                    output_tokens=_to_int(
                        row.get("num_output_tokens")
                        if row.get("num_output_tokens") is not None
                        else row.get("output_tokens", row.get("sum_ac_tokens"))
                    ),
                    final_answer_chars=_final_answer_chars(messages),
                    messages=messages,
                    errors=[str(err)] if err else [],
                )
            )
            continue

        # Format 4: legacy vf-eval results.jsonl
        messages = _parse_messages(row.get("prompt")) + _parse_messages(row.get("completion"))
        prompt = _prompt_from_trace(row, messages)
        task_id = row.get("example_id", row.get("task_id"))
        task_key = f"task:{task_id}" if task_id is not None else (prompt or f"row:{idx}")
        metrics = {
            str(k): fv
            for k, v in (row.get("metrics") or {}).items()
            if (fv := _to_float(v)) is not None
        }
        err = row.get("error")
        episodes.append(
            EvalEpisode(
                task_key=task_key,
                task_id=task_id,
                prompt=prompt,
                split=None,
                ok=not bool(err),
                reward=_to_float(row.get("reward")) or 0.0,
                metrics=metrics,
                num_turns=sum(1 for m in messages if m.role == "assistant"),
                input_tokens=None,
                output_tokens=None,
                final_answer_chars=_final_answer_chars(messages),
                messages=messages,
                errors=[str(err)] if err else [],
            )
        )

    return episodes
