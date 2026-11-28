"""Resilient loaders for Tinker RL log_path directories and eval JSONL files."""

from __future__ import annotations

import io
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from tinker_inspect.schema import (
    CheckpointRecord,
    EvalEpisode,
    EvalMessage,
    GroupStatus,
    GroupSummary,
    LogSummary,
    RunData,
    SpanRecord,
    StepRecord,
    StepTiming,
    TrajectoryRecord,
    TurnRecord,
)

_ITER_DIR_RE = re.compile(r"^iteration_(\d+)$")
_ENV_METRIC_RE = re.compile(r"^env/([^/]+)/(.+)$")
_LOG_ERROR_RE = re.compile(r"\[(ERROR|CRITICAL)\]|^Traceback \(most recent call last\):")
_LOG_WARN_RE = re.compile(r"\[(WARN|WARNING)\]")
_LOG_RETRY_RE = re.compile(
    r"\[(INFO|WARN|WARNING|ERROR|CRITICAL)\].*(?:429|rate[_ -]?limit|retrying|backoff|timed? ?out)",
    re.IGNORECASE,
)
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_MAX_DIFF_BYTES = 100_000
_MAX_LOGTREE_GROUPS_PER_STEP = 8
_MAX_LOGTREE_TRAJS_PER_GROUP = 8
_MAX_LOGTREE_EXCERPT_CHARS = 1600
_TOOL_ERROR_KEYS = ("tool_errors", "tool_error_count", "invalid_tool_calls", "tool_call_errors")


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


def _to_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and math.isfinite(float(val)):
        return float(val) >= 0.5
    return None


def _is_zstd_file(path: Path) -> bool:
    if path.suffix.lower() == ".zst" or path.name.lower().endswith(".jsonl.zst"):
        return True
    try:
        with path.open("rb") as f:
            return f.read(4) == _ZSTD_MAGIC
    except OSError:
        return False


def _open_jsonl_text(path: Path) -> io.TextIOBase:
    if _is_zstd_file(path):
        try:
            from compression import zstd  # type: ignore[import-not-found]
        except ImportError:
            zstd_bin = shutil.which("zstd")
            if zstd_bin is None:
                raise ValueError(
                    f"Compressed .zst file cannot be read directly on Python <3.14 without "
                    f"`zstd` on PATH ({path.name}); install zstd or decompress first "
                    f"(e.g. `zstd -d {path.name}`)."
                ) from None
            try:
                proc = subprocess.run(
                    [zstd_bin, "-dc", "--", str(path)],
                    capture_output=True,
                    check=False,
                    timeout=30,
                )
            except subprocess.TimeoutExpired as exc:
                raise ValueError(
                    f"Timed out after 30s decompressing .zst file ({path.name}); "
                    f"decompress manually first (`zstd -d {path.name}`)."
                ) from exc
            if proc.returncode != 0:
                err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
                detail = f" ({err_msg})" if err_msg else ""
                raise ValueError(
                    f"Compressed .zst file could not be decompressed with zstd "
                    f"({path.name}){detail}."
                ) from None
            if len(proc.stdout) > 512 * 1024 * 1024:
                raise ValueError(
                    f"Decompressed .zst file exceeds 512 MiB safety limit ({path.name})."
                ) from None
            return io.StringIO(proc.stdout.decode("utf-8", errors="replace"))
        try:
            raw = zstd.decompress(path.read_bytes())
        except Exception as exc:
            raise ValueError(
                f"Compressed .zst file could not be decompressed ({path.name}): {exc}"
            ) from exc
        return io.StringIO(raw.decode("utf-8", errors="replace"))
    return path.open("r", encoding="utf-8", errors="replace")


def _iter_jsonl(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with _open_jsonl_text(path) as f:
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
    test_metrics: dict[str, float] = {}
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
        if k.startswith("test/"):
            test_metrics[k.removeprefix("test/")] = fv
        elif k.startswith("tokens/discarded/"):
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
        sample_train_mismatch_frac=_to_float(row.get("optim/sample_train_mismatch_frac")),
        sample_train_max_abs_diff=_to_float(row.get("optim/sample_train_max_abs_diff")),
        entropy=_to_float(row.get("optim/entropy")),
        kept_episodes=_to_int(row.get("env/all/total_episodes")),
        kept_turns=_to_int(row.get("env/all/total_turns")),
        turns_per_episode=_to_float(row.get("env/all/turns_per_episode")),
        ac_tokens_per_turn=_to_float(row.get("env/all/ac_tokens_per_turn")),
        ob_tokens_per_turn=_to_float(row.get("env/all/ob_tokens_per_turn")),
        total_ac_tokens=_to_int(row.get("env/all/total_ac_tokens")),
        total_ob_tokens=_to_int(row.get("env/all/total_ob_tokens")),
        env_metrics=env_metrics,
        test_metrics=test_metrics,
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


def _parse_turn_metrics(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        fv = _to_float(v)
        if fv is not None:
            out[str(k)] = fv
    return out


def _merge_trajectory_metrics(
    turns: list[TurnRecord],
    explicit_metrics: dict[str, float],
) -> dict[str, float]:
    """Merge per-turn ``steps[*].metrics`` into trajectory metrics (last value wins).

    Explicit ``trajectory_metrics`` from the rollout row take precedence when non-empty.
    """
    merged: dict[str, float] = {}
    for tn in turns:
        for k, v in tn.metrics.items():
            merged[k] = v
    merged.update(explicit_metrics)
    return merged


def _load_iteration_summaries(
    log_dir: Path,
    remove_constant: bool,
    warnings: list[str],
    *,
    max_tokens: int | None = None,
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
                        metrics=_parse_turn_metrics(s.get("metrics")),
                    )
                )
            explicit_t_metrics = _parse_turn_metrics(row.get("trajectory_metrics"))
            merged_t_metrics = _merge_trajectory_metrics(turns, explicit_t_metrics)
            traj = TrajectoryRecord(
                iteration=_to_int(row.get("iteration"))
                if row.get("iteration") is not None
                else iter_idx,
                group_idx=gidx,
                traj_idx=tidx if tidx is not None else len(by_group[gidx]),
                tags=[str(t) for t in (row.get("tags") or [])],
                total_reward=_to_float(row.get("total_reward")) or 0.0,
                final_reward=_to_float(row.get("final_reward")) or 0.0,
                trajectory_metrics=merged_t_metrics,
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
            trunc_eps = 0
            max_turns_eps = 0
            tool_errs = 0.0
            for t in trajs:
                es = t.end_state(max_tokens)
                if es == "token_limit":
                    trunc_eps += 1
                elif es == "turn_limit":
                    max_turns_eps += 1
                for ek in _TOOL_ERROR_KEYS:
                    if ek in t.trajectory_metrics:
                        tool_errs += t.trajectory_metrics[ek]
                        break
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
                    trunc_episodes=trunc_eps,
                    max_turns_episodes=max_turns_eps,
                    tool_errors_total=round(tool_errs, 3),
                )
            )
        # Match tinker_cookbook's remove_constant_reward_groups all-constant [0:1] fallback.
        if remove_constant and not any_mixed and group_list:
            group_list[0].kept_after_filter = True
        groups_by_step[iter_idx] = group_list

    return groups_by_step


def _dom_text(node: Any) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return "".join(_dom_text(c) for c in (node.get("children") or []))
    if isinstance(node, list):
        return "".join(_dom_text(c) for c in node)
    return ""


def _extract_table_kv(node: Any, out: list[tuple[str, str]]) -> None:
    if not isinstance(node, dict):
        return
    tag = node.get("tag")
    if tag == "tr":
        cells = [
            _dom_text(c).strip()
            for c in (node.get("children") or [])
            if isinstance(c, dict) and c.get("tag") == "td"
        ]
        if len(cells) >= 2 and cells[0]:
            out.append((cells[0], cells[1]))
        return
    for c in node.get("children") or []:
        _extract_table_kv(c, out)


def _extract_prose_blocks(node: Any, out: list[str]) -> None:
    if not isinstance(node, dict):
        return
    tag = node.get("tag")
    if tag in ("pre", "code", "p"):
        txt = _dom_text(node).strip()
        if txt:
            out.append(txt)
        return
    if tag == "table":
        return
    for c in node.get("children") or []:
        _extract_prose_blocks(c, out)


def _parse_logtree_file(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        warnings.append(f"Skipped malformed {path.name} ({e.msg})")
        return []
    root = data.get("root") if isinstance(data, dict) else None
    if not isinstance(root, dict):
        return []

    group_sections = [
        ch
        for ch in (root.get("children") or [])
        if isinstance(ch, dict) and ch.get("tag") == "section"
    ]
    parsed_groups: list[dict[str, Any]] = []
    for g_pos, gsec in enumerate(group_sections[:_MAX_LOGTREE_GROUPS_PER_STEP]):
        title = ""
        for ch in gsec.get("children") or []:
            if isinstance(ch, dict) and ch.get("tag") in ("h2", "h3"):
                title = _dom_text(ch).strip()
                break
        # Find trajectory sections inside this group section
        traj_sections: list[dict[str, Any]] = []

        def _collect_traj_sections(node: Any, out: list[dict[str, Any]]) -> None:
            if not isinstance(node, dict):
                return
            if node.get("tag") == "section":
                heading = ""
                for ch in node.get("children") or []:
                    if isinstance(ch, dict) and ch.get("tag") in ("h4", "h5"):
                        heading = _dom_text(ch).strip()
                        break
                if heading.lower().startswith("trajectory"):
                    out.append(node)
                    return
            for ch in node.get("children") or []:
                _collect_traj_sections(ch, out)

        _collect_traj_sections(gsec, traj_sections)

        trajs_out: list[dict[str, Any]] = []
        for t_pos, tsec in enumerate(traj_sections[:_MAX_LOGTREE_TRAJS_PER_GROUP]):
            heading = ""
            for ch in tsec.get("children") or []:
                if isinstance(ch, dict) and ch.get("tag") in ("h4", "h5"):
                    heading = _dom_text(ch).strip()
                    break
            m_tidx = re.search(r"(\d+)", heading)
            traj_idx = int(m_tidx.group(1)) if m_tidx else t_pos
            kv_pairs: list[tuple[str, str]] = []
            _extract_table_kv(tsec, kv_pairs)
            kv_map = dict(kv_pairs)
            summary_keys = [
                k
                for k in (
                    "num_turns",
                    "total_return",
                    "correct",
                    "tool_calls_count",
                    "tool_errors",
                    "truncated_thinking",
                    "hit_max_turns",
                )
                if k in kv_map
            ]
            summary = ", ".join(f"{k}={kv_map[k]}" for k in summary_keys)
            prose: list[str] = []
            _extract_prose_blocks(tsec, prose)
            if prose:
                excerpt = "\n\n".join(prose)
            else:
                excerpt = " | ".join(f"{k}: {v}" for k, v in kv_pairs[:16])
            if len(excerpt) > _MAX_LOGTREE_EXCERPT_CHARS:
                excerpt = excerpt[:_MAX_LOGTREE_EXCERPT_CHARS] + "\n... [truncated]"
            trajs_out.append(
                {
                    "traj_idx": traj_idx,
                    "summary": summary or heading or f"Trajectory {traj_idx}",
                    "excerpt": excerpt,
                    "has_text": bool(prose),
                }
            )

        if trajs_out or title:
            parsed_groups.append(
                {
                    "group_idx": g_pos,
                    "title": title or f"Group Rollout {g_pos}",
                    "trajectories": trajs_out,
                }
            )
    return parsed_groups


def _load_logtree_groups(log_dir: Path, warnings: list[str]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    iter_dirs: list[tuple[int, Path]] = []
    for child in log_dir.iterdir():
        if child.is_dir():
            m = _ITER_DIR_RE.match(child.name)
            if m:
                iter_dirs.append((int(m.group(1)), child))
    iter_dirs.sort(key=lambda pair: pair[0])

    for iter_idx, iter_path in iter_dirs:
        lt_file = iter_path / "train_logtree.json"
        if not lt_file.is_file():
            continue
        parsed = _parse_logtree_file(lt_file, warnings)
        if parsed:
            out[iter_idx] = parsed
    return out


def _load_log_summary(log_dir: Path) -> LogSummary | None:
    log_path = log_dir / "logs.log"
    if not log_path.is_file():
        return None
    error_lines = 0
    warn_lines = 0
    retry_lines = 0
    err_samples: list[str] = []
    warn_samples: list[str] = []
    seen: set[str] = set()

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            is_err = bool(_LOG_ERROR_RE.search(stripped))
            is_warn = bool(_LOG_WARN_RE.search(stripped))
            is_retry = bool(_LOG_RETRY_RE.search(stripped))
            if is_err:
                error_lines += 1
            if is_warn:
                warn_lines += 1
            if is_retry:
                retry_lines += 1
            if is_err or is_retry:
                clipped = stripped[:240]
                if clipped not in seen and len(err_samples) < 6:
                    seen.add(clipped)
                    err_samples.append(clipped)
            elif is_warn:
                clipped = stripped[:240]
                if clipped not in seen and len(warn_samples) < 4:
                    seen.add(clipped)
                    warn_samples.append(clipped)

    samples = (err_samples + warn_samples)[:8]
    return LogSummary(
        error_lines=error_lines,
        warn_lines=warn_lines,
        retry_or_rate_limit_lines=retry_lines,
        samples=samples,
    )


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
            raw_attrs = s.get("attributes")
            attrs = dict(raw_attrs) if isinstance(raw_attrs, dict) else {}
            gidx = _to_int(attrs.get("group_idx", s.get("group_idx")))
            spans.append(
                SpanRecord(
                    name=name,
                    duration=dur,
                    wall_start=ws,
                    wall_end=we,
                    group_idx=gidx,
                    attributes=attrs,
                )
            )
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
    discard_rows: list[StepRecord] = []
    for idx, row in enumerate(_iter_jsonl(root / "metrics.jsonl", warnings)):
        rec = _parse_step_record(row, fallback_step=idx)
        # tinker-cookbook main emits:
        #   1. intermediate discard-only cumulative token rows with NO "step" key, and
        #   2. a final post-loop summary row WITH "step" (and no env/optim/time keys).
        is_token_summary_only = (
            not rec.env_metrics
            and not rec.tag_metrics
            and not rec.test_metrics
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
        if is_token_summary_only:
            if row.get("step") is None and row.get("progress/batch") is None:
                discard_rows.append(rec)
            else:
                summary_step = rec
        else:
            step_map[rec.step] = rec
    steps = [step_map[s] for s in sorted(step_map)]

    remove_constant = bool(config.get("remove_constant_reward_groups", False))
    max_tokens = _to_int(config.get("max_tokens"))
    groups_by_step = _load_iteration_summaries(
        root,
        remove_constant,
        warnings,
        max_tokens=max_tokens,
    )
    timings = _load_timings(root, warnings)
    checkpoints = _load_checkpoints(root, warnings)
    log_summary = _load_log_summary(root)
    logtree_groups = _load_logtree_groups(root, warnings)

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
        discard_rows=discard_rows,
        log_summary=log_summary,
        logtree_groups=logtree_groups,
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


def _extract_scalar_metadata(src: dict[str, Any]) -> dict[str, Any]:
    skip = {"prompt", "completion", "messages", "traces", "scores", "metrics", "steps", "error"}
    meta: dict[str, Any] = {}
    for k, v in src.items():
        if k in skip:
            continue
        if isinstance(v, (bool, int, float)):
            if isinstance(v, float) and not math.isfinite(v):
                continue
            meta[str(k)] = v
        elif isinstance(v, str) and len(v) <= 200:
            meta[str(k)] = v
        elif k == "tags" and isinstance(v, list):
            meta["tags"] = ",".join(str(t) for t in v)
    return meta


def _extract_meta(src: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    stop_reason = str(src["stop_reason"]) if src.get("stop_reason") is not None else None
    parse_ok = _to_bool(src.get("parse_ok"))
    if parse_ok is None and "parse_ok" in metrics:
        parse_ok = metrics["parse_ok"] >= 0.5
    truncated_thinking = _to_bool(src.get("truncated_thinking"))
    if truncated_thinking is None and "truncated_thinking" in metrics:
        truncated_thinking = metrics["truncated_thinking"] >= 0.5
    hit_max_turns = _to_bool(src.get("hit_max_turns"))
    if hit_max_turns is None and "hit_max_turns" in metrics:
        hit_max_turns = metrics["hit_max_turns"] >= 0.5
    if hit_max_turns is None and stop_reason == "max_turns":
        hit_max_turns = True
    gold = src.get("gold_answer", src.get("answer"))
    ep_type = src.get("episode_type", src.get("family"))
    graded = _to_bool(src.get("graded_by_env"))
    return {
        "stop_reason": stop_reason,
        "parse_ok": parse_ok,
        "truncated_thinking": truncated_thinking,
        "hit_max_turns": hit_max_turns,
        "gold_answer": str(gold) if gold is not None else None,
        "episode_type": str(ep_type) if ep_type is not None else None,
        "effort": _to_float(src.get("effort")),
        "max_tokens": _to_int(src.get("max_tokens")),
        "max_turns": _to_int(src.get("max_turns")),
        "graded": True if graded is None else graded,
        "metadata": _extract_scalar_metadata(src),
    }


def _messages_from_nodes(raw_nodes: Any) -> list[EvalMessage]:
    if not isinstance(raw_nodes, list):
        return []
    raw_msgs: list[dict[str, Any]] = []
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        msg = node.get("message")
        if isinstance(msg, dict):
            raw_msgs.append(msg)
        elif "role" in node:
            raw_msgs.append(node)
    return _parse_messages(raw_msgs)


def _prime_task_data(row: dict[str, Any]) -> dict[str, Any]:
    task_obj = row.get("task")
    if isinstance(task_obj, dict) and isinstance(task_obj.get("data"), dict):
        return task_obj["data"]
    return {}


def _build_trace_episode(
    row: dict[str, Any],
    trace: dict[str, Any],
    *,
    idx: int,
    top_ok: bool,
    top_errors: list[str],
) -> EvalEpisode:
    task_data = _prime_task_data(row)
    raw_messages = trace.get("messages")
    messages = _parse_messages(raw_messages) if isinstance(raw_messages, list) else []
    if not messages and isinstance(trace.get("nodes"), list):
        messages = _messages_from_nodes(trace.get("nodes"))

    merged_prompt_src = {**task_data, **trace}
    prompt = _prompt_from_trace(merged_prompt_src, messages)
    task_id = trace.get("task_id")
    if task_id is None:
        task_id = task_data.get("id", task_data.get("idx"))
    if task_id is None:
        task_id = row.get("task_id")
    task_key = f"task:{task_id}" if task_id is not None else (prompt or f"row:{idx}")

    split_raw = row.get("split") if row.get("split") is not None else task_data.get("split")
    split = str(split_raw) if split_raw is not None else None

    metrics = _parse_turn_metrics(trace.get("metrics"))
    reward = _extract_trace_reward(trace)
    turns = _to_int(trace.get("sampled_turns"))
    if turns is None:
        turns = _to_int(trace.get("num_turns"))
    if turns is None:
        turns = sum(1 for m in messages if m.role == "assistant")

    trace_errors = [str(e) for e in (trace.get("errors") or [])]
    trace_ok = bool(trace.get("ok", True)) and not bool(trace.get("has_error", False))
    meta_src: dict[str, Any] = {**task_data, **row, **trace}
    raw_stop = trace.get("stop_reason") or trace.get("stop_condition")
    if isinstance(raw_stop, str):
        stop_norm = raw_stop.strip().lower()
        if stop_norm in ("agent_completed", "done", "finished"):
            meta_src["stop_reason"] = "completed"
        elif stop_norm in ("max_turns", "max_turns_reached"):
            meta_src["stop_reason"] = "max_turns"
        elif stop_norm in ("timeout", "rollout_timeout", "timed_out"):
            meta_src["stop_reason"] = "timeout"
        elif stop_norm in ("max_tokens", "length", "token_limit", "prompt_too_long"):
            meta_src["stop_reason"] = "length"
        else:
            meta_src["stop_reason"] = raw_stop
    if _to_bool(trace.get("is_timeout")) is True:
        meta_src["stop_reason"] = "timeout"
    if _to_bool(trace.get("is_truncated")) is True and meta_src.get("stop_reason") is None:
        meta_src.setdefault("truncated_thinking", True)
        meta_src["stop_reason"] = "length"
    if _to_bool(trace.get("is_format_violation")) is True and meta_src.get("parse_ok") is None:
        meta_src["parse_ok"] = False
    meta = _extract_meta(meta_src, metrics)
    return EvalEpisode(
        task_key=task_key,
        task_id=task_id,
        prompt=prompt,
        split=split,
        ok=top_ok and trace_ok and not trace_errors,
        reward=reward,
        metrics=metrics,
        num_turns=turns,
        input_tokens=_to_int(trace.get("num_input_tokens")),
        output_tokens=_to_int(trace.get("num_output_tokens")),
        final_answer_chars=_final_answer_chars(messages),
        messages=messages,
        errors=top_errors + trace_errors,
        **meta,
    )


def load_eval_jsonl_with_warnings(path: str | Path) -> tuple[list[EvalEpisode], list[str]]:
    """Load eval rollouts and return ``(episodes, warnings)`` including skipped malformed lines."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Eval JSONL not found: {p}")

    warnings: list[str] = []
    rows = _iter_jsonl(p, warnings)
    episodes: list[EvalEpisode] = []
    skipped_unknown = 0

    for idx, row in enumerate(rows):
        # Format 1: verifiers v1 / tinker_eval.py / Prime hosted trace JSONL with "traces"
        if isinstance(row.get("traces"), list) and row["traces"]:
            ok = bool(row.get("ok", True))
            top_errors = [str(e) for e in (row.get("errors") or [])]
            for trace in row["traces"]:
                if not isinstance(trace, dict):
                    continue
                episodes.append(
                    _build_trace_episode(
                        row,
                        trace,
                        idx=idx,
                        top_ok=ok,
                        top_errors=top_errors,
                    )
                )
            continue

        # Format 1b: direct Prime hosted single-trace row with top-level "nodes"
        if isinstance(row.get("nodes"), list) and row["nodes"]:
            ok = bool(row.get("ok", True))
            top_errors = [str(e) for e in (row.get("errors") or [])]
            episodes.append(
                _build_trace_episode(
                    row,
                    row,
                    idx=idx,
                    top_ok=ok,
                    top_errors=top_errors,
                )
            )
            continue

        # Format 2: rollout_summaries.jsonl
        if "group_idx" in row and "total_reward" in row:
            gidx = row["group_idx"]
            steps = row.get("steps") or []
            turns_rec = [
                TurnRecord(
                    step_idx=_to_int(s.get("step_idx")) or i,
                    ob_len=_to_int(s.get("ob_len")) or 0,
                    ac_len=_to_int(s.get("ac_len")) or 0,
                    reward=_to_float(s.get("reward")) or 0.0,
                    episode_done=bool(s.get("episode_done", False)),
                    metrics=_parse_turn_metrics(s.get("metrics")),
                )
                for i, s in enumerate(steps)
                if isinstance(s, dict)
            ]
            metrics = _merge_trajectory_metrics(
                turns_rec,
                _parse_turn_metrics(row.get("trajectory_metrics")),
            )
            meta = _extract_meta(row, metrics)
            episodes.append(
                EvalEpisode(
                    task_key=f"group:{gidx}",
                    task_id=gidx,
                    prompt=f"Group {gidx}",
                    split=str(row["split"]) if row.get("split") is not None else None,
                    ok=True,
                    reward=_to_float(row.get("total_reward")) or 0.0,
                    metrics=metrics,
                    num_turns=len(turns_rec),
                    input_tokens=sum(t.ob_len for t in turns_rec),
                    output_tokens=sum(t.ac_len for t in turns_rec),
                    final_answer_chars=0,
                    messages=[],
                    **meta,
                )
            )
            continue

        # Format 3: tinker-cookbook git_tool_rl / native eval JSONL with "scores" dict
        if isinstance(row.get("scores"), dict) and (
            "messages" in row or "prompt" in row or "task_id" in row
        ):
            messages = _parse_messages(row.get("messages"))
            prompt = _prompt_from_trace(row, messages)
            task_id = row.get("task_id")
            task_key = f"task:{task_id}" if task_id is not None else (prompt or f"row:{idx}")
            scores = row["scores"]
            metrics = _parse_turn_metrics(scores)
            reward = _to_float(scores.get("shaped_reward"))
            if reward is None:
                reward = _to_float(scores.get("outcome"))
            if reward is None:
                reward = _to_float(scores.get("success"))
            if reward is None:
                reward = _to_float(scores.get("correct"))
            if reward is None:
                reward = _to_float(row.get("reward")) or 0.0
            err = row.get("error")
            ok = bool(row.get("messages") is not None or row.get("graded_by_env")) and not bool(err)
            meta = _extract_meta(row, metrics)
            if not scores and not row.get("graded_by_env", False):
                meta["graded"] = False
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
                    **meta,
                )
            )
            continue

        # Format 4: legacy vf-eval results.jsonl (strictly require eval-shaped keys!)
        is_legacy_vf = (
            "completion" in row
            or ("prompt" in row and ("reward" in row or "metrics" in row))
            or ("example_id" in row and ("reward" in row or "metrics" in row))
        )
        if is_legacy_vf:
            messages = _parse_messages(row.get("prompt")) + _parse_messages(row.get("completion"))
            prompt = _prompt_from_trace(row, messages)
            task_id = row.get("example_id", row.get("task_id"))
            task_key = f"task:{task_id}" if task_id is not None else (prompt or f"row:{idx}")
            metrics = _parse_turn_metrics(row.get("metrics"))
            err = row.get("error")
            meta = _extract_meta(row, metrics)
            episodes.append(
                EvalEpisode(
                    task_key=task_key,
                    task_id=task_id,
                    prompt=prompt,
                    split=str(row["split"]) if row.get("split") is not None else None,
                    ok=not bool(err),
                    reward=_to_float(row.get("reward")) or 0.0,
                    metrics=metrics,
                    num_turns=sum(1 for m in messages if m.role == "assistant"),
                    input_tokens=_to_int(row.get("num_input_tokens", row.get("input_tokens"))),
                    output_tokens=_to_int(row.get("num_output_tokens", row.get("output_tokens"))),
                    final_answer_chars=_final_answer_chars(messages),
                    messages=messages,
                    errors=[str(err)] if err else [],
                    **meta,
                )
            )
            continue

        skipped_unknown += 1

    if skipped_unknown > 0 and episodes:
        warnings.append(
            f"Skipped {skipped_unknown} unrecognized row(s) in {p.name} "
            "(did not match a known eval format)."
        )

    if not episodes:
        detail = ""
        if skipped_unknown:
            detail = f" ({skipped_unknown} rows did not match any known eval format)"
        elif warnings:
            detail = f" ({warnings[0]})"
        raise ValueError(
            f"0 eval episodes parsed from {p.name}{detail}. "
            "Expected verifiers traces, git_tool_rl scores, rollout_summaries, or vf-eval JSONL."
        )

    return episodes, warnings


def load_eval_jsonl(path: str | Path) -> list[EvalEpisode]:
    """Load eval rollouts from verifiers v1, git_tool_rl, rollout summaries, or vf-eval JSONL."""
    episodes, _ = load_eval_jsonl_with_warnings(path)
    return episodes
