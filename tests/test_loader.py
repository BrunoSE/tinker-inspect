"""Tests for tinker_inspect.loader."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from fixtures import make_synthetic_eval_pair, make_synthetic_run

from tinker_inspect.loader import load_eval_jsonl, load_run


def test_load_run_parses_all_artifacts(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run1", num_steps=3, with_tags=True)
    run = load_run(run_dir)
    assert run.name == "run1"
    assert len(run.steps) == 3
    assert len(run.checkpoints) == 1
    assert len(run.timings) == 3
    assert len(run.groups_by_step) == 3
    assert "tag_0" in run.steps[0].tag_metrics
    assert run.code_diff and "+x = 1" in run.code_diff
    assert run.warnings == []


def test_load_run_all_constant_fallback_keeps_group_zero(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_const", num_steps=2, all_constant_step=1)
    run = load_run(run_dir)
    g_step1 = run.groups_by_step[1]
    assert len(g_step1) == 2
    assert g_step1[0].kept_after_filter is True
    assert g_step1[1].kept_after_filter is False


def test_load_run_tolerates_corrupt_jsonl_line(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_corrupt", num_steps=2)
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write("{broken json\n")
    run = load_run(run_dir)
    assert len(run.steps) == 2
    assert len(run.warnings) == 1
    assert "metrics.jsonl" in run.warnings[0]


def test_load_run_raises_on_empty_dir(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No metrics.jsonl"):
        load_run(empty)


def test_load_run_merges_step_metrics_and_separates_discard_rows(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_inkling_shape", num_steps=2)
    # Rewrite iteration 0 rollout summary to match real Inkling shape:
    # trajectory_metrics={}, metrics live inside steps[-1]["metrics"], final_ob_len=0.
    s0_path = run_dir / "iteration_000000" / "train_rollout_summaries.jsonl"
    rows = [json.loads(line) for line in s0_path.read_text().splitlines() if line.strip()]
    for i, r in enumerate(rows):
        r["trajectory_metrics"] = {}
        r["final_ob_len"] = 0
        r["steps"][-1]["metrics"] = {
            "correct": float(r["total_reward"]),
            "used_python": 1.0 if i % 2 == 0 else 0.0,
            "tool_calls_count": 2.0 if i % 2 == 0 else 0.0,
            "tool_errors": 1.0 if i == 0 else 0.0,
            "truncated_thinking": 1.0 if i == 1 else 0.0,
            "hit_max_turns": 1.0 if i == 2 else 0.0,
            "parse_ok": 1.0,
        }
    s0_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # Add intermediate cumulative discard rows (no "step" key) + final summary row (with "step")
    # + optim mismatch and test/* metrics.
    m_path = run_dir / "metrics.jsonl"
    m_rows = [json.loads(line) for line in m_path.read_text().splitlines() if line.strip()]
    m_rows[0]["optim/sample_train_mismatch_frac"] = 0.001
    m_rows[0]["optim/sample_train_max_abs_diff"] = 1e-5
    m_rows[0]["test/polaris/success"] = 0.45
    combined = [
        {
            "tokens/cumulative/sampled": 100,
            "tokens/discarded/sampled": 100,
            "tokens/discarded/prefill_cached": 50,
            "tokens/discarded/prefill_uncached": 50,
        },
        m_rows[0],
        {
            "tokens/cumulative/sampled": 400,
            "tokens/discarded/sampled": 250,
            "tokens/discarded/prefill_cached": 120,
            "tokens/discarded/prefill_uncached": 130,
        },
        m_rows[1],
        {
            "step": 2,
            "tokens/cumulative/sampled": 900,
            "tokens/discarded/sampled": 300,
            "tokens/discarded/prefill_cached": 150,
            "tokens/discarded/prefill_uncached": 150,
            "cost/cumulative_usd": 0.0,
        },
    ]
    m_path.write_text("\n".join(json.dumps(r) for r in combined) + "\n", encoding="utf-8")

    # Add logs.log and train_logtree.json
    (run_dir / "logs.log").write_text(
        "\n".join(
            [
                "    tool_errors: 0.0",
                "tinker.lib [WARNING] Synchronous method called from async context",
                "tinker.lib [ERROR] HTTP 429 rate limit exceeded, retrying after backoff",
                "Traceback (most recent call last):",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logtree = {
        "title": "Training Log",
        "root": {
            "tag": "body",
            "attrs": {},
            "children": [
                {
                    "tag": "section",
                    "attrs": {"class": "lt-section"},
                    "children": [
                        {"tag": "h2", "attrs": {}, "children": ["Group Rollout"]},
                        {
                            "tag": "div",
                            "attrs": {},
                            "children": [
                                {
                                    "tag": "section",
                                    "attrs": {},
                                    "children": [
                                        {
                                            "tag": "h4",
                                            "attrs": {},
                                            "children": ["Trajectory 0 Episode"],
                                        },
                                        {
                                            "tag": "pre",
                                            "attrs": {},
                                            "children": ["Assistant solved with run_python."],
                                        },
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ],
        },
    }
    (run_dir / "iteration_000000" / "train_logtree.json").write_text(
        json.dumps(logtree), encoding="utf-8"
    )

    run = load_run(run_dir)
    assert len(run.steps) == 2
    assert len(run.discard_rows) == 2
    assert run.summary_step is not None and run.summary_step.step == 2
    assert run.steps[0].sample_train_mismatch_frac == pytest.approx(0.001)
    assert run.steps[0].sample_train_max_abs_diff == pytest.approx(1e-5)
    assert run.steps[0].test_metrics["polaris/success"] == pytest.approx(0.45)

    g0 = run.groups_by_step[0][0]
    assert g0.trajectories[0].trajectory_metrics["used_python"] == 1.0
    assert g0.trajectories[0].trajectory_metrics["tool_errors"] == 1.0
    assert g0.trunc_episodes == 1
    assert g0.tool_errors_total == pytest.approx(1.0)
    assert run.groups_by_step[0][1].max_turns_episodes == 1

    assert run.log_summary is not None
    assert run.log_summary.error_lines == 2
    assert run.log_summary.warn_lines == 1
    assert run.log_summary.retry_or_rate_limit_lines == 1

    assert 0 in run.logtree_groups
    assert len(run.logtree_groups[0]) == 1
    assert "run_python" in run.logtree_groups[0][0]["trajectories"][0]["excerpt"]


def test_load_run_parses_timing_span_group_idx_attributes(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_span_gid", num_steps=1)
    tp = run_dir / "timing_spans.jsonl"
    row = json.loads(tp.read_text().splitlines()[0])
    gid = 0
    for sp in row["spans"]:
        if sp["name"] == "do_group_rollout_and_filter_constant_reward":
            sp["attributes"] = {"group_idx": gid}
            gid += 1
    tp.write_text(json.dumps(row) + "\n", encoding="utf-8")

    run = load_run(run_dir)
    rollout_spans = [
        sp
        for sp in run.timings[0].spans
        if sp.name == "do_group_rollout_and_filter_constant_reward"
    ]
    assert [sp.group_idx for sp in rollout_spans] == [0, 1, 2, 3]
    assert rollout_spans[2].attributes == {"group_idx": 2}


def test_load_eval_jsonl_formats(tmp_path: Path):
    base_p, _ = make_synthetic_eval_pair(tmp_path / "evals")
    eps = load_eval_jsonl(base_p)
    assert len(eps) == 8
    assert eps[0].task_key == "task:0"
    assert eps[0].reward == pytest.approx(0.0)
    assert eps[0].final_answer_chars == len("Clean.")

    # Legacy vf-eval results.jsonl format
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps(
            {
                "example_id": 7,
                "prompt": [{"role": "user", "content": "Solve 2+2"}],
                "completion": [{"role": "assistant", "content": "4"}],
                "reward": 1.0,
                "metrics": {"correct": 1.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    legacy_eps = load_eval_jsonl(legacy)
    assert len(legacy_eps) == 1
    assert legacy_eps[0].task_key == "task:7"
    assert legacy_eps[0].reward == 1.0
    assert legacy_eps[0].final_answer_chars == 1


def test_load_eval_jsonl_rejects_metrics_and_zst(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_m", num_steps=2)
    with pytest.raises(ValueError, match="0 eval episodes parsed"):
        load_eval_jsonl(run_dir / "metrics.jsonl")

    zst_file = tmp_path / "rollouts.jsonl.zst"
    zst_file.write_bytes(b"\x28\xb5\x2f\xfd\x00\x00\x00")
    with pytest.raises(ValueError, match="Compressed .zst"):
        load_eval_jsonl(zst_file)


def test_load_run_separates_post_loop_summary_row(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_summary", num_steps=3, native_cost=True)
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "step": 3,
                    "tokens/cumulative/sampled": 9999,
                    "tokens/discarded/sampled": 500,
                    "cost/cumulative_usd": 0.0,
                    "cost/discarded_usd": 0.0,
                }
            )
            + "\n"
        )
    run = load_run(run_dir)
    assert len(run.steps) == 3
    assert run.summary_step is not None
    assert run.summary_step.native_discarded_tokens["sampled"] == 500


def test_load_eval_jsonl_git_tool_rl_format(tmp_path: Path):
    path = tmp_path / "git_tool_rl.jsonl"
    rows = [
        {
            "task_id": "94cd2f12bd1f",
            "split": "heldout",
            "level": 1,
            "episode_type": "read",
            "prompt": "Check the current status of the repository.",
            "gold_answer": "clean",
            "effort": 0.5,
            "max_tokens": 2048,
            "max_turns": 4,
            "parse_ok": True,
            "truncated_thinking": False,
            "scores": {
                "shaped_reward": 1.0,
                "outcome": 1.0,
                "outcome_facts": 1.0,
                "success": 1.0,
                "success_facts": 1.0,
            },
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "functions.git_status:0",
                            "type": "function",
                            "function": {"name": "git_status", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "working tree clean",
                    "tool_call_id": "functions.git_status:0",
                },
                {"role": "assistant", "content": "The working tree is clean.", "tool_calls": None},
            ],
            "stop_reason": "completed",
            "graded_by_env": True,
        },
        {
            "task_id": "aabbccddeeff",
            "split": "l5_new_tools",
            "level": 5,
            "episode_type": "l5",
            "prompt": "Merge feature-b into main.",
            "scores": {
                "outcome": 0.6,
                "outcome_facts": 0.6,
                "success": 0.0,
                "success_facts": 0.0,
            },
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": []},
            ],
            "stop_reason": "max_turns",
            "graded_by_env": False,
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    eps = load_eval_jsonl(path)
    assert len(eps) == 2
    assert eps[0].task_key == "task:94cd2f12bd1f"
    assert eps[0].split == "heldout"
    assert eps[0].ok is True
    assert eps[0].reward == pytest.approx(1.0)
    assert eps[0].metrics["success"] == pytest.approx(1.0)
    assert eps[0].metrics["outcome_facts"] == pytest.approx(1.0)
    assert eps[0].num_turns == 2
    assert eps[0].final_answer_chars == len("The working tree is clean.")
    assert eps[0].messages[0].tool_calls == [{"name": "git_status", "arguments": "{}"}]
    assert eps[0].gold_answer == "clean"
    assert eps[0].episode_type == "read"
    assert eps[0].effort == pytest.approx(0.5)
    assert eps[0].max_tokens == 2048
    assert eps[0].end_state == "completed"

    # Fallback reward to outcome when shaped_reward is absent
    assert eps[1].reward == pytest.approx(0.6)
    assert eps[1].ok is True
    assert eps[1].num_turns == 1
    assert eps[1].final_answer_chars == 0
    assert eps[1].end_state == "turn_limit"
    assert eps[1].graded is False


def test_load_eval_jsonl_zstd_fallback_and_missing_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    base_p, _ = make_synthetic_eval_pair(tmp_path / "evals")
    zst_path = tmp_path / "base_compressed.jsonl.zst"
    zstd_bin = shutil.which("zstd")
    if zstd_bin is not None:
        proc = subprocess.run(
            [zstd_bin, "-q", "-f", str(base_p), "-o", str(zst_path)],
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0
        eps = load_eval_jsonl(zst_path)
        assert len(eps) == 8
        assert eps[0].task_key == "task:0"

        # Simulate missing zstd binary on Python <3.14
        monkeypatch.setattr("tinker_inspect.loader.shutil.which", lambda _name: None)
        with pytest.raises(ValueError, match="Compressed .zst"):
            load_eval_jsonl(zst_path)
    else:
        zst_path.write_bytes(b"\x28\xb5\x2f\xfd\x00\x00\x00")
        with pytest.raises(ValueError, match="Compressed .zst"):
            load_eval_jsonl(zst_path)


def test_load_eval_jsonl_prime_hosted_trace_format(tmp_path: Path):
    prime_file = tmp_path / "prime_traces.jsonl"
    rows = [
        {
            "id": "ep_001",
            "attempt": 0,
            "ok": True,
            "errors": [],
            "task": {
                "type": "GitTask",
                "data": {
                    "id": "8245edc8d885",
                    "prompt": "Read settings.ini and copy token to out.txt",
                    "split": "fs_basic",
                    "family": "copy-value",
                    "episode_type": "fs",
                    "level": 5,
                },
            },
            "traces": [
                {
                    "id": "tr_001",
                    "ok": True,
                    "reward": 1.0,
                    "rewards": {"shaped_reward": {"score": 1.0, "weight": 1.0}},
                    "metrics": {"outcome": 1.0, "success": 1.0},
                    "num_input_tokens": 320,
                    "num_output_tokens": 45,
                    "num_turns": 2,
                    "stop_condition": "completed",
                    "is_truncated": False,
                    "is_format_violation": False,
                    "has_error": False,
                    "errors": [],
                    "nodes": [
                        {"message": {"role": "system", "content": "You are an assistant."}},
                        {"message": {"role": "user", "content": "Read settings.ini"}},
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {"function": {"name": "read_file", "arguments": "{}"}}
                                ],
                            }
                        },
                        {"message": {"role": "tool", "content": "token=abc"}},
                        {"message": {"role": "assistant", "content": "Done."}},
                    ],
                }
            ],
        },
        {
            "id": "ep_002",
            "ok": True,
            "task": {
                "data": {
                    "id": "9999deadbeef",
                    "prompt": "Direct node trace prompt",
                    "split": "fs_hard",
                    "family": "merge-patch",
                }
            },
            "reward": 0.0,
            "metrics": {"outcome": 0.0, "success": 0.0},
            "num_turns": 1,
            "is_truncated": True,
            "nodes": [
                {"message": {"role": "user", "content": "Direct node trace prompt"}},
                {"message": {"role": "assistant", "content": "Thinking..."}},
            ],
        },
    ]
    prime_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    eps = load_eval_jsonl(prime_file)
    assert len(eps) == 2
    assert eps[0].task_key == "task:8245edc8d885"
    assert eps[0].split == "fs_basic"
    assert eps[0].episode_type == "fs"
    assert eps[0].metadata["family"] == "copy-value"
    assert eps[0].num_turns == 2
    assert eps[0].input_tokens == 320
    assert eps[0].output_tokens == 45
    assert eps[0].final_answer_chars == len("Done.")
    assert eps[0].end_state == "completed"
    assert eps[0].metrics["success"] == pytest.approx(1.0)

    assert eps[1].task_key == "task:9999deadbeef"
    assert eps[1].split == "fs_hard"
    assert eps[1].metadata["family"] == "merge-patch"
    assert eps[1].end_state == "token_limit"

    # Optional external Prime hosted .jsonl.zst smoke test via env var (never hard-coded paths)
    env_prime = os.environ.get("TINKER_INSPECT_PRIME_TRACE_ZST")
    if env_prime and Path(env_prime).is_file() and shutil.which("zstd"):
        real_eps = load_eval_jsonl(env_prime)
        assert len(real_eps) > 0
        assert real_eps[0].prompt
        assert real_eps[0].messages


def test_prime_timeout_truncated_completed_and_multisplit_warnings(tmp_path: Path):
    from tinker_inspect.diagnostics import diff_evals
    from tinker_inspect.series import analyze_eval_series

    def _row(task_id: str, split: str, *, success: float, **extra) -> dict:
        return {
            "id": f"ep_{task_id}",
            "ok": True,
            "task": {"data": {"id": task_id, "prompt": f"Solve {task_id}", "split": split}},
            "reward": success,
            "metrics": {"success": success},
            "num_turns": 2,
            "nodes": [{"message": {"role": "assistant", "content": "done"}}],
            **extra,
        }

    base_p = tmp_path / "base_multi.jsonl"
    final_p = tmp_path / "final_multi.jsonl"
    base_rows = [
        _row("t1", "split_a", success=0.0, is_timeout=True),
        _row(
            "t2",
            "split_b",
            success=1.0,
            stop_condition="agent_completed",
            is_truncated=True,
        ),
    ]
    final_rows = [
        _row("t1", "split_a", success=1.0, stop_condition="agent_completed"),
        _row("t2", "split_b", success=1.0, stop_condition="agent_completed"),
    ]
    base_p.write_text("\n".join(json.dumps(r) for r in base_rows) + "\n", encoding="utf-8")
    final_p.write_text("\n".join(json.dumps(r) for r in final_rows) + "\n", encoding="utf-8")

    base_eps = load_eval_jsonl(base_p)
    final_eps = load_eval_jsonl(final_p)
    assert base_eps[0].end_state == "timeout"
    assert base_eps[1].end_state == "completed"
    assert base_eps[1].metadata.get("is_truncated") is True

    diff = diff_evals(base_eps, final_eps, base_label="Base", final_label="Final")
    assert any("Multiple splits pooled" in w for w in diff.warnings)
    assert any("is_truncated=True" in w for w in diff.warnings)

    series = analyze_eval_series([("Base", base_eps), ("Final", final_eps)])
    assert any("Multiple splits pooled" in w for w in series.warnings)
