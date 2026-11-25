"""Tests for tinker_inspect.loader."""

from __future__ import annotations

import json
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

    # Fallback reward to outcome when shaped_reward is absent
    assert eps[1].reward == pytest.approx(0.6)
    assert eps[1].ok is True
    assert eps[1].num_turns == 1
    assert eps[1].final_answer_chars == 0
