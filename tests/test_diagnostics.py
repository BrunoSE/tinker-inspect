"""Tests for tinker_inspect.diagnostics and pricing."""

from __future__ import annotations

from pathlib import Path

import pytest
from fixtures import make_synthetic_eval_pair, make_synthetic_run

from tinker_inspect.diagnostics import analyze_run, diff_evals
from tinker_inspect.loader import load_eval_jsonl, load_run
from tinker_inspect.pricing import parse_rates_spec


def test_analyze_run_reconstructed_tokens_and_drop_rate(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_rec", num_steps=4, remove_constant=True)
    run = load_run(run_dir)
    rates = parse_rates_spec("0.50,0.10,1.50,1.00")
    diag = analyze_run(run, rates=rates)

    assert diag.max_abs_kl_v1 == 0.0
    assert diag.first_nonzero_kl_step is None
    assert diag.total_groups_sampled == 16
    assert diag.total_groups_kept == 4
    assert diag.total_groups_all_good == 8
    assert diag.total_groups_all_bad == 4
    assert diag.discarded_group_fraction == pytest.approx(0.75)

    # Multi-turn prefill multiplier: sum(1000+1150+1300)=3450 vs final_ob_len=1360 -> 2.5368x
    assert diag.tokens.source == "reconstructed"
    assert diag.tokens.prefill_multiplier == pytest.approx(3450 / 1360, rel=1e-3)
    assert diag.tokens.discarded_token_fraction == pytest.approx(0.75)
    assert diag.tokens.estimated_cost_usd is not None and diag.tokens.estimated_cost_usd > 0
    assert diag.tokens.estimated_discarded_cost_usd is not None

    kl_finding = next(f for f in diag.findings if f.check_id == "kl0")
    assert kl_finding.severity == "pass"
    drop_finding = next(f for f in diag.findings if f.check_id == "constant_groups")
    assert drop_finding.severity == "warn"


def test_analyze_run_native_cost_and_kl_failure(tmp_path: Path):
    run_dir = make_synthetic_run(
        tmp_path / "run_native",
        num_steps=4,
        native_cost=True,
        kl_v1_by_step=[0.0, 0.0, 0.0025, 0.001],
    )
    run = load_run(run_dir)
    diag = analyze_run(run)

    assert diag.tokens.source == "native"
    assert diag.tokens.estimated_cost_usd == pytest.approx(1.60)
    assert diag.tokens.pricing_source == "metrics.jsonl (native)"
    assert diag.first_nonzero_kl_step == 2
    kl_finding = next(f for f in diag.findings if f.check_id == "kl0")
    assert kl_finding.severity == "fail"


def test_analyze_run_detects_verbosity_and_metric_divergence(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_drift", num_steps=4, length_drift=True)
    run = load_run(run_dir)
    diag = analyze_run(run)
    drift = next(f for f in diag.findings if f.check_id == "verbosity_drift")
    assert drift.severity == "warn"
    assert "outcome" in drift.headline


def test_diff_evals_paired_deltas_and_bootstrap_ci(tmp_path: Path):
    base_p, final_p = make_synthetic_eval_pair(tmp_path / "evals")
    diff = diff_evals(load_eval_jsonl(base_p), load_eval_jsonl(final_p), n_boot=500, seed=42)

    assert diff.num_tasks == 4
    assert diff.primary_metric == "success"
    success = next(m for m in diff.metric_deltas if m.metric == "success")
    assert success.base_mean == pytest.approx(0.5)
    assert success.final_mean == pytest.approx(0.75)
    assert success.delta == pytest.approx(0.25)
    assert success.wins == 2
    assert success.regressions == 1
    assert success.ties == 1
    assert success.ci_low <= success.delta <= success.ci_high
    assert diff.answer_chars_change_pct is not None and diff.answer_chars_change_pct > 50.0


def test_single_turn_zero_final_ob_len_fallback():
    from tinker_inspect.schema import TrajectoryRecord, TurnRecord

    traj = TrajectoryRecord(
        iteration=0,
        group_idx=0,
        traj_idx=0,
        tags=[],
        total_reward=0.0,
        final_reward=0.0,
        trajectory_metrics={},
        steps=[TurnRecord(step_idx=0, ob_len=40, ac_len=32, episode_done=True)],
        final_ob_len=0,
    )
    assert traj.unique_tokens == 72


def test_cumulative_rollout_errors_and_zero_reward_on_limit(tmp_path: Path):
    import json

    run_dir = make_synthetic_run(tmp_path / "run_inkling", num_steps=3)
    mpath = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in mpath.read_text().splitlines() if line.strip()]
    for i, r in enumerate(rows):
        # RolloutErrorCounter is cumulative: 1 -> 2 -> 2 (total should be 2, not 5)
        r["rollout_errors/total"] = 1.0 if i == 0 else 2.0
        r["rollout_errors/groups_skipped"] = 1.0 if i >= 1 else 0.0
        r["env/all/zero_reward_on_limit"] = 0.20
    mpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    diag = analyze_run(load_run(run_dir))
    err_f = next(f for f in diag.findings if f.check_id == "rollout_errors")
    assert err_f.metrics["total_rollout_errors"] == 2
    assert err_f.metrics["total_groups_skipped"] == 1
    clamp_f = next(f for f in diag.findings if f.check_id == "zero_reward_on_limit")
    assert clamp_f.severity == "warn"
    assert clamp_f.metrics["mean_clamped_frac"] == pytest.approx(0.20)


def test_diff_evals_prompt_fallback_pairing(tmp_path: Path):
    import json

    # Base uses verifiers v1 format with integer task_id=0,1; final uses git_tool_rl md5 hex task_id
    base_path = tmp_path / "base.jsonl"
    final_path = tmp_path / "final.jsonl"
    base_rows = [
        {
            "split": "heldout",
            "ok": True,
            "errors": [],
            "traces": [
                {
                    "task_id": 0,
                    "reward": 0.0,
                    "metrics": {"success": 0.0, "outcome": 0.5},
                    "sampled_turns": 2,
                    "messages": [
                        {"role": "user", "content": "Check   the status of the repo."},
                        {"role": "assistant", "content": "Dirty."},
                    ],
                }
            ],
        },
        {
            "split": "heldout",
            "ok": True,
            "errors": [],
            "traces": [
                {
                    "task_id": 1,
                    "reward": 1.0,
                    "metrics": {"success": 1.0, "outcome": 1.0},
                    "sampled_turns": 1,
                    "messages": [
                        {"role": "user", "content": "List all branches."},
                        {"role": "assistant", "content": "main, feature-a"},
                    ],
                }
            ],
        },
    ]
    final_rows = [
        {
            "task_id": "94cd2f12bd1f",
            "split": "heldout",
            "prompt": "Check the status\nof the repo.",
            "scores": {"shaped_reward": 1.0, "success": 1.0, "outcome": 1.0},
            "messages": [
                {"role": "assistant", "content": "Clean working tree."},
            ],
            "stop_reason": "completed",
            "graded_by_env": True,
        },
        {
            "task_id": "112233445566",
            "split": "heldout",
            "prompt": "List all branches.",
            "scores": {"shaped_reward": 1.0, "success": 1.0, "outcome": 1.0},
            "messages": [
                {"role": "assistant", "content": "main and feature-a"},
            ],
            "stop_reason": "completed",
            "graded_by_env": True,
        },
    ]
    base_path.write_text("\n".join(json.dumps(r) for r in base_rows) + "\n", encoding="utf-8")
    final_path.write_text("\n".join(json.dumps(r) for r in final_rows) + "\n", encoding="utf-8")

    diff = diff_evals(load_eval_jsonl(base_path), load_eval_jsonl(final_path), n_boot=200, seed=0)
    assert diff.num_tasks == 2
    success = next(m for m in diff.metric_deltas if m.metric == "success")
    assert success.base_mean == pytest.approx(0.5)
    assert success.final_mean == pytest.approx(1.0)
    assert success.delta == pytest.approx(0.5)
    assert success.wins == 1
    assert success.ties == 1
    assert success.regressions == 0
