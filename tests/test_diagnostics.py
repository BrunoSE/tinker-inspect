"""Tests for tinker_inspect.diagnostics, stats, and pricing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fixtures import make_synthetic_eval_pair, make_synthetic_run

from tinker_inspect.diagnostics import analyze_run, diff_evals
from tinker_inspect.loader import load_eval_jsonl, load_run
from tinker_inspect.pricing import parse_rates_spec
from tinker_inspect.schema import EvalEpisode, TrajectoryRecord, TurnRecord
from tinker_inspect.stats import bootstrap_ci_mean_diff


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

    # Multi-turn prefill multiplier uses last_ob_len (1300): sum(1000+1150+1300)/1300 = 2.6538x
    assert diag.tokens.source == "reconstructed"
    assert diag.tokens.prefill_multiplier == pytest.approx(3450 / 1300, rel=1e-3)
    assert diag.tokens.prefill_multiplier >= 1.0
    assert diag.tokens.cache_metric_label == "Est. prefix-cacheable"
    assert diag.tokens.discarded_token_fraction == pytest.approx(0.75)
    assert diag.tokens.discarded_sampled_only_fraction == pytest.approx(0.75)
    assert diag.tokens.estimated_cost_usd is not None and diag.tokens.estimated_cost_usd > 0
    assert diag.tokens.estimated_discarded_cost_usd is not None

    # Pre-filter step summary & window progress
    assert diag.reward_source == "prefilter"
    assert len(diag.prefilter_steps) == 4
    assert diag.reward_window_size == 1
    assert diag.reward_early_mean == pytest.approx(0.625)
    assert diag.reward_late_mean == pytest.approx(0.625)

    # Sampling idle wait = 1 - mean/max
    assert diag.sampling_idle_fraction is not None
    assert 0.0 < diag.sampling_idle_fraction < 1.0

    kl_finding = next(f for f in diag.findings if f.check_id == "kl0")
    assert kl_finding.severity == "pass"
    drop_finding = next(f for f in diag.findings if f.check_id == "constant_groups")
    assert drop_finding.severity == "warn"
    # Truncation finding is always emitted (PASS when < 5%)
    trunc_finding = next(f for f in diag.findings if f.check_id == "truncation")
    assert trunc_finding.severity == "pass"


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
    assert diag.tokens.cache_metric_label == "Measured cache hits"
    assert diag.tokens.estimated_cost_usd == pytest.approx(1.60)
    assert diag.tokens.pricing_source == "metrics.jsonl (native)"
    assert diag.first_nonzero_kl_step == 2
    kl_finding = next(f for f in diag.findings if f.check_id == "kl0")
    assert kl_finding.severity == "fail"


def test_analyze_run_kl_config_aware_off_policy(tmp_path: Path):
    run_dir = make_synthetic_run(
        tmp_path / "run_offpolicy",
        num_steps=3,
        kl_v1_by_step=[0.0, 0.001, 0.002],
    )
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["num_substeps"] = 2
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    diag = analyze_run(load_run(run_dir))
    kl_finding = next(f for f in diag.findings if f.check_id == "kl0")
    assert kl_finding.severity == "info"


def test_analyze_run_detects_verbosity_and_metric_divergence(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_drift", num_steps=4, length_drift=True)
    run = load_run(run_dir)
    diag = analyze_run(run)
    drift = next(f for f in diag.findings if f.check_id == "verbosity_drift")
    assert drift.severity == "warn"
    assert "outcome" in drift.headline


def test_prefilter_vs_postfilter_tool_and_reward_separation(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_pf", num_steps=4, remove_constant=True)
    # Give rollout summaries distinct pre-filter tool rates (25% used_python) while
    # metrics.jsonl reports inflated post-filter rates (90% used_python).
    for step in range(4):
        sp = run_dir / f"iteration_{step:06d}" / "train_rollout_summaries.jsonl"
        rows = [json.loads(line) for line in sp.read_text().splitlines() if line.strip()]
        for i, r in enumerate(rows):
            r["trajectory_metrics"] = {}
            r["final_ob_len"] = 0
            r["steps"][-1]["metrics"] = {
                "used_python": 1.0 if i < 2 else 0.0,
                "tool_calls_count": 2.0 if i < 2 else 0.0,
                "tool_errors": 0.0,
                "hit_max_turns": 1.0 if i == 0 else 0.0,
                "truncated_thinking": 1.0 if i == 1 else 0.0,
            }
            if i == 1:
                r["total_reward"] = 0.0
        sp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    mp = run_dir / "metrics.jsonl"
    mrows = [json.loads(line) for line in mp.read_text().splitlines() if line.strip()]
    for r in mrows:
        r["env/all/used_python"] = 0.90
        r["env/all/hit_max_turns"] = 0.50
    mp.write_text("\n".join(json.dumps(r) for r in mrows) + "\n", encoding="utf-8")

    diag = analyze_run(load_run(run_dir))
    assert diag.tools is not None
    assert diag.tools.source == "prefilter"
    assert diag.tools.used_tool_rate == pytest.approx(0.25)
    assert diag.tools.hit_max_turns_rate == pytest.approx(0.125)
    assert diag.tools.trunc_episode_rate == pytest.approx(0.125)
    assert diag.tools.trunc_reward_mean == pytest.approx(0.0)
    assert diag.tools.untrunc_reward_mean is not None and diag.tools.untrunc_reward_mean > 0.5
    trunc_f = next(f for f in diag.findings if f.check_id == "truncation")
    assert trunc_f.severity == "warn"


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


def test_diff_evals_counts_ungraded_as_zero_and_reports_unmatched_and_config_mismatch():
    def _ep(
        tid: str,
        *,
        success: float | None,
        ok: bool = True,
        graded: bool = True,
        stop_reason: str = "completed",
        truncated: bool = False,
        max_tokens: int = 2048,
        ep_type: str = "alg",
    ) -> EvalEpisode:
        metrics = {"success": success} if success is not None else {}
        return EvalEpisode(
            task_key=f"task:{tid}",
            task_id=tid,
            prompt=f"Solve {tid}",
            split="test",
            ok=ok,
            reward=success or 0.0,
            metrics=metrics,
            num_turns=1,
            input_tokens=100,
            output_tokens=200,
            final_answer_chars=10,
            messages=[],
            stop_reason=stop_reason,
            truncated_thinking=truncated,
            max_tokens=max_tokens,
            episode_type=ep_type,
            graded=graded,
        )

    base = [
        _ep("1", success=1.0, max_tokens=4096, ep_type="A"),
        _ep("1", success=None, ok=False, graded=False, max_tokens=4096, ep_type="A"),
        _ep("2", success=0.0, truncated=True, max_tokens=4096, ep_type="B"),
        _ep("3", success=0.0, stop_reason="parse_error", max_tokens=4096, ep_type="C"),
        _ep("4", success=None, ok=True, graded=False, max_tokens=4096, ep_type="D"),
        _ep("5", success=1.0, stop_reason="max_turns", max_tokens=4096, ep_type="E"),
        _ep("only_base", success=1.0, max_tokens=4096, ep_type="E"),
    ]
    final = [
        _ep("1", success=1.0, max_tokens=3072, ep_type="A"),
        _ep("2", success=1.0, max_tokens=3072, ep_type="B"),
        _ep("3", success=1.0, max_tokens=3072, ep_type="C"),
        _ep("4", success=1.0, max_tokens=3072, ep_type="D"),
        _ep("5", success=1.0, max_tokens=3072, ep_type="E"),
        _ep("only_final", success=0.0, max_tokens=3072, ep_type="E"),
    ]

    diff = diff_evals(base, final, cluster_by="episode_type", n_boot=500, seed=42)
    assert diff.num_tasks == 5
    assert diff.num_clusters == 5
    assert diff.base_unmatched_tasks == 1
    assert diff.final_unmatched_tasks == 1
    # Task 1 base has [1.0, missing->0.0] => 0.5 (NOT 1.0!)
    t1 = next(t for t in diff.tasks if t.task_id == "1")
    assert t1.base_metrics["success"] == pytest.approx(0.5)
    assert t1.final_metrics["success"] == pytest.approx(1.0)
    # Verify end_state classification: parse_error, ungraded, turn_limit (state-graded success!)
    assert diff.base_end_states.get("parse_error") == 1
    assert diff.base_end_states.get("ungraded") == 1
    assert diff.base_end_states.get("turn_limit") == 1
    assert diff.base_end_state_success["turn_limit"] == (1.0, 1)
    assert diff.base_end_state_success["parse_error"] == (0.0, 1)
    assert diff.answered_base_rate == pytest.approx(1 / 6, rel=1e-2)
    assert diff.answered_final_rate == pytest.approx(1.0)
    assert diff.correct_given_answered_base == pytest.approx(1.0)
    assert diff.correct_given_answered_final == pytest.approx(1.0)
    assert diff.cluster_by == "episode_type"
    assert any("Unmatched tasks" in w for w in diff.warnings)
    assert any("max_tokens" in w for w in diff.warnings)
    assert any("Attempt counts differ" in w for w in diff.warnings)
    assert any("counted as 0.0" in w for w in diff.warnings)

    # Missing cluster field and low-cardinality cluster field must fail loudly
    with pytest.raises(ValueError, match="missing"):
        diff_evals(base, final, cluster_by="no_such_field", n_boot=100)
    with pytest.raises(ValueError, match="minimum 5"):
        diff_evals(base, final, cluster_by="split", n_boot=100)


def test_cluster_bootstrap_deterministic_and_single_cluster_raises():
    diffs = [1.0, 1.0, 0.0, 0.0]
    clusters = ["c1", "c1", "c2", "c2"]
    lo1, hi1 = bootstrap_ci_mean_diff(diffs, n_boot=2000, seed=42, cluster_ids=clusters)
    lo2, hi2 = bootstrap_ci_mean_diff(diffs, n_boot=2000, seed=42, cluster_ids=clusters)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 == pytest.approx(0.0)
    assert hi1 == pytest.approx(1.0)

    with pytest.raises(ValueError, match="at least 2 distinct clusters"):
        bootstrap_ci_mean_diff(diffs, n_boot=100, seed=42, cluster_ids=["only"] * 4)


def test_diff_rollout_summaries_warns_on_different_iterations(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_iter_warn", num_steps=2, with_tags=True)
    ep0 = load_eval_jsonl(run_dir / "iteration_000000" / "train_rollout_summaries.jsonl")
    ep1 = load_eval_jsonl(run_dir / "iteration_000001" / "train_rollout_summaries.jsonl")
    diff = diff_evals(ep0, ep1, n_boot=100, seed=42)
    assert any("different iterations" in w for w in diff.warnings)


def test_single_turn_zero_final_ob_len_uses_last_ob_len_never_sub_1x():
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
    assert traj.unique_tokens == 40
    assert traj.sum_ob_len / traj.unique_tokens == pytest.approx(1.0)


def test_cumulative_rollout_errors_and_zero_reward_on_limit(tmp_path: Path):
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


def test_analyze_run_multiturn_tool_diagnostics_and_cache_series(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_tir", num_steps=4, native_cost=True)
    mpath = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in mpath.read_text().splitlines() if line.strip()]
    for i, r in enumerate(rows):
        r["env/all/used_python"] = 0.55
        r["env/all/tool_calls_count"] = 1.10
        r["env/all/tool_errors"] = 0.15 if i < 2 else 0.10
        r["env/all/hit_max_turns"] = 0.08
    mpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    diag = analyze_run(load_run(run_dir))
    assert diag.tools is not None
    assert diag.tools.used_tool_rate == pytest.approx(0.55)
    assert diag.tools.mean_tool_calls == pytest.approx(1.10)
    assert diag.tools.mean_tool_errors == pytest.approx(0.125)
    assert diag.tools.hit_max_turns_rate == pytest.approx(0.08)
    assert len(diag.tools.per_step_cache_hit) == 4
    tool_f = next(f for f in diag.findings if f.check_id == "tool_use")
    assert tool_f.severity == "pass"

    # High tool errors should trigger WARN
    for r in rows:
        r["env/all/tool_errors"] = 0.65
    mpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    diag_warn = analyze_run(load_run(run_dir))
    tool_f_warn = next(f for f in diag_warn.findings if f.check_id == "tool_use")
    assert tool_f_warn.severity == "warn"
