"""Tests for tinker-inspect v0.2/v0.3 series progression analyzer and live run watcher."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from fixtures import make_synthetic_run

from tinker_inspect.diagnostics import diff_evals
from tinker_inspect.loader import load_eval_jsonl
from tinker_inspect.schema import EvalEpisode, EvalMessage
from tinker_inspect.series import (
    analyze_eval_series,
    build_series_html,
    render_series_terminal,
    series_to_json,
)
from tinker_inspect.watch import format_watch_line, watch_run


def _mk_ep(
    task_id: str | int,
    prompt: str,
    score: float,
    *,
    task_key: str | None = None,
    format_ok: float = 1.0,
    output_tokens: int | None = 500,
    num_turns: int = 1,
    used_python: float | None = None,
    tool_errors: float | None = None,
    with_tool_call: bool = False,
    ok: bool = True,
    omit_success: bool = False,
    reward_override: float | None = None,
    episode_type: str | None = None,
    max_tokens: int | None = None,
    effort: float | None = None,
    stop_reason: str | None = None,
    gold_answer: str | None = None,
) -> EvalEpisode:
    metrics: dict[str, float] = {
        "outcome": score,
        "format_ok": format_ok,
    }
    if not omit_success:
        metrics["success"] = score
    if used_python is not None:
        metrics["used_python"] = used_python
    if tool_errors is not None:
        metrics["tool_errors"] = tool_errors
    msgs = [EvalMessage(role="user", content=prompt)]
    if with_tool_call:
        msgs.append(
            EvalMessage(
                role="assistant",
                content="",
                tool_calls=[{"name": "run_python", "arguments": '{"code": "print(1)"}'}],
            )
        )
        msgs.append(EvalMessage(role="tool", name="run_python", content="1"))
    msgs.append(EvalMessage(role="assistant", content="\\boxed{1}"))
    return EvalEpisode(
        task_key=task_key if task_key is not None else f"task:{task_id}",
        task_id=task_id,
        prompt=prompt,
        split="test",
        ok=ok,
        reward=reward_override if reward_override is not None else score,
        metrics=metrics,
        num_turns=num_turns,
        input_tokens=200,
        output_tokens=output_tokens,
        final_answer_chars=len("\\boxed{1}"),
        messages=msgs,
        stop_reason=stop_reason,
        gold_answer=gold_answer,
        episode_type=episode_type,
        effort=effort,
        max_tokens=max_tokens,
    )


def _assert_inline_js_valid(html: str) -> None:
    if not shutil.which("node"):
        return
    scripts = re.findall(r'<script(?![^>]*type="application/json")[^>]*>(.*?)</script>', html, re.S)
    assert scripts
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tf:
        tf.write("\n;\n".join(scripts))
        js_path = Path(tf.name)
    try:
        proc = subprocess.run(["node", "--check", str(js_path)], capture_output=True, text=True)
        assert proc.returncode == 0, f"node --check failed: {proc.stderr}"
    finally:
        js_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Series analyzer tests
# ---------------------------------------------------------------------------


def test_analyze_eval_series_three_runs_and_ci_consistency():
    # 4 tasks across 3 runs:
    # t0: 0 -> 0 -> 1 (flipped_solved)
    # t1: 0 -> 1 -> 1 (flipped_solved)
    # t2: 1 -> 1 -> 0 (flipped_regressed)
    # t3: 1 -> 1 -> 1 (always solved)
    prompts = [
        "Solve problem A in /Users/secret/proj",
        "Solve problem B",
        "Solve problem C",
        "Solve problem D",
    ]
    r0 = [_mk_ep(i, prompts[i], s, output_tokens=1000) for i, s in enumerate([0.0, 0.0, 1.0, 1.0])]
    r1 = [_mk_ep(i, prompts[i], s, output_tokens=900) for i, s in enumerate([0.0, 1.0, 1.0, 1.0])]
    r2 = [
        _mk_ep(
            i,
            prompts[i],
            s,
            output_tokens=700,
            num_turns=2,
            used_python=1.0 if i < 2 else 0.0,
            tool_errors=0.5 if i == 0 else 0.0,
            with_tool_call=(i < 2),
        )
        for i, s in enumerate([1.0, 1.0, 0.0, 1.0])
    ]

    summary = analyze_eval_series([("Base", r0), ("Step 16", r1), ("TIR Step 32", r2)])
    assert summary.num_tasks == 4
    assert summary.primary_metric == "success"
    assert len(summary.points) == 3

    p0, p1, p2 = summary.points
    assert p0.primary_mean == pytest.approx(0.5)
    assert p0.delta_vs_base == 0.0
    assert p1.primary_mean == pytest.approx(0.75)
    assert p1.delta_vs_base == pytest.approx(0.25)
    assert (p1.wins_vs_base, p1.regressions_vs_base, p1.ties_vs_base) == (1, 0, 3)

    assert p2.primary_mean == pytest.approx(0.75)
    assert p2.delta_vs_base == pytest.approx(0.25)
    assert (p2.wins_vs_base, p2.regressions_vs_base, p2.ties_vs_base) == (2, 1, 1)
    assert p2.used_tool_rate == pytest.approx(0.5)
    assert p2.mean_tool_errors == pytest.approx(0.125)
    assert p2.mean_output_tokens == pytest.approx(700.0)
    assert p2.end_states.get("completed") == 4

    # Cross-check exact paired CI against diagnostics.diff_evals using shared stats default
    direct_diff = diff_evals(r0, r2)
    direct_succ = next(m for m in direct_diff.metric_deltas if m.metric == "success")
    assert p2.delta_vs_base == pytest.approx(direct_succ.delta)
    assert p2.ci_low == pytest.approx(direct_succ.ci_low)
    assert p2.ci_high == pytest.approx(direct_succ.ci_high)

    assert sum(1 for t in summary.tasks if t.flipped_solved) == 2
    assert sum(1 for t in summary.tasks if t.flipped_regressed) == 1
    assert len(summary.tasks[0].arm_summaries) == 3


def test_series_no_reward_fallback_warnings_and_clustering():
    # Arm 0 has an unmatched extra task and max_tokens=4096 across 5 distinct families.
    # Arm 1 has max_tokens=3072, an episode missing "success" with shaped reward=0.75
    # (must NOT fall back to 0.75 — must score 0.0 for "success"!), and 2 attempts on task 0.
    r0 = [
        _mk_ep(0, "P0", 0.0, episode_type="famA", max_tokens=4096, gold_answer="42"),
        _mk_ep(1, "P1", 1.0, episode_type="famB", max_tokens=4096),
        _mk_ep(2, "P2", 0.0, episode_type="famC", max_tokens=4096),
        _mk_ep(3, "P3", 1.0, episode_type="famD", max_tokens=4096),
        _mk_ep(4, "P4", 0.0, episode_type="famE", max_tokens=4096),
        _mk_ep(99, "Extra base only", 1.0, episode_type="famE", max_tokens=4096),
    ]
    r1 = [
        _mk_ep(
            0,
            "P0",
            0.0,
            omit_success=True,
            reward_override=0.75,
            ok=False,
            episode_type="famA",
            max_tokens=3072,
            stop_reason="length",
        ),
        _mk_ep(0, "P0", 1.0, episode_type="famA", max_tokens=3072),
        _mk_ep(1, "P1", 1.0, episode_type="famB", max_tokens=3072),
        _mk_ep(2, "P2", 1.0, episode_type="famC", max_tokens=3072),
        _mk_ep(3, "P3", 1.0, episode_type="famD", max_tokens=3072),
        _mk_ep(4, "P4", 1.0, episode_type="famE", max_tokens=3072),
    ]
    summary = analyze_eval_series(
        [("CoT 4096", r0), ("TIR 3072", r1)],
        primary_metric="success",
        cluster_by="episode_type",
        n_boot=500,
        seed=42,
    )
    assert summary.num_tasks == 5
    assert summary.num_clusters == 5
    assert summary.unmatched_counts == [1, 0]
    assert summary.cluster_by == "episode_type"
    # Task 0 in arm 1 has two attempts: (0.0 + 1.0) / 2 = 0.5 (NOT (0.75 + 1.0) / 2 = 0.875!)
    t0 = next(t for t in summary.tasks if t.task_id == 0)
    assert t0.scores[1] == pytest.approx(0.5)
    assert t0.gold_answer == "42"
    assert any("Unmatched tasks" in w for w in summary.warnings)
    assert any("Attempt counts differ" in w for w in summary.warnings)
    assert any("Config mismatch" in w and "max_tokens" in w for w in summary.warnings)

    # Unknown or single-valued cluster field must raise (never silently produce 0-width CI)
    with pytest.raises(ValueError, match="missing"):
        analyze_eval_series([("A", r0), ("B", r1)], cluster_by="no_such_field")
    with pytest.raises(ValueError, match="minimum 5"):
        analyze_eval_series([("A", r0), ("B", r1)], cluster_by="split")

    html = build_series_html(summary)
    assert "Comparison Warnings" in html or "warn-wrap" in html
    _assert_inline_js_valid(html)


def test_analyze_eval_series_prompt_fallback_and_errors():
    # Mixing integer task_ids (0, 1) with md5 hex task_ids ("aa", "bb") -> falls back to prompt!
    r0 = [
        _mk_ep(0, "Compute   2 + 2.", 0.0, task_key="task:0"),
        _mk_ep(1, "Compute 3 + 3.", 1.0, task_key="task:1"),
    ]
    r1 = [
        _mk_ep("aa", "Compute 2 + 2.", 1.0, task_key="task:aa"),
        _mk_ep("bb", "Compute\n3 + 3.", 1.0, task_key="task:bb"),
    ]
    summary = analyze_eval_series([("IntIDs", r0), ("HexIDs", r1)])
    assert summary.num_tasks == 2
    assert summary.points[1].delta_vs_base == pytest.approx(0.5)

    with pytest.raises(ValueError, match="at least 2"):
        analyze_eval_series([("OnlyOne", r0)])

    r_disjoint = [_mk_ep("zz", "Completely unrelated prompt", 1.0, task_key="task:zz")]
    with pytest.raises(ValueError, match="No common tasks"):
        analyze_eval_series([("A", r0), ("B", r_disjoint)])


def test_series_terminal_json_and_html_report():
    r0 = [
        _mk_ep(0, "Check /Users/alice/secret/file.py", 0.0, output_tokens=1200),
        _mk_ep(1, "Solve x + 1 = 2 </script>", 1.0, output_tokens=800),
    ]
    r1 = [
        _mk_ep(0, "Check /Users/alice/secret/file.py", 1.0, output_tokens=900, used_python=1.0),
        _mk_ep(1, "Solve x + 1 = 2 </script>", 1.0, output_tokens=700, used_python=0.0),
    ]
    summary = analyze_eval_series([("Base", r0), ("Final", r1)], title="Demo Series")

    term = render_series_terminal(summary, color=False)
    assert "Demo Series" in term
    assert "Base" in term and "Final" in term
    assert "+0.500" in term

    parsed_json = json.loads(series_to_json(summary))
    assert parsed_json["num_tasks"] == 2
    assert len(parsed_json["points"]) == 2

    html = build_series_html(summary, sanitize_paths=True)
    assert "/Users/alice" not in html
    assert "~/secret/file.py" in html
    assert "</script><" not in html
    assert "Connected Curve" in html
    assert "Grouped Bars / Dots" in html
    assert "95% CI of &Delta; vs baseline drawn around arm mean" in html

    # Embedded JSON payload must parse cleanly
    m = re.search(
        r'<script id="tinker-series-data" type="application/json">(.*?)</script>', html, re.S
    )
    assert m is not None
    payload = json.loads(m.group(1))
    assert payload["num_tasks"] == 2
    assert "/Users/" not in json.dumps(payload)
    _assert_inline_js_valid(html)


def test_series_on_real_ood_polaris18_evals():
    env_dir = os.environ.get("TINKER_INSPECT_EVALS_DIR")
    if not env_dir:
        pytest.skip("Set TINKER_INSPECT_EVALS_DIR to run real OOD Polaris 1/8 eval smoke test")
    evals_dir = Path(env_dir)
    specs = [
        ("Base CoT", evals_dir / "base_polaris18_4096.jsonl"),
        ("RL v2 Step16", evals_dir / "final_v2_step16_polaris18_4096.jsonl"),
        ("RL v3 Step32", evals_dir / "final_v3_step32_polaris18_4096.jsonl"),
        ("Base TIR", evals_dir / "base_tir_polaris18_3072.jsonl"),
        ("RL TIR", evals_dir / "final_tir_polaris18_3072.jsonl"),
    ]
    if not all(p.is_file() for _, p in specs):
        pytest.skip("Real OOD Polaris 1/8 eval JSONLs not present in local workspace")

    runs = [(lbl, load_eval_jsonl(p)) for lbl, p in specs]
    summary = analyze_eval_series(runs, title="Inkling-2-Mini OOD Polaris 1/8 Progression")
    assert summary.num_tasks == 48
    assert len(summary.points) == 5
    # TIR points should report positive tool usage rate while CoT points report 0.0
    assert summary.points[0].used_tool_rate == pytest.approx(0.0)
    assert summary.points[3].used_tool_rate is not None and summary.points[3].used_tool_rate > 0.0
    html = build_series_html(summary, sanitize_paths=True)
    assert "/Users/" not in html


# ---------------------------------------------------------------------------
# Live watch tests
# ---------------------------------------------------------------------------


def test_watch_run_waits_streams_steps_dedups_warnings_and_exits_on_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "live_run"
    # Build a full 2-step run in a staging dir, then reveal it incrementally across polls!
    staged = make_synthetic_run(tmp_path / "staged", num_steps=2, remove_constant=True)
    metric_lines = (staged / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    summary_row = json.dumps(
        {
            "step": 2,
            "tokens/cumulative/sampled": 5000,
            "tokens/discarded/sampled": 200,
            "cost/cumulative_usd": 0.0,
        }
    )

    poll_state = {"n": 0}

    def fake_sleep(_sec: float) -> None:
        poll_state["n"] += 1
        n = poll_state["n"]
        if n == 1:
            # Create run_dir with step 0 only
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(staged / "config.json", run_dir / "config.json")
            shutil.copytree(staged / "iteration_000000", run_dir / "iteration_000000")
            (run_dir / "metrics.jsonl").write_text(metric_lines[0] + "\n", encoding="utf-8")
        elif n == 2:
            # Idle poll with no new steps -> should not duplicate step 0 or its warning
            return
        elif n == 3:
            # Append intermediate step=None cumulative token row + step 1 (must NOT exit yet!)
            inter_row = json.dumps(
                {
                    "tokens/cumulative/sampled": 2500,
                    "tokens/discarded/sampled": 100,
                    "cost/cumulative_usd": 0.0,
                }
            )
            shutil.copytree(staged / "iteration_000001", run_dir / "iteration_000001")
            (run_dir / "metrics.jsonl").write_text(
                "\n".join([metric_lines[0], inter_row, metric_lines[1]]) + "\n", encoding="utf-8"
            )
        elif n == 4:
            # Append post-loop summary row (with explicit step=2 -> triggers clean exit 0)
            (run_dir / "metrics.jsonl").write_text(
                "\n".join([*metric_lines[:2], summary_row]) + "\n", encoding="utf-8"
            )

    monkeypatch.setattr("tinker_inspect.watch.time.sleep", fake_sleep)
    buf = io.StringIO()
    rc = watch_run(run_dir, interval_s=0.01, color=False, max_polls=8, stream=buf)
    output = buf.getvalue()

    assert rc == 0
    assert "waiting for metrics" in output
    assert output.count("step=0 |") == 1
    assert output.count("step=1 |") == 1
    assert "Run complete ::" in output
    # Step 0 warning should only appear once despite the idle poll (deduplicated by headline)
    assert output.count("1/4 kept") == 1


def test_watch_finished_run_no_double_findings_and_per_step_status(tmp_path: Path):
    # Finished run (checkpoint name='final') with clean step 0 (kl=0) and bad step 1 (kl=0.05)
    run_dir = make_synthetic_run(
        tmp_path / "finished_run",
        num_steps=2,
        remove_constant=False,
        kl_v1_by_step=[0.0, 0.05],
    )
    (run_dir / "checkpoints.jsonl").write_text(
        json.dumps({"name": "final", "batch": 1, "sampler_path": "tinker://final"}) + "\n",
        encoding="utf-8",
    )
    buf = io.StringIO()
    rc = watch_run(run_dir, interval_s=0.0, color=False, max_polls=3, stream=buf)
    out = buf.getvalue()
    assert rc == 0
    # Step 0 has clean per-step status [PASS], while step 1 has [FAIL]
    step0_line = next(ln for ln in out.splitlines() if ln.startswith("step=0 |"))
    step1_line = next(ln for ln in out.splitlines() if ln.startswith("step=1 |"))
    assert "status=[PASS]" in step0_line
    assert "status=[FAIL]" in step1_line
    # Findings must NOT be printed twice on a finished run!
    assert out.count("KL0 broken") == 1
    assert "Run complete ::" in out and "status=[FAIL]" in out


def test_watch_run_fail_fast_exits_2_on_kl_drift(tmp_path: Path):
    run_dir = make_synthetic_run(
        tmp_path / "kl_bad",
        num_steps=3,
        kl_v1_by_step=[0.0, 0.02, 0.03],
    )
    buf = io.StringIO()
    rc = watch_run(
        run_dir,
        interval_s=0.0,
        fail_fast=True,
        color=False,
        max_polls=3,
        stream=buf,
    )
    out = buf.getvalue()
    assert rc == 2
    assert "FAIL-FAST TRIGGERED" in out
    assert "KL0 broken" in out


def test_format_watch_line_contents(tmp_path: Path):
    from tinker_inspect.diagnostics import analyze_run
    from tinker_inspect.loader import load_run

    run_dir = make_synthetic_run(tmp_path / "fmt_run", num_steps=2, remove_constant=True)
    run = load_run(run_dir)
    diag = analyze_run(run)
    line = format_watch_line(run, diag, 0, color=False)
    assert "step=0" in line
    assert "rew=" in line
    assert "kl_v1=" in line
    assert "kept=1/4 (all_good=2, all_bad=1)" in line
    assert "trunc=0.0%" in line
    assert "ac_tok=" in line
    assert "wall=" in line
    assert "status=[" in line
