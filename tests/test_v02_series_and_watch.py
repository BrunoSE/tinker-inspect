"""Tests for tinker-inspect v0.2 series progression analyzer and live run watcher."""

from __future__ import annotations

import io
import json
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
) -> EvalEpisode:
    metrics: dict[str, float] = {
        "success": score,
        "outcome": score,
        "format_ok": format_ok,
    }
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
        ok=True,
        reward=score,
        metrics=metrics,
        num_turns=num_turns,
        input_tokens=200,
        output_tokens=output_tokens,
        final_answer_chars=len("\\boxed{1}"),
        messages=msgs,
    )


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

    # Cross-check exact paired CI against diagnostics.diff_evals(seed=42, n_boot=2000)
    direct_diff = diff_evals(r0, r2, n_boot=2000, seed=42)
    direct_succ = next(m for m in direct_diff.metric_deltas if m.metric == "success")
    assert p2.delta_vs_base == pytest.approx(direct_succ.delta)
    assert p2.ci_low == pytest.approx(direct_succ.ci_low)
    assert p2.ci_high == pytest.approx(direct_succ.ci_high)

    assert sum(1 for t in summary.tasks if t.flipped_solved) == 2
    assert sum(1 for t in summary.tasks if t.flipped_regressed) == 1


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

    # Embedded JSON payload must parse cleanly
    m = re.search(
        r'<script id="tinker-series-data" type="application/json">(.*?)</script>', html, re.S
    )
    assert m is not None
    payload = json.loads(m.group(1))
    assert payload["num_tasks"] == 2
    assert "/Users/" not in json.dumps(payload)

    # Validate inline JS syntax via node --check if node is installed
    if shutil.which("node"):
        scripts = re.findall(
            r'<script(?![^>]*type="application/json")[^>]*>(.*?)</script>', html, re.S
        )
        assert scripts
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tf:
            tf.write("\n;\n".join(scripts))
            js_path = Path(tf.name)
        try:
            proc = subprocess.run(["node", "--check", str(js_path)], capture_output=True, text=True)
            assert proc.returncode == 0, f"node --check failed: {proc.stderr}"
        finally:
            js_path.unlink(missing_ok=True)


def test_series_on_real_ood_polaris18_evals():
    evals_dir = Path("/Users/bruno/Documents/inkling-experiments/evals")
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
