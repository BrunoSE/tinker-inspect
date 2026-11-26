"""Tests for self-contained HTML report generation, escaping, and CLI."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from fixtures import make_synthetic_eval_pair, make_synthetic_run

from tinker_inspect.cli import _build_parser, _rewrite_shorthand_argv, main
from tinker_inspect.diagnostics import analyze_run, diff_evals
from tinker_inspect.html_report import (
    build_compare_html,
    build_eval_diff_html,
    build_run_report_html,
)
from tinker_inspect.loader import load_eval_jsonl, load_run
from tinker_inspect.schema import LogSummary
from tinker_inspect.terminal import render_compare_terminal, render_run_terminal


def _extract_payload(html: str) -> dict:
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert m is not None
    return json.loads(m.group(1))


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


def test_html_report_is_self_contained_escaped_and_sanitized(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_html", num_steps=3, with_tags=True)
    base_p, final_p = make_synthetic_eval_pair(tmp_path / "evals")
    run = load_run(run_dir)
    # Attach log_summary and logtree_groups to verify v0.3 serialization
    run.log_summary = LogSummary(
        error_lines=2,
        warn_lines=3,
        retry_or_rate_limit_lines=1,
        samples=["ERROR in /Users/secret/train.py: timeout"],
    )
    run.logtree_groups[0] = [
        {
            "group_idx": 0,
            "title": "Group 0 prompt in /Users/secret/repo",
            "trajectories": [{"traj_idx": 0, "summary": "r=1.0", "excerpt": "ok"}],
        }
    ]
    diag = analyze_run(run)
    diff = diff_evals(load_eval_jsonl(base_p), load_eval_jsonl(final_p))

    html = build_run_report_html(
        run, diag, eval_diff=diff, extra_eval_diffs=[diff], sanitize_paths=True
    )
    # Zero external script/stylesheet links
    assert "<script src=" not in html
    assert '<link rel="stylesheet"' not in html
    # Raw </script> inside model output must not break out of JSON script block
    assert "</script><img onerror=alert(1)>" not in html
    # Local home paths redacted
    assert "/Users/tester" not in html
    assert "/Users/secret" not in html

    payload = _extract_payload(html)
    assert payload["mode"] == "run"
    assert payload["run"]["diagnostics"]["num_steps"] == 3
    assert payload["run"]["log_summary"]["error_lines"] == 2
    assert "0" in payload["run"]["logtree_groups"]
    assert payload["eval_diff"]["num_tasks"] == 4
    assert len(payload["extra_eval_diffs"]) == 1
    assert "base_end_states" in payload["eval_diff"]
    assert "warnings" in payload["eval_diff"]
    first_grp = payload["run"]["groups_by_step"]["0"][0]
    assert "trunc_episodes" in first_grp
    assert "end_state" in first_grp["trajectories"][0]
    _assert_inline_js_valid(html)


def test_diff_and_compare_html_modes_and_js_syntax(tmp_path: Path):
    base_p, final_p = make_synthetic_eval_pair(tmp_path / "evals")
    diff = diff_evals(load_eval_jsonl(base_p), load_eval_jsonl(final_p))
    diff_html = build_eval_diff_html(diff)
    assert _extract_payload(diff_html)["mode"] == "diff"
    _assert_inline_js_valid(diff_html)

    run_a = load_run(make_synthetic_run(tmp_path / "run_alpha_long_experiment_name", num_steps=3))
    run_b = load_run(
        make_synthetic_run(
            tmp_path / "run_beta_long_experiment_name", num_steps=4, remove_constant=False
        )
    )
    diag_a, diag_b = analyze_run(run_a), analyze_run(run_b)
    cmp_html = build_compare_html(run_a, diag_a, run_b, diag_b)
    payload = _extract_payload(cmp_html)
    assert payload["mode"] == "compare"
    assert payload["run_a"]["name"] == "run_alpha_long_experiment_name"
    assert payload["run_b"]["name"] == "run_beta_long_experiment_name"
    _assert_inline_js_valid(cmp_html)

    # Terminal compare must NOT truncate run names to 18 chars
    term_cmp = render_compare_terminal(run_a, diag_a, run_b, diag_b, color=False)
    assert "run_alpha_long_experiment_name" in term_cmp
    assert "run_beta_long_experiment_name" in term_cmp


def test_cli_leading_flag_shorthand_and_sanitize_help(tmp_path: Path, capsys):
    assert _rewrite_shorthand_argv(["--help"]) == ["--help"]
    assert _rewrite_shorthand_argv(["-h"]) == ["-h"]
    assert _rewrite_shorthand_argv(["--version"]) == ["--version"]
    assert _rewrite_shorthand_argv(["diff", "a.jsonl", "b.jsonl"]) == ["diff", "a.jsonl", "b.jsonl"]
    assert _rewrite_shorthand_argv(["--no-html", "/tmp/run"]) == [
        "inspect",
        "--no-html",
        "/tmp/run",
    ]
    assert _rewrite_shorthand_argv(["--out", "x.html", "/tmp/run"]) == [
        "inspect",
        "--out",
        "x.html",
        "/tmp/run",
    ]

    run_ok = make_synthetic_run(tmp_path / "ok_run", num_steps=3)
    rc = main(["--no-html", "--no-color", str(run_ok)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tinker-inspect :: ok_run" in out

    # Verify --sanitize-paths help text mentions tinker:// URIs and W&B names
    parser = _build_parser()
    subparsers_action = next(
        a for a in parser._actions if getattr(a, "choices", None) and "inspect" in a.choices
    )
    ins_help = subparsers_action.choices["inspect"].format_help()
    assert "tinker://" in ins_help
    assert "W&B" in ins_help


def test_cli_inspect_diff_compare_series_and_regression_gate(tmp_path: Path, capsys):
    run_ok = make_synthetic_run(tmp_path / "ok_run", num_steps=3)
    out_html = tmp_path / "ok.html"
    rc = main([str(run_ok), "--out", str(out_html), "--json"])
    assert rc == 0
    assert out_html.is_file()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["diagnostics"]["num_steps"] == 3

    # Broken KL0 fails under --strict
    run_bad = make_synthetic_run(tmp_path / "bad_run", num_steps=2, kl_v1_by_step=[0.0, 0.05])
    rc_bad = main(["inspect", str(run_bad), "--no-html", "--strict", "--no-color"])
    assert rc_bad == 2
    assert "KL0 broken" in capsys.readouterr().out

    # --help must work cleanly on top-level and every subcommand (no unescaped % crashes)
    for subcmd in ([], ["inspect"], ["diff"], ["compare"], ["series"], ["watch"]):
        with pytest.raises(SystemExit) as exc_h:
            main([*subcmd, "--help"])
        assert exc_h.value.code == 0
        capsys.readouterr()

    # Diff subcommand with --json, --seed, --n-boot, --metric
    base_p, final_p = make_synthetic_eval_pair(tmp_path / "evals")
    diff_out = tmp_path / "diff.html"
    rc_diff = main(
        [
            "diff",
            str(base_p),
            str(final_p),
            "--metric",
            "outcome",
            "--seed",
            "7",
            "--n-boot",
            "500",
            "--json",
            "--out",
            str(diff_out),
        ]
    )
    assert rc_diff == 0
    assert diff_out.is_file()
    diff_json = json.loads(capsys.readouterr().out)
    assert diff_json["primary_metric"] == "outcome"
    assert diff_json["seed"] == 7
    assert diff_json["n_boot"] == 500
    assert "base_failed" in diff_json
    assert "final_failed" in diff_json
    assert "base_unmatched_tasks" in diff_json
    assert "final_unmatched_tasks" in diff_json
    assert "base_end_states" in diff_json
    assert len(diff_json["tasks"]) == 4

    # Single-cluster --cluster-by split must fail cleanly with exit code 1 (no silent 0-width CI)
    rc_bad_cl = main(["diff", str(base_p), str(final_p), "--cluster-by", "split", "--no-html"])
    assert rc_bad_cl == 1
    assert "minimum 5" in capsys.readouterr().err

    # --fail-if-regression exits 2 when final regresses significantly (CI upper bound < 0)
    reg_base = tmp_path / "reg_base.jsonl"
    reg_final = tmp_path / "reg_final.jsonl"
    reg_base.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": i,
                    "family": f"fam_{i}",
                    "prompt": f"Problem {i}",
                    "scores": {"outcome": 1.0, "success": 1.0},
                    "messages": [{"role": "assistant", "content": "ok"}],
                    "graded_by_env": True,
                }
            )
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    reg_final.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": i,
                    "family": f"fam_{i}",
                    "prompt": f"Problem {i}",
                    "scores": {"outcome": 0.0, "success": 0.0},
                    "messages": [{"role": "assistant", "content": "wrong"}],
                    "graded_by_env": True,
                }
            )
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    rc_reg = main(
        [
            "diff",
            str(reg_base),
            str(reg_final),
            "--no-html",
            "--no-color",
            "--n-boot",
            "500",
            "--cluster-by",
            "family",
            "--fail-if-regression",
            "outcome",
        ]
    )
    reg_out = capsys.readouterr().out
    assert rc_reg == 2
    assert "cluster_by=family (6 clusters)" in reg_out

    # --fail-if-regression exits 0 when final improves over base
    rc_noreg = main(
        [
            "diff",
            str(reg_final),
            str(reg_base),
            "--no-html",
            "--no-color",
            "--n-boot",
            "500",
            "--fail-if-regression",
            "outcome",
        ]
    )
    capsys.readouterr()
    assert rc_noreg == 0

    # Compare subcommand with --json
    cmp_out = tmp_path / "cmp.html"
    rc_cmp = main(["compare", str(run_ok), str(run_bad), "--out", str(cmp_out), "--json"])
    assert rc_cmp == 0
    assert cmp_out.is_file()
    cmp_json = json.loads(capsys.readouterr().out)
    assert cmp_json["run_a"]["name"] == "ok_run"
    assert cmp_json["run_b"]["name"] == "bad_run"
    assert "diagnostics" in cmp_json["run_a"]

    # Series subcommand with --seed, --n-boot, --cluster-by (6 distinct clusters)
    ser_out = tmp_path / "series.html"
    rc_ser = main(
        [
            "series",
            f"Base={reg_final}",
            f"Final={reg_base}",
            "--seed",
            "11",
            "--n-boot",
            "500",
            "--cluster-by",
            "family",
            "--out",
            str(ser_out),
            "--sanitize-paths",
            "--no-color",
        ]
    )
    assert rc_ser == 0
    assert ser_out.is_file()
    ser_out_txt = capsys.readouterr().out
    assert "cluster_by=family (6 clusters)" in ser_out_txt

    # Watch subcommand (bounded polls + fail-fast)
    rc_w_ok = main(["watch", str(run_ok), "--interval", "0.01", "--max-polls", "1", "--no-color"])
    capsys.readouterr()
    assert rc_w_ok == 0
    rc_w_bad = main(
        [
            "watch",
            str(run_bad),
            "--interval",
            "0.01",
            "--max-polls",
            "1",
            "--fail-fast",
            "--no-color",
        ]
    )
    capsys.readouterr()
    assert rc_w_bad == 2


def test_render_run_terminal_shows_log_summary_and_cache_label(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "term_run", num_steps=4)
    run = load_run(run_dir)
    run.log_summary = LogSummary(error_lines=1, warn_lines=2, retry_or_rate_limit_lines=4)
    diag = analyze_run(run)
    text = render_run_terminal(run, diag, color=False)
    assert "logs.log Signals" in text
    assert "retries/rate_limits=4" in text
    assert diag.tokens.cache_metric_label in text
    if diag.reward_early_mean is not None:
        assert "Training Progress" in text
    else:
        pytest.fail("Expected Training Progress in terminal output")
