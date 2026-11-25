"""Tests for self-contained HTML report generation, escaping, and CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fixtures import make_synthetic_eval_pair, make_synthetic_run

from tinker_inspect.cli import main
from tinker_inspect.diagnostics import analyze_run, diff_evals
from tinker_inspect.html_report import (
    build_compare_html,
    build_eval_diff_html,
    build_run_report_html,
)
from tinker_inspect.loader import load_eval_jsonl, load_run


def _extract_payload(html: str) -> dict:
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert m is not None
    return json.loads(m.group(1))


def test_html_report_is_self_contained_escaped_and_sanitized(tmp_path: Path):
    run_dir = make_synthetic_run(tmp_path / "run_html", num_steps=3, with_tags=True)
    base_p, final_p = make_synthetic_eval_pair(tmp_path / "evals")
    run = load_run(run_dir)
    diag = analyze_run(run)
    diff = diff_evals(load_eval_jsonl(base_p), load_eval_jsonl(final_p))

    html = build_run_report_html(run, diag, eval_diff=diff, sanitize_paths=True)
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
    assert payload["eval_diff"]["num_tasks"] == 4


def test_diff_and_compare_html_modes(tmp_path: Path):
    base_p, final_p = make_synthetic_eval_pair(tmp_path / "evals")
    diff = diff_evals(load_eval_jsonl(base_p), load_eval_jsonl(final_p))
    diff_html = build_eval_diff_html(diff)
    assert _extract_payload(diff_html)["mode"] == "diff"

    run_a = load_run(make_synthetic_run(tmp_path / "run_a", num_steps=3))
    run_b = load_run(make_synthetic_run(tmp_path / "run_b", num_steps=4, remove_constant=False))
    cmp_html = build_compare_html(run_a, analyze_run(run_a), run_b, analyze_run(run_b))
    payload = _extract_payload(cmp_html)
    assert payload["mode"] == "compare"
    assert payload["run_a"]["name"] == "run_a"
    assert payload["run_b"]["name"] == "run_b"


def test_cli_inspect_diff_compare_and_strict(tmp_path: Path, capsys):
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

    # Diff subcommand
    base_p, final_p = make_synthetic_eval_pair(tmp_path / "evals")
    diff_out = tmp_path / "diff.html"
    rc_diff = main(["diff", str(base_p), str(final_p), "--out", str(diff_out), "--no-color"])
    assert rc_diff == 0
    assert diff_out.is_file()

    # Compare subcommand
    cmp_out = tmp_path / "cmp.html"
    rc_cmp = main(["compare", str(run_ok), str(run_bad), "--out", str(cmp_out), "--no-color"])
    assert rc_cmp == 0
    assert cmp_out.is_file()
