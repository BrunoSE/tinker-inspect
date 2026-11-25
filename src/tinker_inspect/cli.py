"""Command-line entrypoint for tinker-inspect."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from tinker_inspect import __version__
from tinker_inspect.diagnostics import analyze_run, diff_evals
from tinker_inspect.html_report import (
    build_compare_html,
    build_eval_diff_html,
    build_run_report_html,
    write_html,
)
from tinker_inspect.loader import load_eval_jsonl, load_run
from tinker_inspect.pricing import fetch_tinker_pricing, parse_rates_spec
from tinker_inspect.schema import TokenCostRates
from tinker_inspect.series import (
    analyze_eval_series,
    build_series_html,
    render_series_terminal,
    series_to_json,
)
from tinker_inspect.terminal import (
    diagnostics_to_json,
    has_failures,
    render_compare_terminal,
    render_eval_diff_terminal,
    render_run_terminal,
)
from tinker_inspect.watch import watch_run


def _resolve_rates(
    model_name: str | None,
    *,
    rates_spec: str | None,
    fetch_pricing: bool,
) -> TokenCostRates | None:
    if rates_spec:
        return parse_rates_spec(rates_spec)
    if fetch_pricing and model_name:
        return fetch_tinker_pricing(model_name)
    return None


def _cmd_inspect(args: argparse.Namespace) -> int:
    run = load_run(args.log_dir)
    rates = _resolve_rates(
        run.config.get("model_name"),
        rates_spec=args.rates,
        fetch_pricing=args.fetch_pricing,
    )
    diag = analyze_run(run, rates=rates)

    eval_diff = None
    if args.eval_before and args.eval_after:
        base_eps = load_eval_jsonl(args.eval_before)
        final_eps = load_eval_jsonl(args.eval_after)
        eval_diff = diff_evals(
            base_eps,
            final_eps,
            base_label=args.base_label or Path(args.eval_before).stem,
            final_label=args.final_label or Path(args.eval_after).stem,
        )
    elif args.eval_before or args.eval_after:
        raise SystemExit("error: --eval-before and --eval-after must be provided together.")

    if args.json:
        print(diagnostics_to_json(diag, eval_diff=eval_diff))
    else:
        use_color = sys.stdout.isatty() and not args.no_color
        print(render_run_terminal(run, diag, eval_diff=eval_diff, color=use_color))

    if not args.no_html:
        out_path = Path(args.out) if args.out else Path(run.log_dir) / "tinker_inspect.html"
        html = build_run_report_html(
            run,
            diag,
            eval_diff=eval_diff,
            sanitize_paths=args.sanitize_paths,
        )
        written = write_html(html, out_path)
        if not args.json:
            print(f"\nHTML report: {written}")
        if args.open:
            webbrowser.open(written.as_uri())

    if args.strict and has_failures(diag, fail_on_warn=args.fail_on_warn):
        return 2
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    base_eps = load_eval_jsonl(args.base_jsonl)
    final_eps = load_eval_jsonl(args.final_jsonl)
    diff = diff_evals(
        base_eps,
        final_eps,
        base_label=args.base_label or Path(args.base_jsonl).stem,
        final_label=args.final_label or Path(args.final_jsonl).stem,
    )
    if args.json:
        import json
        from dataclasses import asdict

        print(
            json.dumps(
                {
                    "base_label": diff.base_label,
                    "final_label": diff.final_label,
                    "num_tasks": diff.num_tasks,
                    "primary_metric": diff.primary_metric,
                    "answer_chars_change_pct": diff.answer_chars_change_pct,
                    "turns_change_pct": diff.turns_change_pct,
                    "metric_deltas": [asdict(m) for m in diff.metric_deltas],
                },
                indent=2,
            )
        )
    else:
        use_color = sys.stdout.isatty() and not args.no_color
        print(render_eval_diff_terminal(diff, color=use_color))

    if not args.no_html:
        out_path = Path(args.out) if args.out else Path("tinker_eval_diff.html")
        written = write_html(
            build_eval_diff_html(diff, sanitize_paths=args.sanitize_paths),
            out_path,
        )
        if not args.json:
            print(f"\nHTML diff report: {written}")
        if args.open:
            webbrowser.open(written.as_uri())
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    run_a = load_run(args.run_a)
    run_b = load_run(args.run_b)
    rates_a = _resolve_rates(
        run_a.config.get("model_name"), rates_spec=args.rates, fetch_pricing=args.fetch_pricing
    )
    rates_b = _resolve_rates(
        run_b.config.get("model_name"), rates_spec=args.rates, fetch_pricing=args.fetch_pricing
    )
    diag_a = analyze_run(run_a, rates=rates_a)
    diag_b = analyze_run(run_b, rates=rates_b)

    use_color = sys.stdout.isatty() and not args.no_color
    print(render_compare_terminal(run_a, diag_a, run_b, diag_b, color=use_color))

    if not args.no_html:
        out_path = Path(args.out) if args.out else Path("tinker_compare.html")
        written = write_html(
            build_compare_html(run_a, diag_a, run_b, diag_b, sanitize_paths=args.sanitize_paths),
            out_path,
        )
        print(f"\nHTML comparison report: {written}")
        if args.open:
            webbrowser.open(written.as_uri())
    return 0


def _parse_series_inputs(
    raw_paths: Sequence[str],
    labels_arg: str | None,
) -> list[tuple[str, Path]]:
    explicit_labels = [s.strip() for s in labels_arg.split(",")] if labels_arg else []
    if explicit_labels and len(explicit_labels) != len(raw_paths):
        raise ValueError(
            f"--labels count ({len(explicit_labels)}) must match eval files ({len(raw_paths)})."
        )
    parsed: list[tuple[str, Path]] = []
    for idx, spec in enumerate(raw_paths):
        if not explicit_labels and "=" in spec and not Path(spec).exists():
            lbl, path_s = spec.split("=", 1)
            parsed.append((lbl.strip() or Path(path_s).stem, Path(path_s)))
        else:
            p = Path(spec)
            lbl = explicit_labels[idx] if explicit_labels else p.stem
            parsed.append((lbl, p))
    return parsed


def _cmd_series(args: argparse.Namespace) -> int:
    labeled_paths = _parse_series_inputs(args.eval_jsonls, args.labels)
    runs = [(label, load_eval_jsonl(path)) for label, path in labeled_paths]
    summary = analyze_eval_series(
        runs,
        primary_metric=args.metric,
        title=args.title,
    )
    if args.json:
        print(series_to_json(summary))
    else:
        use_color = sys.stdout.isatty() and not args.no_color
        print(render_series_terminal(summary, color=use_color))

    if not args.no_html:
        out_path = Path(args.out) if args.out else Path("tinker_series.html")
        written = write_html(
            build_series_html(summary, sanitize_paths=args.sanitize_paths),
            out_path,
        )
        if not args.json:
            print(f"\nHTML series report: {written}")
        if args.open:
            webbrowser.open(written.as_uri())
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    rates = _resolve_rates(
        None,
        rates_spec=args.rates,
        fetch_pricing=False,
    )
    use_color = sys.stdout.isatty() and not args.no_color
    return watch_run(
        args.log_dir,
        interval_s=args.interval,
        fail_fast=args.fail_fast,
        fail_on_warn=args.fail_on_warn,
        rates=rates,
        color=use_color,
        max_polls=args.max_polls,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinker-inspect",
        description=(
            "Zero-dependency diagnostics CLI and single-file HTML viewer for Tinker RL runs."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    # inspect
    p_ins = sub.add_parser("inspect", help="Inspect a Tinker RL log_path directory (default).")
    p_ins.add_argument("log_dir", help="Path to a Tinker RL log_path directory.")
    p_ins.add_argument("--out", help="Output HTML path (default: <log_dir>/tinker_inspect.html).")
    p_ins.add_argument("--no-html", action="store_true", help="Skip writing the HTML report.")
    p_ins.add_argument("--open", action="store_true", help="Open the HTML report in your browser.")
    p_ins.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout.")
    p_ins.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    p_ins.add_argument("--strict", action="store_true", help="Exit 2 on FAIL health checks.")
    p_ins.add_argument(
        "--fail-on-warn", action="store_true", help="With --strict, also exit 2 on WARN."
    )
    p_ins.add_argument("--eval-before", help="Optional base eval JSONL to attach as paired diff.")
    p_ins.add_argument("--eval-after", help="Optional final eval JSONL to attach as paired diff.")
    p_ins.add_argument("--base-label", help="Display label for --eval-before.")
    p_ins.add_argument("--final-label", help="Display label for --eval-after.")
    p_ins.add_argument(
        "--rates", help="Custom USD/1M token rates: prefill,cached_prefill,sample,train."
    )
    p_ins.add_argument(
        "--fetch-pricing", action="store_true", help="Fetch live Tinker pricing from models.json."
    )
    p_ins.add_argument(
        "--sanitize-paths", action="store_true", help="Redact local home directory paths in HTML."
    )
    p_ins.set_defaults(func=_cmd_inspect)

    # diff
    p_diff = sub.add_parser(
        "diff", help="Compare two eval JSONL files with bootstrap CIs & transcript diffs."
    )
    p_diff.add_argument("base_jsonl", help="Base checkpoint eval JSONL.")
    p_diff.add_argument("final_jsonl", help="Final checkpoint eval JSONL.")
    p_diff.add_argument("--base-label", help="Display label for base.")
    p_diff.add_argument("--final-label", help="Display label for final.")
    p_diff.add_argument("--out", help="Output HTML path (default: ./tinker_eval_diff.html).")
    p_diff.add_argument("--no-html", action="store_true", help="Skip writing HTML.")
    p_diff.add_argument("--open", action="store_true", help="Open HTML in browser.")
    p_diff.add_argument("--json", action="store_true", help="Emit JSON summary to stdout.")
    p_diff.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    p_diff.add_argument("--sanitize-paths", action="store_true", help="Redact local paths in HTML.")
    p_diff.set_defaults(func=_cmd_diff)

    # compare
    p_cmp = sub.add_parser("compare", help="Overlay two Tinker RL log_path runs side-by-side.")
    p_cmp.add_argument("run_a", help="First Tinker RL log_path directory.")
    p_cmp.add_argument("run_b", help="Second Tinker RL log_path directory.")
    p_cmp.add_argument("--out", help="Output HTML path (default: ./tinker_compare.html).")
    p_cmp.add_argument("--no-html", action="store_true", help="Skip writing HTML.")
    p_cmp.add_argument("--open", action="store_true", help="Open HTML in browser.")
    p_cmp.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    p_cmp.add_argument(
        "--rates", help="Custom USD/1M token rates: prefill,cached_prefill,sample,train."
    )
    p_cmp.add_argument(
        "--fetch-pricing", action="store_true", help="Fetch live Tinker pricing from models.json."
    )
    p_cmp.add_argument("--sanitize-paths", action="store_true", help="Redact local paths in HTML.")
    p_cmp.set_defaults(func=_cmd_compare)

    # series
    p_ser = sub.add_parser(
        "series",
        help="Compare 2+ eval JSONLs as a progression curve, Pareto chart, and task heatmap.",
    )
    p_ser.add_argument(
        "eval_jsonls",
        nargs="+",
        help="Ordered eval JSONL paths (optionally prefixed as 'Label=path.jsonl').",
    )
    p_ser.add_argument(
        "--labels",
        help="Comma-separated display labels matching eval_jsonls.",
    )
    p_ser.add_argument("--metric", help="Primary metric override (default: auto-detected).")
    p_ser.add_argument("--title", help="Report title.")
    p_ser.add_argument("--out", help="Output HTML path (default: ./tinker_series.html).")
    p_ser.add_argument("--no-html", action="store_true", help="Skip writing HTML.")
    p_ser.add_argument("--open", action="store_true", help="Open HTML in browser.")
    p_ser.add_argument("--json", action="store_true", help="Emit JSON summary to stdout.")
    p_ser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    p_ser.add_argument("--sanitize-paths", action="store_true", help="Redact local paths in HTML.")
    p_ser.set_defaults(func=_cmd_series)

    # watch
    p_w = sub.add_parser(
        "watch",
        help="Live-monitor an active Tinker RL log_path directory and alert on health issues.",
    )
    p_w.add_argument("log_dir", help="Path to an active Tinker RL log_path directory.")
    p_w.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Poll interval in seconds (default: 2.0).",
    )
    p_w.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit immediately with code 2 on FAIL health checks.",
    )
    p_w.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="With --fail-fast, also exit 2 on WARN health checks.",
    )
    p_w.add_argument(
        "--rates", help="Custom USD/1M token rates: prefill,cached_prefill,sample,train."
    )
    p_w.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    p_w.add_argument(
        "--max-polls",
        type=int,
        default=None,
        help="Optional maximum poll iterations before exiting cleanly.",
    )
    p_w.set_defaults(func=_cmd_watch)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    known_subcommands = {"inspect", "diff", "compare", "series", "watch"}
    # Allow `tinker-inspect /path/to/log_dir` shorthand without typing `inspect`.
    if raw_argv and raw_argv[0] not in known_subcommands and not raw_argv[0].startswith("-"):
        raw_argv = ["inspect", *raw_argv]

    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
