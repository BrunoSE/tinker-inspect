# tinker-inspect

[![CI](https://github.com/BrunoSE/tinker-inspect/actions/workflows/ci.yml/badge.svg)](https://github.com/BrunoSE/tinker-inspect/actions/workflows/ci.yml)

Zero-dependency CLI and single-file HTML diagnostics viewer for **Tinker RL** (`tinker-cookbook`) runs, live monitoring (`watch`), paired eval diffs (`diff`), run comparisons (`compare`), and multi-checkpoint progressions (`series`).

Point it at any Tinker `log_path` directory (`metrics.jsonl`, `config.json`, `timing_spans.jsonl`, `checkpoints.jsonl`, `iteration_*/train_rollout_summaries.jsonl`) for an instant terminal health check and self-contained, zero-CDN `.html` report.

## Live Demos

Browse all interactive reports at **[brunose.github.io/tinker-inspect](https://brunose.github.io/tinker-inspect/)**:

| Demo | What it shows |
|---|---|
| [Zero-Shot AIME 2024+2025 (`series`)](https://brunose.github.io/tinker-inspect/series_aime2425_ladder.html) | 5-arm `$0.00` ladder (`max_tokens=4096`): Base CoT `59.2%` → Base TIR `69.2%` → Step-32 CoT RL `71.7%` → **Step-16 TIR RL `80.8%` (`+21.7 pp`, `-42%` tokens)** → Step-32 TIR RL `74.2%`. |
| [Multi-Turn TIR RL + 3 Eval Tabs (`inspect`)](https://brunose.github.io/tinker-inspect/demo_inkling2_polaris_tir_rl_ood18.html) | One report with 3 paired eval tabs (**AIME 24+25 `+11.7 pp`**, **Polaris 1/8 OOD `+15.6 pp`**, **Held-Out `+10.9 pp`**), pre-filter reward (`0.527 → 0.703`), `1.86×` prefill, and `50%` sampling idle wait. |
| [OOD Olympiad `1/8` Progression (`series`)](https://brunose.github.io/tinker-inspect/series_polaris18_ladder.html) | 48-task paired progression from `25.0%` → **`46.9%` (`+21.9 pp`, `-38%` tokens)** with 95% CI curve, Pareto scatter, and flip heatmap. |
| [Run #1 vs Run #2 (`compare`)](https://brunose.github.io/tinker-inspect/compare_polaris_v1_vs_v2.html) | Acting on truncation alerts cut `max_tokens` cutoffs `61.3% → 19.8%` and turned a flat run into `+10.9 pp`. |
| [Qwen3.5-4B Git Tool RL (`inspect`)](https://brunose.github.io/tinker-inspect/demo_full25.html) | `3.45×` multi-turn prefill amplification (`~76%` cacheable), `71%` constant-reward drop rate, and timing Gantt. |

## What It Catches

- **Hidden constant-reward drop rates:** Reconciles pre-filter `train_rollout_summaries.jsonl` against post-filter `metrics.jsonl` to surface true reward progression and `mixed` / `all_good` / `all_bad` group drop rates.
- **Multi-turn prefill & cost:** Reads native `tokens/*` and `cost/*` from `tinker-cookbook`, falling back to exact turn-by-turn reconstruction (`sum(ob_len)` vs `final_ob_len`, cache-hit split). Pass `--rates` or `--fetch-pricing` for offline/live cost estimates.
- **KL0 drift, truncation & stragglers:** Checks `kl_sample_train_v1`/`_v2`, flags `max_tokens`/`max_turns`/timeouts, and maps `timing_spans.jsonl` (with `group_idx` links) into an interactive per-step Gantt.
- **Paired evals & multi-arm series:** Supports JSONL and `.jsonl.zst` (including Prime hosted traces), deterministic paired/clustered 95% bootstrap CIs, `Ended How` decomposition, and side-by-side transcripts.

## Quick Start

Python 3.11+ stdlib only (zero runtime dependencies):

```bash
# Run directly with uvx
uvx --from git+https://github.com/BrunoSE/tinker-inspect tinker-inspect /path/to/log_dir

# Attach multiple before/after eval pairs as interactive tabs (.jsonl or .jsonl.zst)
tinker-inspect /path/to/log_dir \
  --eval-before base_heldout.jsonl --eval-after final_heldout.jsonl \
  --base-label "Base Held-Out" --final-label "Final Held-Out" \
  --eval-before base_ood.jsonl --eval-after final_ood.jsonl \
  --base-label "Base OOD" --final-label "Final OOD" \
  --sanitize-paths --out report.html --open

# Default HTML outputs refuse silent overwrite; pass --force (or --out / --no-html)
tinker-inspect /path/to/log_dir --force

# JSON output or CI health gate (exits 2 on FAIL)
tinker-inspect /path/to/log_dir --json --no-html
tinker-inspect /path/to/log_dir --strict --no-html

# Standalone paired eval diff (with cluster bootstrap + regression gate)
tinker-inspect diff base.jsonl.zst final.jsonl.zst --cluster-by family --fail-if-regression success

# Overlay two runs side-by-side
tinker-inspect compare run_a/ run_b/ --out compare.html

# Multi-checkpoint / multi-method progression report
tinker-inspect series "Base=base.jsonl" "Step 16=s16.jsonl" "TIR RL=tir.jsonl" --out series.html

# Live-monitor an active run (exits 2 immediately on KL0 drift or health failure)
tinker-inspect watch /path/to/active/log_dir --interval 5 --fail-fast
```

## Development

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -v
```

## License

MIT
