# tinker-inspect

[![CI](https://github.com/BrunoSE/tinker-inspect/actions/workflows/ci.yml/badge.svg)](https://github.com/BrunoSE/tinker-inspect/actions/workflows/ci.yml)

Zero-dependency CLI and single-file HTML diagnostics viewer for **Tinker RL** (`tinker-cookbook`) training runs, live run monitoring (`watch`), paired evaluation diffs (`diff`), and multi-checkpoint progression reports (`series`).

Point it at any Tinker `log_path` directory (`metrics.jsonl`, `config.json`, `timing_spans.jsonl`, `checkpoints.jsonl`, `iteration_*/train_rollout_summaries.jsonl`) and get an instant terminal health check plus a self-contained, zero-CDN `.html` report you can open locally, attach to a PR, or host on GitHub Pages.

## Live Demos

Browse the interactive showcase at **[brunose.github.io/tinker-inspect](https://brunose.github.io/tinker-inspect/)** (source files in [`docs/`](docs/index.html)):

| Demo | Local / Repo | What it reveals |
|---|---|---|
| [5-Arm Zero-Shot AIME 2024+2025 Progression (`series`)](https://brunose.github.io/tinker-inspect/series_aime2425_ladder.html) | [`docs/series_aime2425_ladder.html`](docs/series_aime2425_ladder.html) | Zero-shot transfer (`0` train overlap, `max_tokens=4096`): Base CoT (`59.2%`) → Base TIR (`69.2%`) → Step-32 CoT RL (`71.7%`) → **Step-16 TIR RL (`80.8%`, `+21.7 pp` `[+10.8, +33.3]`, `-42%` tokens, `85.0%` AIME 2024 / `76.7%` AIME 2025)** → Step-32 TIR RL (`74.2%`). |
| [5-Point OOD Olympiad `1/8` Progression (`series`)](https://brunose.github.io/tinker-inspect/series_polaris18_ladder.html) | [`docs/series_polaris18_ladder.html`](docs/series_polaris18_ladder.html) | Full 48-task paired progression across Base CoT (`25.0%`) → Base TIR (`31.2%`) → Step-16 CoT RL (`34.4%`) → Step-32 CoT RL (`37.5%`) → **16-Step Multi-Turn TIR RL (`46.9%`, `+21.9 pp` `[+11.5, +32.3]`, `-38%` tokens)** with accuracy/Pareto charts & per-task heatmap. |
| [Inkling-2-Mini Multi-Turn Python TIR RL → Multi-Eval Tabs (AIME + OOD `1/8` + Held-Out)](https://brunose.github.io/tinker-inspect/demo_inkling2_polaris_tir_rl_ood18.html) | [`docs/demo_inkling2_polaris_tir_rl_ood18.html`](docs/demo_inkling2_polaris_tir_rl_ood18.html) | Repeatable multi-eval tabs in one report: **AIME 24+25 (`69.2% → 80.8%`, `+11.7 pp`)**, **Polaris OOD `1/8` (`31.2% → 46.9%`, `+15.6 pp`)**, and **Held-Out `2..4/8` (`56.2% → 67.2%`, `+10.9 pp`)**, plus pre-filter reward (`0.527 → 0.703`), `1.86×` prefill (`55%` measured cache hits), exact KL0, `$0.00`. |
| [Inkling-2-Mini Step-32 RL → OOD Olympiad `1/8`](https://brunose.github.io/tinker-inspect/demo_inkling2_polaris_rl_v3_step32_ood18.html) | [`docs/demo_inkling2_polaris_rl_v3_step32_ood18.html`](docs/demo_inkling2_polaris_rl_v3_step32_ood18.html) | **+12.5 pp** zero-shot OOD Olympiad `1/8` transfer (`25.0% → 37.5%`, 95% CI `[+2.1, +22.9]`, `13W/3R/32T`, `-9.8%` tokens) at exact KL0 (`0.000000`) and `$0.00`. |
| [Inkling-2-Mini Competition Math RL (Run #2)](https://brunose.github.io/tinker-inspect/demo_inkling2_polaris_rl_v2.html) | [`docs/demo_inkling2_polaris_rl_v2.html`](docs/demo_inkling2_polaris_rl_v2.html) | **+10.9 pp** held-out Polaris gain (`56.2% → 67.2%`, 95% CI `[+2.3, +20.3]`) on `Inkling-2-Mini:peft:65536:free` with exact KL0 (`0.000000`) at `$0.00`. |
| [Diagnostic-Driven Iteration (Run #1 vs Run #2)](https://brunose.github.io/tinker-inspect/compare_polaris_v1_vs_v2.html) | [`docs/compare_polaris_v1_vs_v2.html`](docs/compare_polaris_v1_vs_v2.html) | Side-by-side comparison showing how acting on `tinker-inspect`'s truncation alert cut `max_tokens` cutoffs `61.3% → 19.8%` and all-bad groups `109 → 25`. |
| [Qwen3.5-4B Multi-Turn Git Tool-Use RL](https://brunose.github.io/tinker-inspect/demo_full25.html) | [`docs/demo_full25.html`](docs/demo_full25.html) | Reconstructs `3.45×` multi-turn prefill amplification (`~76%` prefix-cacheable), `71%` hidden constant-group drop rate (`269` all-good), and `+8.2 pp` held-out gain. |
| [Qwen3.5-4B vs Inkling-2-Mini (`l5_new_tools`)](https://brunose.github.io/tinker-inspect/qwen_vs_inkling_l5_new_tools.html) | [`docs/qwen_vs_inkling_l5_new_tools.html`](docs/qwen_vs_inkling_l5_new_tools.html) | Cross-format paired transcript diff (32 tasks) exposing Inkling's default `effort=0.9` mid-thinking truncation (`+45.1%` turns, `+87.5%` answer chars). |
| [Thinking-Effort Pareto Chart](https://brunose.github.io/tinker-inspect/effort_pareto.svg) | [`docs/effort_pareto.svg`](docs/effort_pareto.svg) | Tuning `tml_v0` thinking effort from `0.9` → `0.2–0.5` lifted `l5_new_tools` success `87.5% → 94.8%–95.8%` at `2.7×–3.9×` fewer action tokens. |

## Example terminal output

Real output from inspecting a 16-step free-tier multi-turn `run_python` TIR RL run (`Inkling-2-Mini:peft:65536:free`) with paired before/after OOD Olympiad `1/8` evaluations attached:

```text
tinker-inspect :: inkling2_polaris_tir_rl  (thinkingmachines/Inkling-2-Mini:peft:65536:free)
  steps=16  |  wall=30.2m  |  lr=0.00015  |  lora_rank=32  |  checkpoints=3

Health Checks
  [PASS] Sampler-Trainer KL0 Check: Exact KL0 (0.000000 across all steps) (max mismatch_frac=0.0000, max |logp_diff|=0.00e+00)
  [WARN] Constant-Reward Group Drop Rate: 52% of groups dropped (61/128 kept; 43 all-good, 24 all-bad)
  [INFO] Multi-Turn Prefill Amplification: 1.86x prefill-to-unique-token ratio (55% measured cache hits)
  [WARN] Straggler & Sampling Idle Time: 50% sampling idle wait (median straggler 2.01x, peak 3.77x at step 12; sampling 75% of wall) (policy_sample=14.18s/turn, env_step=0.13s/turn)
  [PASS] Response Length Stability: Action tokens/turn (pre-filter) stable (-8.9% early vs late)
  [WARN] Episode Token-Limit Truncation: 10.2% of episodes hit token limit (104/1024) (reward: 0.00 truncated vs 0.70 untruncated)
  [PASS] Multi-Turn Tool Execution: (prefilter) 48.1% used tools, 0.94 calls/ep, 0.26 errors/ep, 11.4% hit max_turns

Tokens & Cost (native)
  prefill=2.43M (1.86x unique 1.31M)  Measured cache hits=55%
  sampled=1.91M  trained=1.41M  discarded_sampling=37%  discarded_action_only=41%
  estimated_usd=$0.00  (discarded=$0.00)  [metrics.jsonl (native)]

Sampling Timing & Stragglers
  idle_wait=50%  |  median_straggler=2.01x  |  policy_sample=14.18s  |  env_step=0.13s

Multi-Turn Tool & Cache (prefilter)
  turns/ep=1.94  |  used_tool=48%  |  calls/ep=0.94  |  errors/ep=0.26  |  max_turns=11.4%  |  trunc_ep=10.2%  |  cache_hit=55%->60%
  trunc_reward=0.000 vs untrunc_reward=0.696

Training Progress (first 4 vs last 4 steps, pre-filter all sampled)
  reward (mean)     0.527 -> 0.703 (+0.176)
  reward (trained)  0.604 -> 0.607 (+0.003)  [post-filter kept, pooled]
  ac_tokens/turn    1061.8 -> 967.7  (pre-filter)

Eval Diff :: Base Inkling-2-Mini + run_python (free) -> 16-Step TIR RL Inkling-2-Mini + run_python (free) (48 paired tasks)
  metric                base   final    delta              95% CI        W/R/T
  success              0.312   0.469   +0.156    [+0.062, +0.260]      15/3/30
  format_ok            0.583   0.802   +0.219    [+0.104, +0.333]      16/1/31
  output_tokens      2698.021 1780.500 -917.521 [-1148.948, -695.542]    2/45/1
  tool_errors          0.344   0.219   -0.125    [-0.229, -0.021]      4/14/30
  (identical to success: outcome, correct, reward, shaped_reward)
  answer_chars=+1.1%  |  turns=-12.1%
  completed=58.3%->80.2% (+21.9pp)  |  success|completed=53.6%->58.4% (+4.9pp)
  ended_how (base)  : completed=56 (58%, succ=54%), token_limit=28 (29%, succ=0%), turn_limit=12 (12%, succ=0%)
  ended_how (final) : completed=77 (80%, succ=58%), token_limit=11 (11%, succ=0%), turn_limit=8 (8%, succ=0%)
  bootstrap: n=10000, seed=42
```

## Why this exists

Standard metric logs hide several of the most expensive failure modes in multi-turn RL:

1. **Hidden constant-reward drop rates:** In `tinker_cookbook.rl.train`, `metrics.jsonl` is computed *after* `remove_constant_reward_groups`, so `env/all/by_group/frac_mixed` reads `1.0` even when 70%+ of sampled groups were dropped. `tinker-inspect` reconciles pre-filter `train_rollout_summaries.jsonl` against `metrics.jsonl` to surface the true `mixed` / `all_good` / `all_bad` split and discarded compute.
2. **Multi-turn prefill amplification:** Counting only unique trajectory tokens undercounts multi-turn RL prefill by 3–4× because each turn re-prefills the growing history. `tinker-inspect` reads native `tokens/*` and `cost/*` from `tinker-cookbook` `main`, and falls back cleanly to exact turn-by-turn reconstruction (`sum(ob_len)` vs `final_ob_len`, plus prefix-cacheable vs uncached split) on older runs.
3. **Sampler-trainer KL0 drift & max-token truncation:** Checks `optim/kl_sample_train_v1` and `_v2` across every step, flags `max_tokens` turn truncation and `zero_reward_on_limit` clamping, and supports `--strict` exit codes for CI.
4. **Synchronous straggler bottlenecks:** Turns `timing_spans.jsonl` into an interactive per-step Gantt showing concurrent `do_group_rollout_and_filter_constant_reward` bars alongside `train_step` and `save_checkpoint`.
5. **Verbosity / reward-hacking drift & paired eval diffs:** Tracks generation length growth alongside paired literal vs facts metrics, and compares before/after evaluation JSONLs (with automatic prompt-fallback pairing across differing `task_id` formats) using deterministic 95% bootstrap CIs and expandable side-by-side transcripts.

## Quick start

Zero runtime dependencies (Python 3.11+ stdlib only):

```bash
# Run directly with uvx (no install needed)
uvx --from git+https://github.com/BrunoSE/tinker-inspect tinker-inspect /path/to/tinker/log_dir

# Or from a local checkout
uvx --from . tinker-inspect /path/to/tinker/log_dir

# Attach one or multiple before/after eval JSONL / .jsonl.zst pairs as tabs
tinker-inspect /path/to/log_dir \
  --eval-before base_heldout.jsonl --eval-after final_heldout.jsonl \
  --base-label "Base Held-Out" --final-label "Final Held-Out" \
  --eval-before base_ood.jsonl --eval-after final_ood.jsonl \
  --base-label "Base OOD" --final-label "Final OOD" \
  --sanitize-paths --out demo.html --open

# Default HTML outputs refuse silent overwrite; pass --force (or --out / --no-html)
tinker-inspect /path/to/log_dir --force

# Machine-readable JSON or CI health gate (exits 2 on FAIL)
tinker-inspect /path/to/log_dir --json --no-html
tinker-inspect /path/to/log_dir --strict --no-html

# Standalone paired eval diff (verifiers, git_tool_rl, Prime .jsonl.zst traces, vf-eval)
tinker-inspect diff base_heldout.jsonl.zst final_heldout.jsonl.zst \
  --cluster-by family --fail-if-regression success --sanitize-paths --out diff.html

# Overlay two Tinker RL training runs side-by-side (HTML or --json)
tinker-inspect compare run_a/ run_b/ --sanitize-paths --out compare.html

# Compare 3+ checkpoints or methods as a paired progression curve + Pareto + task heatmap
tinker-inspect series \
  "Base CoT=base.jsonl" "Step 16=step16.jsonl" "Step 32=step32.jsonl" "TIR RL=tir.jsonl" \
  --title "OOD Olympiad Progression" --sanitize-paths --out series.html

# Live-monitor an active Tinker RL run and exit 2 immediately on KL0 drift or health failure
tinker-inspect watch /path/to/active/log_dir --interval 5 --fail-fast
```

### Cost estimation

- Runs produced by `tinker-cookbook` `main` (`cost_estimation.py`) display native API-reported `tokens/*` and `cost/*` metrics automatically (including `$0.00` for `:free` variants).
- Older runs without `cost/*` stay strictly offline by default; pass `--rates prefill,cached_prefill,sample,train` (USD per 1M tokens) or opt in to `--fetch-pricing` to fetch live rates from Tinker's `models.json`.

## Development

```bash
uv run ruff check src tests
uv run pytest -v
```

## License

MIT
