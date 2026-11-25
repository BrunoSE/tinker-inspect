# tinker-inspect

Zero-dependency CLI and single-file HTML diagnostics viewer for **Tinker RL** (`tinker-cookbook`) training runs and paired evaluation diffs (`verifiers v1`, legacy `vf-eval`, `git_tool_rl` / native JSONL, or rollout summaries).

Point it at any Tinker `log_path` directory (`metrics.jsonl`, `config.json`, `timing_spans.jsonl`, `checkpoints.jsonl`, `iteration_*/train_rollout_summaries.jsonl`) and get an instant terminal health check plus a self-contained, zero-CDN `.html` report you can open locally, attach to a PR, or host on GitHub Pages.

## Live Demos

Browse the interactive showcase at **[brunose.github.io/tinker-inspect](https://brunose.github.io/tinker-inspect/)** (source files in [`docs/`](docs/index.html)):

| Demo | Local / Repo | What it reveals |
|---|---|---|
| [Inkling-2-Mini Multi-Turn Python TIR RL → OOD `1/8`](https://brunose.github.io/tinker-inspect/demo_inkling2_polaris_tir_rl_ood18.html) | [`docs/demo_inkling2_polaris_tir_rl_ood18.html`](docs/demo_inkling2_polaris_tir_rl_ood18.html) | **+15.6 pp** multi-turn `run_python` TIR RL gain on OOD Olympiad `1/8` (`31.2% → 46.9%`, 95% CI `[+6.2, +26.0]`; `+21.9 pp` vs text-only base), `-34%` tokens, `-36%` tool errors, `55%` prefix-cacheable, exact KL0, `$0.00`. |
| [Inkling-2-Mini Step-32 RL → OOD Olympiad `1/8`](https://brunose.github.io/tinker-inspect/demo_inkling2_polaris_rl_v3_step32_ood18.html) | [`docs/demo_inkling2_polaris_rl_v3_step32_ood18.html`](docs/demo_inkling2_polaris_rl_v3_step32_ood18.html) | **+12.5 pp** zero-shot OOD Olympiad `1/8` transfer (`25.0% → 37.5%`, 95% CI `[+2.1, +22.9]`, `13W/3R/32T`, `-9.8%` tokens) at exact KL0 (`0.000000`) and `$0.00`. |
| [Inkling-2-Mini Competition Math RL (Run #2)](https://brunose.github.io/tinker-inspect/demo_inkling2_polaris_rl_v2.html) | [`docs/demo_inkling2_polaris_rl_v2.html`](docs/demo_inkling2_polaris_rl_v2.html) | **+10.9 pp** held-out Polaris gain (`56.2% → 67.2%`, 95% CI `[+2.3, +20.3]`) on `Inkling-2-Mini:peft:65536:free` with exact KL0 (`0.000000`) at `$0.00`. |
| [Diagnostic-Driven Iteration (Run #1 vs Run #2)](https://brunose.github.io/tinker-inspect/compare_polaris_v1_vs_v2.html) | [`docs/compare_polaris_v1_vs_v2.html`](docs/compare_polaris_v1_vs_v2.html) | Side-by-side comparison showing how acting on `tinker-inspect`'s truncation alert cut `max_tokens` cutoffs `61.3% → 19.8%` and all-bad groups `109 → 25`. |
| [Qwen3.5-4B Multi-Turn Git Tool-Use RL](https://brunose.github.io/tinker-inspect/demo_full25.html) | [`docs/demo_full25.html`](docs/demo_full25.html) | Reconstructs `3.45×` multi-turn prefill amplification (`~76%` prefix-cacheable), `71%` hidden constant-group drop rate (`269` all-good), and `+8.2 pp` held-out gain. |
| [Qwen3.5-4B vs Inkling-2-Mini (`l5_new_tools`)](https://brunose.github.io/tinker-inspect/qwen_vs_inkling_l5_new_tools.html) | [`docs/qwen_vs_inkling_l5_new_tools.html`](docs/qwen_vs_inkling_l5_new_tools.html) | Cross-format paired transcript diff (32 tasks) exposing Inkling's default `effort=0.9` mid-thinking truncation (`+45.1%` turns, `+87.5%` answer chars). |
| [Thinking-Effort Pareto Chart](https://brunose.github.io/tinker-inspect/effort_pareto.svg) | [`docs/effort_pareto.svg`](docs/effort_pareto.svg) | Tuning `tml_v0` thinking effort from `0.9` → `0.2–0.5` lifted `l5_new_tools` success `87.5% → 94.8%–95.8%` at `2.7×–3.9×` fewer action tokens. |

## Example terminal output

Real output from inspecting a 16-step free-tier `Inkling-2-Mini:peft:65536:free` RL run with paired before/after held-out evaluations attached:

```text
tinker-inspect :: inkling2_polaris_rl_v2  (thinkingmachines/Inkling-2-Mini:peft:65536:free)
  steps=16  |  wall=24.1m  |  lr=0.00015  |  lora_rank=32  |  checkpoints=3

Health Checks
  [PASS] Sampler-Trainer KL0 Check: Exact KL0 (0.000000 across all steps)
  [WARN] Constant-Reward Group Drop Rate: 63% of groups dropped (47/128 kept; 56 all-good, 25 all-bad)
  [PASS] Straggler & Wall-Clock Bottleneck: Sampling is 83% of wall time; median straggler ratio 1.54x (peak 2.02x at step 15)
  [PASS] Response Length Stability: Action tokens/turn stable (-11.0% early vs late)
  [WARN] Max-Token Truncation: 19.8% of turns hit max_tokens=4096 (203/1024)

Tokens & Cost (native)
  prefill=139.0k  cacheable=43%
  sampled=2.37M  trained=1.14M  discarded_sampling=54%
  estimated_usd=$0.00  (discarded=$0.00)  [metrics.jsonl (native)]

Training Progress (step 0 -> final)
  reward/total      0.875 -> 0.125 (-0.750)
  outcome           0.875 -> 0.125 (-0.750)
  success           0.875 -> 0.125 (-0.750)
  ac_tokens/turn    3058.2 -> 4092.9

Eval Diff :: Base Inkling-2-Mini (free) -> RL Inkling-2-Mini (free) (64 paired tasks)
  metric                base   final    delta              95% CI        W/R/T
  success              0.562   0.672   +0.109    [+0.023, +0.203]      15/3/46
  outcome              0.562   0.672   +0.109    [+0.023, +0.203]      15/3/46
  correct              0.562   0.672   +0.109    [+0.023, +0.203]      15/3/46
  reward               0.562   0.672   +0.109    [+0.023, +0.203]      15/3/46
  format_ok            0.789   0.828   +0.039    [-0.039, +0.117]       8/3/53
  shaped_reward        0.562   0.672   +0.109    [+0.023, +0.203]      15/3/46
  answer_chars=+0.2%  |  turns=+0.0%
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

# Attach before/after eval JSONLs and redact local paths for public sharing
tinker-inspect /path/to/log_dir \
  --eval-before base_heldout.jsonl \
  --eval-after final_heldout.jsonl \
  --sanitize-paths \
  --out demo.html --open

# Machine-readable JSON or CI health gate (exits 2 on FAIL)
tinker-inspect /path/to/log_dir --json --no-html
tinker-inspect /path/to/log_dir --strict --no-html

# Standalone paired eval diff (verifiers v1, git_tool_rl, legacy vf-eval, or rollout summaries)
tinker-inspect diff base_heldout.jsonl final_heldout.jsonl --sanitize-paths --out diff.html

# Overlay two Tinker RL training runs side-by-side
tinker-inspect compare run_a/ run_b/ --sanitize-paths --out compare.html
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
