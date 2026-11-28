# AGENTS.md — tinker-inspect

Zero-dependency Python 3.11+ CLI & single-file HTML diagnostics viewer for `tinker-cookbook` RL runs and paired evals.

## Hard Invariants

- **Zero runtime dependencies:** `src/tinker_inspect/` uses Python 3.11+ stdlib only (no third-party imports in runtime code).
- **Self-contained HTML:** Zero CDN links or external scripts/fonts. Inline JS in `report_template.html` and `series.py` must pass `node --check`.
- **No personal paths:** Never hardcode `/Users/...` paths in `src/`, `tests/`, `README.md`, or `docs/` (use env vars for optional external fixture smoke tests, and `--sanitize-paths` when regenerating `docs/*.html`).
- **Safe overwrite:** Default HTML targets refuse to overwrite without `-f/--force`, checked **before** analysis or stdout/JSON emission. Explicit `--out` overwrites.
- **Strict stats:** Ungraded rows count as `0.0` (never silently dropped); cluster bootstrap errors on unknown fields or `< 5` clusters; deterministic seeded bootstrap (`seed=42`, `n_boot=10000`).

## Module Map

- `src/tinker_inspect/schema.py` — Dataclasses for runs, steps, timing spans (`group_idx`, `attributes`), groups, and eval episodes (`end_state`).
- `src/tinker_inspect/loader.py` — Parsers for Tinker run logs, eval JSONLs, `.jsonl.zst` (`compression.zstd` / `zstd -dc` fallback), and Prime hosted traces.
- `src/tinker_inspect/stats.py` — Deterministic paired & clustered bootstrap CIs and cluster-ID extraction.
- `src/tinker_inspect/diagnostics.py` — Health checks, pre-filter vs post-filter reconciliation, token/cache accounting, and paired `diff_evals`.
- `src/tinker_inspect/series.py` — Multi-arm `series` analysis, terminal table, and self-contained HTML builder.
- `src/tinker_inspect/watch.py` — Live run tailer with deduplicated findings and `--fail-fast`.
- `src/tinker_inspect/html_report.py` & `assets/report_template.html` — Payload serialization, path sanitization, and interactive light/dark/mobile HTML viewer.
- `src/tinker_inspect/terminal.py` & `cli.py` — Terminal formatters and `inspect`/`diff`/`compare`/`series`/`watch` CLI.
- `src/tinker_inspect/pricing.py` — Optional `--rates` / `--fetch-pricing` cost lookup.

## Verification Before Commit

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/pytest -v
```

When changing `report_template.html` or `series.py`, also verify inline JS syntax (`node --check`) and check 390px mobile + dark mode contrast. Keep `pyproject.toml` and `src/tinker_inspect/__init__.py` versions in sync on release.
