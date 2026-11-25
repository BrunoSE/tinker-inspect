"""Synthetic Tinker RL run and eval JSONL builders for tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def make_synthetic_run(
    root: Path,
    *,
    num_steps: int = 4,
    remove_constant: bool = True,
    native_cost: bool = False,
    kl_v1_by_step: list[float] | None = None,
    with_tags: bool = False,
    all_constant_step: int | None = None,
    length_drift: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": "Qwen/Qwen3.5-4B",
        "renderer_name": "qwen3_5_disable_thinking",
        "learning_rate": 1e-5,
        "lora_rank": 32,
        "max_tokens": 1024,
        "log_path": "/Users/tester/runs/my_run",
        "remove_constant_reward_groups": remove_constant,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (root / "code.diff").write_text("diff --git a/foo.py b/foo.py\n+x = 1\n", encoding="utf-8")
    (root / "checkpoints.jsonl").write_text(
        json.dumps({"name": "final", "batch": num_steps, "sampler_path": "tinker://abc/final"})
        + "\n",
        encoding="utf-8",
    )

    metrics_lines: list[str] = []
    timing_lines: list[str] = []

    for step in range(num_steps):
        it_dir = root / f"iteration_{step:06d}"
        it_dir.mkdir(exist_ok=True)
        rollout_rows: list[dict[str, Any]] = []

        # 4 groups per step: by default 1 mixed, 2 all-good, 1 all-bad (75% constant)
        if all_constant_step == step:
            group_specs = [
                ("all_good", [1.0, 1.0]),
                ("all_bad", [0.0, 0.0]),
            ]
        else:
            group_specs = [
                ("mixed", [0.0, 1.0]),
                ("all_good", [1.0, 1.0]),
                ("all_good", [1.0, 1.0]),
                ("all_bad", [0.0, 0.0]),
            ]

        for gidx, (_, rewards) in enumerate(group_specs):
            tag = [f"tag_{gidx % 2}"] if with_tags else []
            for tidx, rew in enumerate(rewards):
                ac_base = 60 + (step * 20 if length_drift else 0)
                turns = [
                    {
                        "step_idx": 0,
                        "ob_len": 1000,
                        "ac_len": ac_base,
                        "reward": 0.0,
                        "episode_done": False,
                    },
                    {
                        "step_idx": 1,
                        "ob_len": 1150,
                        "ac_len": ac_base,
                        "reward": 0.0,
                        "episode_done": False,
                    },
                    {
                        "step_idx": 2,
                        "ob_len": 1300,
                        "ac_len": ac_base,
                        "reward": 0.0,
                        "episode_done": True,
                    },
                ]
                rollout_rows.append(
                    {
                        "schema_version": 1,
                        "split": "train",
                        "iteration": step,
                        "group_idx": gidx,
                        "traj_idx": tidx,
                        "tags": tag,
                        "sampling_client_step": step,
                        "total_reward": rew,
                        "final_reward": rew,
                        "trajectory_metrics": {"outcome": rew, "outcome_facts": rew},
                        "steps": turns,
                        "final_ob_len": 1300 + ac_base,
                    }
                )

        (it_dir / "train_rollout_summaries.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rollout_rows) + "\n",
            encoding="utf-8",
        )

        kl1 = kl_v1_by_step[step] if kl_v1_by_step else 0.0
        ac_turn = 60.0 + (step * 20.0 if length_drift else 0.0)
        outcome_val = 0.70 + (0.05 * step if length_drift else 0.01 * step)
        facts_val = 0.90 + (0.002 * step if length_drift else 0.01 * step)
        row: dict[str, Any] = {
            "step": step,
            "optim/lr": 1e-5,
            "env/all/reward/total": 0.5 + 0.05 * step,
            "env/all/outcome": outcome_val,
            "env/all/outcome_facts": facts_val,
            "env/all/ac_tokens_per_turn": ac_turn,
            "env/all/ob_tokens_per_turn": 1150.0,
            "env/all/turns_per_episode": 3.0,
            "env/all/total_episodes": 2 if remove_constant else 8,
            "env/all/total_turns": 6 if remove_constant else 24,
            "env/all/total_ac_tokens": int(ac_turn * (6 if remove_constant else 24)),
            "env/all/total_ob_tokens": 6900 if remove_constant else 27600,
            "optim/kl_sample_train_v1": kl1,
            "optim/kl_sample_train_v2": kl1,
            "optim/entropy": 0.15 - 0.01 * step,
            "time/sampling": 40.0 + step * 5.0,
            "time/train_step": 5.0,
            "time/save_checkpoint": 1.5,
            "time/total": 46.5 + step * 5.0,
        }
        if with_tags:
            row["env/tag_0/reward/total"] = 0.6 + 0.04 * step
            row["env/tag_1/reward/total"] = 0.4 + 0.06 * step
        if native_cost:
            row.update(
                {
                    "tokens/prefill_cached": 4000,
                    "tokens/prefill_uncached": 2900,
                    "tokens/sampled": 360,
                    "tokens/trained": 2720,
                    "tokens/discarded/prefill_cached": 12000,
                    "tokens/discarded/prefill_uncached": 8700,
                    "tokens/discarded/sampled": 1080,
                    "cost/step_usd": 0.12,
                    "cost/discarded_usd": 0.28,
                    "cost/cumulative_usd": 0.40 * (step + 1),
                }
            )
        metrics_lines.append(json.dumps(row))

        # 4 concurrent rollout spans with a clear straggler
        spans = [
            {
                "name": "do_group_rollout_and_filter_constant_reward",
                "duration": 10.0,
                "wall_start": 0.0,
                "wall_end": 10.0,
            },
            {
                "name": "do_group_rollout_and_filter_constant_reward",
                "duration": 14.0,
                "wall_start": 0.0,
                "wall_end": 14.0,
            },
            {
                "name": "do_group_rollout_and_filter_constant_reward",
                "duration": 16.0,
                "wall_start": 0.0,
                "wall_end": 16.0,
            },
            {
                "name": "do_group_rollout_and_filter_constant_reward",
                "duration": 40.0 + step * 5.0,
                "wall_start": 0.0,
                "wall_end": 40.0 + step * 5.0,
            },
            {
                "name": "sampling",
                "duration": 40.0 + step * 5.0,
                "wall_start": 0.0,
                "wall_end": 40.0 + step * 5.0,
            },
            {
                "name": "prepare_minibatch",
                "duration": 0.1,
                "wall_start": 40.0 + step * 5.0,
                "wall_end": 40.1 + step * 5.0,
            },
            {
                "name": "train_step",
                "duration": 5.0,
                "wall_start": 40.1 + step * 5.0,
                "wall_end": 45.1 + step * 5.0,
            },
            {
                "name": "save_checkpoint",
                "duration": 1.5,
                "wall_start": 45.1 + step * 5.0,
                "wall_end": 46.6 + step * 5.0,
            },
        ]
        timing_lines.append(json.dumps({"step": step, "spans": spans}))

    (root / "metrics.jsonl").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    (root / "timing_spans.jsonl").write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    return root


def make_synthetic_eval_pair(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    base_p = root / "base.jsonl"
    final_p = root / "final.jsonl"
    base_rows: list[str] = []
    final_rows: list[str] = []

    specs = [
        # (task_id, prompt, base_lit, final_lit, base_facts, final_facts, base_ans, final_ans)
        (
            0,
            "Check status of /Users/secret/repo",
            0.0,
            1.0,
            1.0,
            1.0,
            "Clean.",
            (
                "On branch main\nnothing to commit, working tree clean. "
                "</script><img onerror=alert(1)>"
            ),
        ),
        (
            1,
            "List branches",
            1.0,
            1.0,
            1.0,
            1.0,
            "main, dev",
            "Branches: main, dev (current: main)",
        ),
        (2, "Create branch feat", 1.0, 0.0, 1.0, 0.0, "Created feat.", "Failed to create."),
        (3, "Show latest commit", 0.0, 1.0, 0.0, 1.0, "Unknown.", "Commit abc1234: initial commit"),
    ]
    for tid, prompt, bl, fl, bf, ff, ba, fa in specs:
        for _ in range(2):
            base_rows.append(
                json.dumps(
                    {
                        "split": "heldout",
                        "ok": True,
                        "errors": [],
                        "traces": [
                            {
                                "task_id": tid,
                                "prompt": prompt,
                                "rewards": {"shaped": {"score": bl, "weight": 1.0}},
                                "metrics": {
                                    "outcome": bl,
                                    "outcome_facts": bf,
                                    "success": bl,
                                    "success_facts": bf,
                                },
                                "sampled_turns": 2,
                                "num_input_tokens": 1200,
                                "num_output_tokens": 40,
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {"function": {"name": "git_status", "arguments": "{}"}}
                                        ],
                                    },
                                    {
                                        "role": "tool",
                                        "name": "git_status",
                                        "content": "On branch main",
                                    },
                                    {"role": "assistant", "content": ba},
                                ],
                            }
                        ],
                    }
                )
            )
            final_rows.append(
                json.dumps(
                    {
                        "split": "heldout",
                        "ok": True,
                        "errors": [],
                        "traces": [
                            {
                                "task_id": tid,
                                "prompt": prompt,
                                "rewards": {"shaped": {"score": fl, "weight": 1.0}},
                                "metrics": {
                                    "outcome": fl,
                                    "outcome_facts": ff,
                                    "success": fl,
                                    "success_facts": ff,
                                },
                                "sampled_turns": 2,
                                "num_input_tokens": 1250,
                                "num_output_tokens": 80,
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {"function": {"name": "git_status", "arguments": "{}"}}
                                        ],
                                    },
                                    {
                                        "role": "tool",
                                        "name": "git_status",
                                        "content": "On branch main",
                                    },
                                    {"role": "assistant", "content": fa},
                                ],
                            }
                        ],
                    }
                )
            )
    base_p.write_text("\n".join(base_rows) + "\n", encoding="utf-8")
    final_p.write_text("\n".join(final_rows) + "\n", encoding="utf-8")
    return base_p, final_p
