"""Core diagnostic analyzers for Tinker RL runs and paired evaluation diffs."""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Any

from tinker_inspect.schema import (
    DiagnosticFinding,
    EvalDiffSummary,
    EvalEpisode,
    EvalMetricDelta,
    EvalTaskDiff,
    RunData,
    RunDiagnostics,
    TokenCostRates,
    TokenSummary,
    ToolDiagnostics,
)

_PRIMARY_METRIC_PRIORITY = ("success", "outcome", "correct", "reward")


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _compute_token_summary(run: RunData, rates: TokenCostRates | None) -> TokenSummary:
    all_token_steps = [*run.steps, *([run.summary_step] if run.summary_step is not None else [])]
    has_native = any(
        bool(s.native_tokens) or bool(s.native_discarded_tokens) for s in all_token_steps
    )
    is_multiturn = any((s.turns_per_episode or 0.0) > 1.05 for s in run.steps) or any(
        tr.num_turns > 1
        for groups in run.groups_by_step.values()
        for g in groups
        for tr in g.trajectories
    )
    if has_native:
        prefill_cached = sum(s.native_tokens.get("prefill_cached", 0) for s in run.steps)
        prefill_uncached = sum(s.native_tokens.get("prefill_uncached", 0) for s in run.steps)
        sampled = sum(s.native_tokens.get("sampled", 0) for s in run.steps)
        trained = sum(s.native_tokens.get("trained", 0) for s in run.steps)
        disc_cached = sum(
            s.native_discarded_tokens.get("prefill_cached", 0) for s in all_token_steps
        )
        disc_uncached = sum(
            s.native_discarded_tokens.get("prefill_uncached", 0) for s in all_token_steps
        )
        disc_sampled = sum(s.native_discarded_tokens.get("sampled", 0) for s in all_token_steps)

        kept_prefill = prefill_cached + prefill_uncached
        disc_prefill = disc_cached + disc_uncached
        prefill_total = kept_prefill + disc_prefill
        sampled_total = sampled + disc_sampled

        unique_tokens = sum(
            g.unique_final_ob_len for groups in run.groups_by_step.values() for g in groups
        )
        multiplier = (
            (prefill_total / unique_tokens) if (is_multiturn and unique_tokens > 0) else None
        )
        cache_frac = (prefill_cached + disc_cached) / prefill_total if prefill_total > 0 else None
        total_sampling_tok = prefill_total + sampled_total
        disc_frac = (
            (disc_prefill + disc_sampled) / total_sampling_tok if total_sampling_tok > 0 else None
        )

        native_step_costs = [
            s.native_cost_usd for s in all_token_steps if s.native_cost_usd is not None
        ]
        native_disc_costs = [
            s.native_discarded_cost_usd
            for s in all_token_steps
            if s.native_discarded_cost_usd is not None
        ]
        native_cum_costs = [
            s.native_cumulative_cost_usd
            for s in all_token_steps
            if s.native_cumulative_cost_usd is not None
        ]

        est_cost: float | None = None
        est_disc_cost: float | None = None
        pricing_src: str | None = None
        if native_cum_costs:
            est_cost = native_cum_costs[-1]
            est_disc_cost = sum(native_disc_costs) if native_disc_costs else None
            pricing_src = "metrics.jsonl (native)"
        elif native_step_costs:
            est_disc_cost = sum(native_disc_costs) if native_disc_costs else 0.0
            est_cost = sum(native_step_costs) + (est_disc_cost or 0.0)
            pricing_src = "metrics.jsonl (native)"
        elif rates is not None:
            est_disc_cost = rates.estimate_usd(
                prefill_uncached=disc_uncached,
                prefill_cached=disc_cached,
                sampled=disc_sampled,
                trained=0,
            )
            est_cost = rates.estimate_usd(
                prefill_uncached=prefill_uncached + disc_uncached,
                prefill_cached=prefill_cached + disc_cached,
                sampled=sampled_total,
                trained=trained,
            )
            pricing_src = rates.source

        return TokenSummary(
            source="native",
            prefill_total=prefill_total,
            prefill_cached=prefill_cached + disc_cached,
            prefill_uncached=prefill_uncached + disc_uncached,
            sampled_total=sampled_total,
            trained_total=trained,
            unique_trajectory_tokens=unique_tokens,
            prefill_multiplier=multiplier,
            cacheable_fraction=cache_frac,
            kept_prefill=kept_prefill,
            discarded_prefill=disc_prefill,
            kept_sampled=sampled,
            discarded_sampled=disc_sampled,
            discarded_token_fraction=disc_frac,
            estimated_cost_usd=est_cost,
            estimated_discarded_cost_usd=est_disc_cost,
            pricing_source=pricing_src,
        )

    if run.groups_by_step:
        kept_prefill = 0
        disc_prefill = 0
        kept_sampled = 0
        disc_sampled = 0
        prefill_cached = 0
        prefill_uncached = 0
        disc_cached = 0
        disc_uncached = 0
        unique_tokens = 0
        trained_total = 0

        for groups in run.groups_by_step.values():
            for g in groups:
                unique_tokens += g.unique_final_ob_len
                prefill_cached += g.estimated_prefill_cached
                prefill_uncached += g.estimated_prefill_uncached
                if g.kept_after_filter:
                    kept_prefill += g.sum_ob_len
                    kept_sampled += g.sum_ac_len
                    trained_total += g.unique_final_ob_len
                else:
                    disc_prefill += g.sum_ob_len
                    disc_sampled += g.sum_ac_len
                    disc_cached += g.estimated_prefill_cached
                    disc_uncached += g.estimated_prefill_uncached

        prefill_total = kept_prefill + disc_prefill
        sampled_total = kept_sampled + disc_sampled
        multiplier = (
            (prefill_total / unique_tokens) if (is_multiturn and unique_tokens > 0) else None
        )
        cache_frac = (prefill_cached / prefill_total) if prefill_total > 0 else None
        total_sampling_tok = prefill_total + sampled_total
        disc_frac = (
            (disc_prefill + disc_sampled) / total_sampling_tok if total_sampling_tok > 0 else None
        )

        est_cost = None
        est_disc_cost = None
        pricing_src = None
        if rates is not None:
            est_disc_cost = rates.estimate_usd(
                prefill_uncached=disc_uncached,
                prefill_cached=disc_cached,
                sampled=disc_sampled,
                trained=0,
            )
            est_cost = rates.estimate_usd(
                prefill_uncached=prefill_uncached,
                prefill_cached=prefill_cached,
                sampled=sampled_total,
                trained=trained_total,
            )
            pricing_src = f"{rates.source} (reconstructed)"

        return TokenSummary(
            source="reconstructed",
            prefill_total=prefill_total,
            prefill_cached=prefill_cached,
            prefill_uncached=prefill_uncached,
            sampled_total=sampled_total,
            trained_total=trained_total,
            unique_trajectory_tokens=unique_tokens,
            prefill_multiplier=multiplier,
            cacheable_fraction=cache_frac,
            kept_prefill=kept_prefill,
            discarded_prefill=disc_prefill,
            kept_sampled=kept_sampled,
            discarded_sampled=disc_sampled,
            discarded_token_fraction=disc_frac,
            estimated_cost_usd=est_cost,
            estimated_discarded_cost_usd=est_disc_cost,
            pricing_source=pricing_src,
        )

    # Metrics-only fallback (kept trajectories only).
    ob_total = sum(s.total_ob_tokens or 0 for s in run.steps)
    ac_total = sum(s.total_ac_tokens or 0 for s in run.steps)
    return TokenSummary(
        source="metrics_only",
        prefill_total=ob_total,
        prefill_uncached=ob_total,
        sampled_total=ac_total,
        kept_prefill=ob_total,
        kept_sampled=ac_total,
    )


def _detect_paired_metric_divergence(run: RunData) -> DiagnosticFinding | None:
    """Detect length inflation or divergence between literal and facts/secondary metrics."""
    if len(run.steps) < 4:
        return None
    q = max(1, len(run.steps) // 4)
    first_slice = run.steps[:q]
    last_slice = run.steps[-q:]

    ac_first = [s.ac_tokens_per_turn for s in first_slice if s.ac_tokens_per_turn is not None]
    ac_last = [s.ac_tokens_per_turn for s in last_slice if s.ac_tokens_per_turn is not None]
    len_growth_pct: float | None = None
    if ac_first and ac_last and _mean(ac_first) > 1e-9:
        len_growth_pct = ((_mean(ac_last) - _mean(ac_first)) / _mean(ac_first)) * 100.0

    # Check known paired metric families (e.g. outcome vs outcome_facts, success vs success_facts).
    all_keys = {k for s in run.steps for k in s.env_metrics}
    divergent_pair: tuple[str, str, float, float] | None = None
    for base_key in ("success", "outcome"):
        facts_key = f"{base_key}_facts"
        if base_key in all_keys and facts_key in all_keys:
            b_first = [s.env_metrics[base_key] for s in first_slice if base_key in s.env_metrics]
            b_last = [s.env_metrics[base_key] for s in last_slice if base_key in s.env_metrics]
            f_first = [s.env_metrics[facts_key] for s in first_slice if facts_key in s.env_metrics]
            f_last = [s.env_metrics[facts_key] for s in last_slice if facts_key in s.env_metrics]
            if b_first and b_last and f_first and f_last:
                b_delta = _mean(b_last) - _mean(b_first)
                f_delta = _mean(f_last) - _mean(f_first)
                if b_delta >= 0.04 and b_delta - f_delta >= 0.03:
                    divergent_pair = (base_key, facts_key, b_delta, f_delta)
                    break

    if len_growth_pct is not None and len_growth_pct >= 25.0 and divergent_pair is not None:
        bk, fk, bd, fd = divergent_pair
        return DiagnosticFinding(
            check_id="verbosity_drift",
            title="Verbosity / Reward-Hack Drift",
            severity="warn",
            headline=(
                f"Action tokens/turn grew {len_growth_pct:+.0f}% while {bk} ({bd:+.3f}) "
                f"outpaced {fk} ({fd:+.3f})"
            ),
            detail=(
                f"Comparing the first {q} vs last {q} steps: mean ac_tokens_per_turn rose from "
                f"{_mean(ac_first):.1f} to {_mean(ac_last):.1f} ({len_growth_pct:+.1f}%). "
                f"Paired metric `{bk}` gained {bd:+.3f} while `{fk}` moved {fd:+.3f}. "
                "Inspect eval diffs to verify the model is not gaming a string-matching grader "
                "via longer outputs."
            ),
            metrics={
                "len_growth_pct": round(len_growth_pct, 2),
                "proxy_metric": bk,
                "proxy_delta": round(bd, 4),
                "facts_metric": fk,
                "facts_delta": round(fd, 4),
            },
        )
    if len_growth_pct is not None and len_growth_pct >= 30.0:
        return DiagnosticFinding(
            check_id="verbosity_drift",
            title="Response Length Drift",
            severity="warn",
            headline=f"Action tokens/turn increased {len_growth_pct:+.0f}% over training",
            detail=(
                f"Mean ac_tokens_per_turn moved from {_mean(ac_first):.1f} (first {q} steps) "
                f"to {_mean(ac_last):.1f} (last {q} steps). Check whether longer generations "
                "reflect genuine reasoning or verbosity/paste gaming."
            ),
            metrics={"len_growth_pct": round(len_growth_pct, 2)},
        )
    if len_growth_pct is not None:
        return DiagnosticFinding(
            check_id="verbosity_drift",
            title="Response Length Stability",
            severity="pass",
            headline=f"Action tokens/turn stable ({len_growth_pct:+.1f}% early vs late)",
            detail=(
                f"Mean ac_tokens_per_turn moved from {_mean(ac_first):.1f} (first {q} steps) "
                f"to {_mean(ac_last):.1f} (last {q} steps)."
            ),
            metrics={"len_growth_pct": round(len_growth_pct, 2)},
        )
    return None


def analyze_run(run: RunData, rates: TokenCostRates | None = None) -> RunDiagnostics:
    """Compute all health checks, token/cost stats, and timing breakdowns for a run."""
    findings: list[DiagnosticFinding] = []
    remove_constant = bool(run.config.get("remove_constant_reward_groups", False))

    # 1. KL0 sampler-trainer alignment check
    kl_v1_vals = [(s.step, abs(s.kl_v1)) for s in run.steps if s.kl_v1 is not None]
    kl_v2_vals = [(s.step, abs(s.kl_v2)) for s in run.steps if s.kl_v2 is not None]
    max_kl_v1 = max((v for _, v in kl_v1_vals), default=None)
    max_kl_v2 = max((v for _, v in kl_v2_vals), default=None)
    nonzero_steps = [step for step, v in kl_v1_vals + kl_v2_vals if v > 1e-12]
    first_nonzero_kl = min(nonzero_steps) if nonzero_steps else None

    if max_kl_v1 is None and max_kl_v2 is None:
        findings.append(
            DiagnosticFinding(
                check_id="kl0",
                title="Sampler-Trainer KL0 Check",
                severity="info",
                headline="No optim/kl_sample_train metrics logged",
                detail="metrics.jsonl does not include optim/kl_sample_train_v1 or _v2.",
            )
        )
    else:
        worst_kl = max(v for v in (max_kl_v1, max_kl_v2) if v is not None)
        if worst_kl <= 1e-12:
            findings.append(
                DiagnosticFinding(
                    check_id="kl0",
                    title="Sampler-Trainer KL0 Check",
                    severity="pass",
                    headline="Exact KL0 (0.000000 across all steps)",
                    detail=(
                        f"Both optim/kl_sample_train_v1 and _v2 were 0.0 on all {len(kl_v1_vals)} "
                        "steps — token IDs and sampling logprobs match training forward passes."
                    ),
                    metrics={"max_abs_kl_v1": max_kl_v1, "max_abs_kl_v2": max_kl_v2},
                )
            )
        elif worst_kl < 1e-4:
            findings.append(
                DiagnosticFinding(
                    check_id="kl0",
                    title="Sampler-Trainer KL0 Check",
                    severity="warn",
                    headline=(
                        f"Small KL drift detected (max |KL| = {worst_kl:.2e}, "
                        f"first at step {first_nonzero_kl})"
                    ),
                    detail=(
                        "Small non-zero KL between sampler and trainer logprobs can arise from "
                        "numerical differences or minor renderer/tokenization mismatches."
                    ),
                    step=first_nonzero_kl,
                    metrics={"max_abs_kl_v1": max_kl_v1, "max_abs_kl_v2": max_kl_v2},
                )
            )
        else:
            findings.append(
                DiagnosticFinding(
                    check_id="kl0",
                    title="Sampler-Trainer KL0 Check",
                    severity="fail",
                    headline=(
                        f"KL0 broken: max |KL| = {worst_kl:.4f} "
                        f"(started at step {first_nonzero_kl})"
                    ),
                    detail=(
                        "Large divergence between sampled and trained logprobs indicates a token "
                        "bridging, chat-template, or off-policy mismatch. Importance sampling "
                        "ratios may destabilize training."
                    ),
                    step=first_nonzero_kl,
                    metrics={"max_abs_kl_v1": max_kl_v1, "max_abs_kl_v2": max_kl_v2},
                )
            )

    # 2. Constant-reward group filtering & effective batch size
    total_groups_sampled: int | None = None
    total_groups_kept: int | None = None
    total_groups_all_good: int | None = None
    total_groups_all_bad: int | None = None
    total_eps_sampled: int | None = None
    total_eps_kept: int | None = None
    discarded_group_frac: float | None = None
    per_step_drop_stats: list[dict[str, Any]] = []

    if run.groups_by_step:
        total_groups_sampled = 0
        total_groups_kept = 0
        total_groups_all_good = 0
        total_groups_all_bad = 0
        total_eps_sampled = 0
        total_eps_kept = 0

        for step_idx in sorted(run.groups_by_step):
            groups = run.groups_by_step[step_idx]
            n_sampled = len(groups)
            n_kept = sum(1 for g in groups if g.kept_after_filter)
            n_mixed = sum(1 for g in groups if g.status == "mixed")
            n_good = sum(1 for g in groups if g.status == "all_good")
            n_bad = sum(1 for g in groups if g.status == "all_bad")
            n_const_other = sum(1 for g in groups if g.status == "constant")
            eps_sampled = sum(len(g.trajectories) for g in groups)
            eps_kept = sum(len(g.trajectories) for g in groups if g.kept_after_filter)

            total_groups_sampled += n_sampled
            total_groups_kept += n_kept
            total_groups_all_good += n_good
            total_groups_all_bad += n_bad
            total_eps_sampled += eps_sampled
            total_eps_kept += eps_kept

            per_step_drop_stats.append(
                {
                    "step": step_idx,
                    "groups_sampled": n_sampled,
                    "groups_kept": n_kept,
                    "groups_mixed": n_mixed,
                    "groups_all_good": n_good,
                    "groups_all_bad": n_bad,
                    "groups_constant_other": n_const_other,
                    "episodes_sampled": eps_sampled,
                    "episodes_kept": eps_kept,
                    "drop_frac": (n_sampled - n_kept) / n_sampled if n_sampled else 0.0,
                    "constant_frac": (n_sampled - n_mixed) / n_sampled if n_sampled else 0.0,
                }
            )

        if total_groups_sampled > 0:
            total_constant = total_groups_sampled - sum(
                r["groups_mixed"] for r in per_step_drop_stats
            )
            const_frac = total_constant / total_groups_sampled
            discarded_group_frac = (total_groups_sampled - total_groups_kept) / total_groups_sampled
            if remove_constant:
                sev = "warn" if discarded_group_frac >= 0.50 else "pass"
                findings.append(
                    DiagnosticFinding(
                        check_id="constant_groups",
                        title="Constant-Reward Group Drop Rate",
                        severity=sev,
                        headline=(
                            f"{discarded_group_frac:.0%} of groups dropped "
                            f"({total_groups_kept}/{total_groups_sampled} kept; "
                            f"{total_groups_all_good} all-good, {total_groups_all_bad} all-bad)"
                        ),
                        detail=(
                            f"Reconstructed from pre-filter rollout summaries: {total_eps_sampled} "
                            f"episodes sampled, {total_eps_kept} trained on "
                            f"({total_eps_kept / len(per_step_drop_stats):.1f}/step avg). "
                            "High all-good drop rates suggest the taskset is near saturation for "
                            "this base model and curriculum filtering or harder tasks would "
                            "improve sample efficiency."
                        ),
                        metrics={
                            "discarded_group_frac": round(discarded_group_frac, 4),
                            "all_good": total_groups_all_good,
                            "all_bad": total_groups_all_bad,
                            "kept": total_groups_kept,
                            "sampled": total_groups_sampled,
                        },
                    )
                )
            elif const_frac >= 0.50:
                findings.append(
                    DiagnosticFinding(
                        check_id="constant_groups",
                        title="Zero-Advantage Groups Not Filtered",
                        severity="warn",
                        headline=(
                            f"{const_frac:.0%} of groups have constant reward "
                            "with remove_constant_reward_groups=False"
                        ),
                        detail=(
                            f"{total_constant}/{total_groups_sampled} groups "
                            f"({total_groups_all_good} all-good, {total_groups_all_bad} all-bad) "
                            "have zero advantage yet are still sent to forward_backward. "
                            "Enabling remove_constant_reward_groups=True avoids paying training "
                            "compute on zero-gradient trajectories."
                        ),
                        metrics={"constant_frac": round(const_frac, 4)},
                    )
                )
            else:
                mixed_n = total_groups_sampled - total_constant
                findings.append(
                    DiagnosticFinding(
                        check_id="constant_groups",
                        title="Group Reward Variance",
                        severity="pass",
                        headline=f"{1.0 - const_frac:.0%} of groups have mixed rewards",
                        detail=(
                            f"{mixed_n}/{total_groups_sampled} groups had non-constant rewards."
                        ),
                    )
                )

    # 3. Token & prefill amplification summary
    tokens = _compute_token_summary(run, rates)
    if tokens.prefill_multiplier is not None and tokens.prefill_multiplier >= 2.0:
        cache_note = (
            f" (~{tokens.cacheable_fraction:.0%} prefix-cacheable across turns)"
            if tokens.cacheable_fraction is not None
            else ""
        )
        findings.append(
            DiagnosticFinding(
                check_id="prefill_amplification",
                title="Multi-Turn Prefill Amplification",
                severity="info",
                headline=(
                    f"{tokens.prefill_multiplier:.2f}x prefill-to-unique-token ratio{cache_note}"
                ),
                detail=(
                    f"Total billed prefill ({tokens.prefill_total:,} tokens) is "
                    f"{tokens.prefill_multiplier:.2f}x unique trajectory tokens "
                    f"({tokens.unique_trajectory_tokens:,}) because each turn re-prefills the "
                    "growing conversation history. Prefix caching absorbs repeated history."
                ),
                metrics={
                    "prefill_multiplier": round(tokens.prefill_multiplier, 3),
                    "prefill_total": tokens.prefill_total,
                    "unique_trajectory_tokens": tokens.unique_trajectory_tokens,
                },
            )
        )

    # 4. Wall-clock timing & straggler amplification
    total_wall = 0.0
    sampling_wall_total = 0.0
    train_wall_total = 0.0
    ckpt_wall_total = 0.0
    straggler_pairs: list[tuple[int, float]] = []

    for s in run.steps:
        t_rec = run.timings.get(s.step)
        step_total = s.time_metrics.get("total")
        if step_total is None and t_rec is not None:
            step_total = t_rec.total_wall
        if step_total is not None:
            total_wall += step_total

        samp = s.time_metrics.get("sampling")
        if samp is None and t_rec is not None:
            samp = t_rec.sampling_wall
        if samp is not None:
            sampling_wall_total += samp

        trn = s.time_metrics.get("train_step")
        if trn is None and t_rec is not None:
            trn = t_rec.train_wall
        if trn is not None:
            train_wall_total += trn

        ck = s.time_metrics.get("save_checkpoint")
        if ck is None and t_rec is not None:
            ck = t_rec.checkpoint_wall
        if ck is not None:
            ckpt_wall_total += ck

        ratio: float | None = None
        if t_rec is not None and t_rec.straggler_ratio is not None:
            ratio = t_rec.straggler_ratio
        else:
            r_max = s.time_metrics.get("do_group_rollout_and_filter_constant_reward:max")
            r_mean = s.time_metrics.get("do_group_rollout_and_filter_constant_reward:mean")
            if r_max is not None and r_mean is not None and r_mean > 1e-9:
                ratio = r_max / r_mean
        if ratio is not None:
            straggler_pairs.append((s.step, ratio))

    sampling_frac = (sampling_wall_total / total_wall) if total_wall > 0 else None
    train_frac = (train_wall_total / total_wall) if total_wall > 0 else None
    ckpt_frac = (ckpt_wall_total / total_wall) if total_wall > 0 else None

    median_straggler: float | None = None
    max_straggler: float | None = None
    max_straggler_step: int | None = None
    if straggler_pairs:
        median_straggler = statistics.median(r for _, r in straggler_pairs)
        max_straggler_step, max_straggler = max(straggler_pairs, key=lambda p: p[1])
        sev = "warn" if (median_straggler >= 2.0 or (sampling_frac or 0) >= 0.85) else "pass"
        findings.append(
            DiagnosticFinding(
                check_id="stragglers",
                title="Straggler & Wall-Clock Bottleneck",
                severity=sev,
                headline=(
                    f"Sampling is {(sampling_frac or 0):.0%} of wall time; "
                    f"median straggler ratio {median_straggler:.2f}x "
                    f"(peak {max_straggler:.2f}x at step {max_straggler_step})"
                ),
                detail=(
                    f"Total wall time {total_wall / 60:.1f} min "
                    f"(sampling {sampling_wall_total / 60:.1f} min, "
                    f"train {train_wall_total / 60:.1f} min, "
                    f"checkpoints {ckpt_wall_total / 60:.1f} min). "
                    "In synchronous batches every step waits for the single slowest rollout group; "
                    "stream_minibatch or async training can recover this idle time."
                ),
                step=max_straggler_step,
                metrics={
                    "median_straggler_ratio": round(median_straggler, 3),
                    "max_straggler_ratio": round(max_straggler, 3),
                    "sampling_wall_frac": round(sampling_frac or 0.0, 4),
                },
            )
        )

    # 5. Verbosity / reward-hacking drift watch
    drift_finding = _detect_paired_metric_divergence(run)
    if drift_finding is not None:
        findings.append(drift_finding)

    # 6. Rollout errors, zero_reward_on_limit clamping, & max_tokens truncation check
    max_tokens = run.config.get("max_tokens")
    # RolloutErrorCounter metrics are cumulative across steps, so take the max/final count.
    total_rollout_errors = max((s.rollout_errors.get("total", 0.0) for s in run.steps), default=0.0)
    total_groups_skipped = max(
        (s.rollout_errors.get("groups_skipped", 0.0) for s in run.steps), default=0.0
    )
    clamped_vals = [
        s.env_metrics["zero_reward_on_limit"]
        for s in run.steps
        if "zero_reward_on_limit" in s.env_metrics
    ]
    mean_clamped = _mean(clamped_vals) if clamped_vals else 0.0
    trunc_turns = 0
    all_turns = 0
    if isinstance(max_tokens, int) and max_tokens > 0 and run.groups_by_step:
        for groups in run.groups_by_step.values():
            for g in groups:
                for t in g.trajectories:
                    for turn in t.steps:
                        all_turns += 1
                        if turn.ac_len >= max_tokens:
                            trunc_turns += 1
    trunc_frac = (trunc_turns / all_turns) if all_turns > 0 else 0.0

    if total_rollout_errors > 0 or total_groups_skipped > 0:
        findings.append(
            DiagnosticFinding(
                check_id="rollout_errors",
                title="Rollout Errors",
                severity="warn",
                headline=(
                    f"{int(total_rollout_errors)} rollout errors, "
                    f"{int(total_groups_skipped)} groups skipped"
                ),
                detail=(
                    "Cumulative RolloutErrorCounter totals across the run. Check logs.log "
                    "and rollout_errors/* series for failing environments or rate limits."
                ),
                metrics={
                    "total_rollout_errors": int(total_rollout_errors),
                    "total_groups_skipped": int(total_groups_skipped),
                },
            )
        )
    if mean_clamped >= 0.05:
        findings.append(
            DiagnosticFinding(
                check_id="zero_reward_on_limit",
                title="Limit Reward Clamping (zero_reward_on_limit)",
                severity="warn",
                headline=f"{mean_clamped:.1%} of trajectories clamped to 0.0 on turn/token limit",
                detail=(
                    "Agentic presets (such as Inkling defaults) clamp positive rewards to 0.0 "
                    "when a trajectory hits max_turns, max_tool_calls, or max_trajectory_tokens."
                ),
                metrics={"mean_clamped_frac": round(mean_clamped, 4)},
            )
        )
    if all_turns > 0 and trunc_frac >= 0.05:
        findings.append(
            DiagnosticFinding(
                check_id="truncation",
                title="Max-Token Truncation",
                severity="warn",
                headline=(
                    f"{trunc_frac:.1%} of turns hit max_tokens={max_tokens} "
                    f"({trunc_turns}/{all_turns})"
                ),
                detail=(
                    "Generations hitting max_tokens are cut off mid-turn, which can break tool "
                    "call parsing or truncate answers."
                ),
                metrics={"trunc_frac": round(trunc_frac, 4), "trunc_turns": trunc_turns},
            )
        )

    # 7. Multi-turn tool-use & prefix-cache diagnostics
    tool_diag, tool_finding = _compute_tool_diagnostics(
        run,
        trunc_frac=trunc_frac if all_turns > 0 else None,
    )
    if tool_finding is not None:
        findings.append(tool_finding)

    # Collect available scalar metrics & tags
    skip_metrics = {
        "ac_tokens_per_turn",
        "ob_tokens_per_turn",
        "turns_per_episode",
        "total_episodes",
        "total_turns",
        "total_ac_tokens",
        "total_ob_tokens",
        "by_group/frac_mixed",
        "by_group/frac_all_good",
        "by_group/frac_all_bad",
    }
    avail_metrics = sorted({k for s in run.steps for k in s.env_metrics if k not in skip_metrics})
    avail_tags = sorted({tag for s in run.steps for tag in s.tag_metrics})

    return RunDiagnostics(
        run_name=run.name,
        model_name=run.config.get("model_name"),
        renderer_name=run.config.get("renderer_name"),
        num_steps=len(run.steps),
        total_wall_seconds=total_wall,
        sampling_wall_fraction=sampling_frac,
        train_wall_fraction=train_frac,
        checkpoint_wall_fraction=ckpt_frac,
        max_abs_kl_v1=max_kl_v1,
        max_abs_kl_v2=max_kl_v2,
        first_nonzero_kl_step=first_nonzero_kl,
        total_groups_sampled=total_groups_sampled,
        total_groups_kept=total_groups_kept,
        total_groups_all_good=total_groups_all_good,
        total_groups_all_bad=total_groups_all_bad,
        total_episodes_sampled=total_eps_sampled,
        total_episodes_kept=total_eps_kept,
        discarded_group_fraction=discarded_group_frac,
        median_straggler_ratio=median_straggler,
        max_straggler_ratio=max_straggler,
        max_straggler_step=max_straggler_step,
        tokens=tokens,
        findings=findings,
        per_step_drop_stats=per_step_drop_stats,
        available_metrics=avail_metrics,
        available_tags=avail_tags,
        tools=tool_diag,
    )


def _first_present(d: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _compute_tool_diagnostics(
    run: RunData,
    *,
    trunc_frac: float | None,
) -> tuple[ToolDiagnostics | None, DiagnosticFinding | None]:
    """Compute multi-turn tool-use, limit-hit, and per-step prefix-cache diagnostics."""
    used_keys = ("used_python", "used_tool", "tool_used")
    calls_keys = ("tool_calls_count", "tool_calls", "num_tool_calls")
    err_keys = ("tool_errors", "tool_error_count", "invalid_tool_calls", "tool_call_errors")
    max_turns_keys = ("hit_max_turns", "max_turns_reached")

    # Prefer pre-filter trajectory_metrics from rollout summaries when available,
    # falling back to step env_metrics.
    all_trajs = [
        tr for groups in run.groups_by_step.values() for g in groups for tr in g.trajectories
    ]

    def _collect(keys: tuple[str, ...]) -> list[float]:
        if all_trajs:
            vals = [
                v
                for tr in all_trajs
                if (v := _first_present(tr.trajectory_metrics, keys)) is not None
            ]
            if vals:
                return vals
        return [v for s in run.steps if (v := _first_present(s.env_metrics, keys)) is not None]

    used_vals = _collect(used_keys)
    calls_vals = _collect(calls_keys)
    err_vals = _collect(err_keys)
    max_turns_vals = _collect(max_turns_keys)

    turns_vals = (
        [float(tr.num_turns) for tr in all_trajs]
        if all_trajs
        else [s.turns_per_episode for s in run.steps if s.turns_per_episode is not None]
    )
    mean_turns = _mean(turns_vals) if turns_vals else None

    if not used_vals and calls_vals:
        used_vals = [1.0 if v > 0 else 0.0 for v in calls_vals]

    used_rate = _mean(used_vals) if used_vals else None
    mean_calls = _mean(calls_vals) if calls_vals else None
    mean_errors = _mean(err_vals) if err_vals else None
    hit_max_turns = _mean(max_turns_vals) if max_turns_vals else None

    # Per-step prefix-cache hit fraction + step-level tool metrics
    per_step_cache: list[dict[str, Any]] = []
    for s in run.steps:
        cached = s.native_tokens.get("prefill_cached", 0) + s.native_discarded_tokens.get(
            "prefill_cached", 0
        )
        uncached = s.native_tokens.get("prefill_uncached", 0) + s.native_discarded_tokens.get(
            "prefill_uncached", 0
        )
        if cached + uncached == 0 and s.step in run.groups_by_step:
            cached = sum(g.estimated_prefill_cached for g in run.groups_by_step[s.step])
            uncached = sum(g.estimated_prefill_uncached for g in run.groups_by_step[s.step])
        total_pre = cached + uncached
        row: dict[str, Any] = {
            "step": s.step,
            "cache_hit_frac": round(cached / total_pre, 4) if total_pre > 0 else None,
            "used_tool": _first_present(s.env_metrics, used_keys),
            "tool_calls": _first_present(s.env_metrics, calls_keys),
            "tool_errors": _first_present(s.env_metrics, err_keys),
            "hit_max_turns": _first_present(s.env_metrics, max_turns_keys),
        }
        if any(
            row[k] is not None
            for k in ("cache_hit_frac", "used_tool", "tool_calls", "tool_errors", "hit_max_turns")
        ):
            per_step_cache.append(row)

    has_tool_metrics = any(
        v is not None for v in (used_rate, mean_calls, mean_errors, hit_max_turns)
    )
    is_multiturn = mean_turns is not None and mean_turns > 1.05
    if not has_tool_metrics and not is_multiturn:
        return None, None

    td = ToolDiagnostics(
        used_tool_rate=round(used_rate, 4) if used_rate is not None else None,
        mean_tool_calls=round(mean_calls, 3) if mean_calls is not None else None,
        mean_tool_errors=round(mean_errors, 3) if mean_errors is not None else None,
        hit_max_turns_rate=round(hit_max_turns, 4) if hit_max_turns is not None else None,
        hit_max_tokens_rate=round(trunc_frac, 4) if trunc_frac is not None else None,
        mean_turns_per_episode=round(mean_turns, 2) if mean_turns is not None else None,
        per_step_cache_hit=per_step_cache,
    )

    finding: DiagnosticFinding | None = None
    if has_tool_metrics:
        high_errors = mean_errors is not None and mean_errors >= 0.4
        high_max_turns = hit_max_turns is not None and hit_max_turns >= 0.25
        sev: str = "warn" if (high_errors or high_max_turns) else "pass"
        parts: list[str] = []
        if used_rate is not None:
            parts.append(f"{used_rate:.0%} used tools")
        if mean_calls is not None:
            parts.append(f"{mean_calls:.2f} calls/ep")
        if mean_errors is not None:
            parts.append(f"{mean_errors:.2f} errors/ep")
        if hit_max_turns is not None:
            parts.append(f"{hit_max_turns:.1%} hit max_turns")
        headline = ", ".join(parts)
        detail = (
            f"Multi-turn tool execution summary (mean {mean_turns:.2f} turns/ep). "
            if mean_turns is not None
            else "Multi-turn tool execution summary. "
        )
        if high_errors or high_max_turns:
            detail += (
                "Elevated tool errors or max_turns cutoffs indicate episodes burning turns on "
                "failing calls or loops; inspect trajectories and consider tuning thinking effort "
                "or tool error feedback."
            )
        else:
            detail += (
                "Tool calls and turn termination are well within healthy bounds across rollouts."
            )
        finding = DiagnosticFinding(
            check_id="tool_use",
            title="Multi-Turn Tool Execution",
            severity=sev,  # type: ignore[arg-type]
            headline=headline,
            detail=detail,
            metrics={
                "used_tool_rate": td.used_tool_rate,
                "mean_tool_calls": td.mean_tool_calls,
                "mean_tool_errors": td.mean_tool_errors,
                "hit_max_turns_rate": td.hit_max_turns_rate,
                "mean_turns_per_episode": td.mean_turns_per_episode,
            },
        )
    return td, finding


def _bootstrap_ci_mean_diff(
    diffs: list[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (diffs[0], diffs[0])
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        samples.append(s / n)
    samples.sort()
    lo_idx = max(0, int(0.025 * n_boot))
    hi_idx = min(n_boot - 1, int(0.975 * n_boot))
    return (samples[lo_idx], samples[hi_idx])


def _episode_metric_dict(ep: EvalEpisode) -> dict[str, float]:
    out = dict(ep.metrics)
    out.setdefault("reward", ep.reward)
    return out


def diff_evals(
    base_episodes: list[EvalEpisode],
    final_episodes: list[EvalEpisode],
    *,
    base_label: str = "base",
    final_label: str = "final",
    n_boot: int = 2000,
    seed: int = 0,
) -> EvalDiffSummary:
    """Compute paired task-level deltas, 95% bootstrap CIs, and transcript diffs."""
    base_by_task: dict[str, list[EvalEpisode]] = defaultdict(list)
    final_by_task: dict[str, list[EvalEpisode]] = defaultdict(list)
    for ep in base_episodes:
        base_by_task[ep.task_key].append(ep)
    for ep in final_episodes:
        final_by_task[ep.task_key].append(ep)

    shared_keys = [k for k in base_by_task if k in final_by_task]
    if not shared_keys:
        base_by_prompt: dict[str, list[EvalEpisode]] = defaultdict(list)
        final_by_prompt: dict[str, list[EvalEpisode]] = defaultdict(list)
        for ep in base_episodes:
            norm_prompt = " ".join(ep.prompt.split())
            if norm_prompt:
                base_by_prompt[norm_prompt].append(ep)
        for ep in final_episodes:
            norm_prompt = " ".join(ep.prompt.split())
            if norm_prompt:
                final_by_prompt[norm_prompt].append(ep)
        shared_prompts = [p for p in base_by_prompt if p in final_by_prompt]
        if shared_prompts:
            base_by_task = base_by_prompt
            final_by_task = final_by_prompt
            shared_keys = shared_prompts
    if not shared_keys:
        raise ValueError("No matching tasks found between base and final eval files.")

    # Discover all shared metrics across paired tasks
    base_metric_keys = {k for ep in base_episodes for k in _episode_metric_dict(ep)}
    final_metric_keys = {k for ep in final_episodes for k in _episode_metric_dict(ep)}
    shared_metrics = sorted(base_metric_keys & final_metric_keys)
    # Order priority metrics first
    ordered_metrics = [m for m in _PRIMARY_METRIC_PRIORITY if m in shared_metrics]
    ordered_metrics += [m for m in shared_metrics if m not in ordered_metrics]
    primary_metric = ordered_metrics[0] if ordered_metrics else "reward"

    tasks: list[EvalTaskDiff] = []
    for tk in shared_keys:
        b_eps = base_by_task[tk]
        f_eps = final_by_task[tk]
        b_metrics: dict[str, float] = {}
        f_metrics: dict[str, float] = {}
        deltas: dict[str, float] = {}
        for m in ordered_metrics:
            bv = [_episode_metric_dict(e)[m] for e in b_eps if m in _episode_metric_dict(e)]
            fv = [_episode_metric_dict(e)[m] for e in f_eps if m in _episode_metric_dict(e)]
            if bv and fv:
                bm, fm = _mean(bv), _mean(fv)
                b_metrics[m] = bm
                f_metrics[m] = fm
                deltas[m] = fm - bm

        b_chars = _mean([float(e.final_answer_chars) for e in b_eps])
        f_chars = _mean([float(e.final_answer_chars) for e in f_eps])
        ratio = (f_chars / b_chars) if b_chars > 0 else None
        b_turns = _mean([float(e.num_turns) for e in b_eps])
        f_turns = _mean([float(e.num_turns) for e in f_eps])

        tasks.append(
            EvalTaskDiff(
                task_key=b_eps[0].task_key,
                task_id=b_eps[0].task_id,
                prompt=b_eps[0].prompt or f_eps[0].prompt,
                base_count=len(b_eps),
                final_count=len(f_eps),
                base_metrics=b_metrics,
                final_metrics=f_metrics,
                deltas=deltas,
                base_mean_answer_chars=b_chars,
                final_mean_answer_chars=f_chars,
                answer_chars_ratio=ratio,
                base_mean_turns=b_turns,
                final_mean_turns=f_turns,
                base_episodes=b_eps,
                final_episodes=f_eps,
            )
        )

    metric_deltas: list[EvalMetricDelta] = []
    for m in ordered_metrics:
        b_vals = [t.base_metrics[m] for t in tasks if m in t.base_metrics and m in t.final_metrics]
        f_vals = [t.final_metrics[m] for t in tasks if m in t.base_metrics and m in t.final_metrics]
        d_vals = [f - b for b, f in zip(b_vals, f_vals, strict=True)]
        if not d_vals:
            continue
        ci_lo, ci_hi = _bootstrap_ci_mean_diff(d_vals, n_boot=n_boot, seed=seed)
        wins = sum(1 for d in d_vals if d > 1e-9)
        regs = sum(1 for d in d_vals if d < -1e-9)
        ties = len(d_vals) - wins - regs
        metric_deltas.append(
            EvalMetricDelta(
                metric=m,
                base_mean=_mean(b_vals),
                final_mean=_mean(f_vals),
                delta=_mean(d_vals),
                ci_low=ci_lo,
                ci_high=ci_hi,
                wins=wins,
                regressions=regs,
                ties=ties,
            )
        )

    base_chars_all = _mean([t.base_mean_answer_chars for t in tasks])
    final_chars_all = _mean([t.final_mean_answer_chars for t in tasks])
    chars_pct = (
        ((final_chars_all - base_chars_all) / base_chars_all) * 100.0
        if base_chars_all > 1e-9
        else None
    )
    base_turns_all = _mean([t.base_mean_turns for t in tasks])
    final_turns_all = _mean([t.final_mean_turns for t in tasks])
    turns_pct = (
        ((final_turns_all - base_turns_all) / base_turns_all) * 100.0
        if base_turns_all > 1e-9
        else None
    )

    return EvalDiffSummary(
        base_label=base_label,
        final_label=final_label,
        num_tasks=len(tasks),
        base_total_episodes=len(base_episodes),
        final_total_episodes=len(final_episodes),
        base_failed=sum(1 for e in base_episodes if not e.ok),
        final_failed=sum(1 for e in final_episodes if not e.ok),
        primary_metric=primary_metric,
        metric_deltas=metric_deltas,
        answer_chars_change_pct=chars_pct,
        turns_change_pct=turns_pct,
        tasks=tasks,
    )
