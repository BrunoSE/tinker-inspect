"""Core diagnostic analyzers for Tinker RL runs and paired evaluation diffs."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from tinker_inspect.schema import (
    DiagnosticFinding,
    EvalDiffSummary,
    EvalEpisode,
    EvalMetricDelta,
    EvalTaskDiff,
    PrefilterStepSummary,
    RunData,
    RunDiagnostics,
    TokenCostRates,
    TokenSummary,
    ToolDiagnostics,
)
from tinker_inspect.stats import (
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    bootstrap_ci_mean_diff,
    extract_cluster_ids,
)

_PRIMARY_METRIC_PRIORITY = ("success", "outcome", "correct", "reward")
_OUTCOME_METRIC_NAMES = frozenset(
    {"success", "outcome", "correct", "reward", "shaped_reward", "format_ok", "parse_ok"}
)


def is_outcome_metric(name: str) -> bool:
    """Return True if ``name`` is a higher-is-better outcome/score metric."""
    return name in _OUTCOME_METRIC_NAMES or name.endswith("_facts")


_USED_TOOL_KEYS = ("used_python", "used_tool", "tool_used")
_TOOL_CALLS_KEYS = ("tool_calls_count", "tool_calls", "num_tool_calls")
_TOOL_ERR_KEYS = ("tool_errors", "tool_error_count", "invalid_tool_calls", "tool_call_errors")
_MAX_TURNS_KEYS = ("hit_max_turns", "max_turns_reached")


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _first_present(d: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _compute_prefilter_steps(
    run: RunData,
    max_tokens: int | None,
) -> tuple[
    list[PrefilterStepSummary],
    float | None,
    float | None,
    int,
    str,
    list[dict[str, Any]],
]:
    """Compute ground-truth pre-filter per-step summaries and early-vs-late windows."""
    pf_steps: list[PrefilterStepSummary] = []

    if run.groups_by_step:
        for step_idx in sorted(run.groups_by_step):
            groups = run.groups_by_step[step_idx]
            all_trajs = [tr for g in groups for tr in g.trajectories]
            kept_trajs = [tr for g in groups if g.kept_after_filter for tr in g.trajectories]
            if not all_trajs:
                continue

            total_ac = sum(tr.sum_ac_len for tr in all_trajs)
            total_turns = sum(tr.num_turns for tr in all_trajs)
            ac_per_turn = (total_ac / total_turns) if total_turns > 0 else 0.0

            used_vals = [
                v
                for tr in all_trajs
                if (v := _first_present(tr.trajectory_metrics, _USED_TOOL_KEYS)) is not None
            ]
            calls_vals = [
                v
                for tr in all_trajs
                if (v := _first_present(tr.trajectory_metrics, _TOOL_CALLS_KEYS)) is not None
            ]
            if not used_vals and calls_vals:
                used_vals = [1.0 if v > 0 else 0.0 for v in calls_vals]
            err_vals = [
                v
                for tr in all_trajs
                if (v := _first_present(tr.trajectory_metrics, _TOOL_ERR_KEYS)) is not None
            ]
            max_turns_vals = [
                v
                for tr in all_trajs
                if (v := _first_present(tr.trajectory_metrics, _MAX_TURNS_KEYS)) is not None
            ]
            parse_ok_vals = [
                tr.trajectory_metrics["parse_ok"]
                for tr in all_trajs
                if "parse_ok" in tr.trajectory_metrics
            ]

            end_counts: Counter[str] = Counter()
            trunc_ep_hits = 0
            for tr in all_trajs:
                es = tr.end_state(max_tokens)
                end_counts[es] += 1
                if es == "token_limit":
                    trunc_ep_hits += 1

            tag_buckets: dict[str, list[float]] = defaultdict(list)
            for g in groups:
                for tr in g.trajectories:
                    tags = tr.tags or g.tags
                    for tag in tags:
                        tag_buckets[tag].append(tr.total_reward)
            tag_rewards = {
                tag: (round(_mean(vals), 4), len(vals)) for tag, vals in sorted(tag_buckets.items())
            }

            pf_steps.append(
                PrefilterStepSummary(
                    step=step_idx,
                    episodes_sampled=len(all_trajs),
                    episodes_kept=len(kept_trajs),
                    mean_reward=round(_mean([tr.total_reward for tr in all_trajs]), 4),
                    kept_mean_reward=(
                        round(_mean([tr.total_reward for tr in kept_trajs]), 4)
                        if kept_trajs
                        else None
                    ),
                    mean_turns=round(_mean([float(tr.num_turns) for tr in all_trajs]), 3),
                    ac_tokens_per_turn=round(ac_per_turn, 2),
                    used_tool_rate=round(_mean(used_vals), 4) if used_vals else None,
                    tool_calls_per_ep=round(_mean(calls_vals), 3) if calls_vals else None,
                    tool_errors_per_ep=round(_mean(err_vals), 3) if err_vals else None,
                    trunc_episode_rate=round(trunc_ep_hits / len(all_trajs), 4),
                    max_turns_rate=round(_mean(max_turns_vals), 4) if max_turns_vals else None,
                    parse_error_rate=(
                        round(sum(1 for v in parse_ok_vals if v < 0.5) / len(parse_ok_vals), 4)
                        if parse_ok_vals
                        else None
                    ),
                    tag_rewards=tag_rewards,
                    end_states=dict(end_counts),
                )
            )

    if pf_steps:
        q = max(1, len(pf_steps) // 4)
        early_slice = pf_steps[:q]
        late_slice = pf_steps[-q:]

        def _ep_weighted_mean(slice_steps: list[PrefilterStepSummary]) -> float:
            tot_n = sum(s.episodes_sampled for s in slice_steps)
            if tot_n <= 0:
                return 0.0
            return sum(s.mean_reward * s.episodes_sampled for s in slice_steps) / tot_n

        early_mean = round(_ep_weighted_mean(early_slice), 4)
        late_mean = round(_ep_weighted_mean(late_slice), 4)

        all_tags = sorted({tag for s in pf_steps for tag in s.tag_rewards})
        tag_window_summary: list[dict[str, Any]] = []
        for tag in all_tags:
            e_sum, e_n = 0.0, 0
            for s in early_slice:
                if tag in s.tag_rewards:
                    m_val, cnt = s.tag_rewards[tag]
                    e_sum += m_val * cnt
                    e_n += cnt
            l_sum, l_n = 0.0, 0
            for s in late_slice:
                if tag in s.tag_rewards:
                    m_val, cnt = s.tag_rewards[tag]
                    l_sum += m_val * cnt
                    l_n += cnt
            t_sum, t_n = 0.0, 0
            for s in pf_steps:
                if tag in s.tag_rewards:
                    m_val, cnt = s.tag_rewards[tag]
                    t_sum += m_val * cnt
                    t_n += cnt
            tag_window_summary.append(
                {
                    "tag": tag,
                    "early_mean": round(e_sum / e_n, 4) if e_n > 0 else None,
                    "early_n": e_n,
                    "late_mean": round(l_sum / l_n, 4) if l_n > 0 else None,
                    "late_n": l_n,
                    "overall_mean": round(t_sum / t_n, 4) if t_n > 0 else 0.0,
                    "total_n": t_n,
                }
            )
        return pf_steps, early_mean, late_mean, q, "prefilter", tag_window_summary

    # Fallback: post-filter step rewards from metrics.jsonl
    rew_steps = [s.reward_total for s in run.steps if s.reward_total is not None]
    if rew_steps:
        q = max(1, len(rew_steps) // 4)
        return (
            [],
            round(_mean(rew_steps[:q]), 4),
            round(_mean(rew_steps[-q:]), 4),
            q,
            "postfilter",
            [],
        )
    return [], None, None, 0, "postfilter", []


def _compute_token_summary(run: RunData, rates: TokenCostRates | None) -> TokenSummary:
    all_token_steps = [
        *run.steps,
        *run.discard_rows,
        *([run.summary_step] if run.summary_step is not None else []),
    ]
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

        # Distinguish per-step discarded rows (on step records) from cumulative discard rows
        # emitted by tinker-cookbook main (run.discard_rows + run.summary_step).
        step_has_disc = any(bool(s.native_discarded_tokens) for s in run.steps)
        if step_has_disc:
            disc_sources = [
                *run.steps,
                *([run.summary_step] if run.summary_step is not None else []),
            ]
            disc_cached = sum(
                s.native_discarded_tokens.get("prefill_cached", 0) for s in disc_sources
            )
            disc_uncached = sum(
                s.native_discarded_tokens.get("prefill_uncached", 0) for s in disc_sources
            )
            disc_sampled = sum(s.native_discarded_tokens.get("sampled", 0) for s in disc_sources)
        elif run.summary_step is not None and run.summary_step.native_discarded_tokens:
            d = run.summary_step.native_discarded_tokens
            disc_cached = d.get("prefill_cached", 0)
            disc_uncached = d.get("prefill_uncached", 0)
            disc_sampled = d.get("sampled", 0)
        elif run.discard_rows:
            d = run.discard_rows[-1].native_discarded_tokens
            disc_cached = d.get("prefill_cached", 0)
            disc_uncached = d.get("prefill_uncached", 0)
            disc_sampled = d.get("sampled", 0)
        else:
            disc_cached = disc_uncached = disc_sampled = 0

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
        disc_sampled_only_frac = (disc_sampled / sampled_total) if sampled_total > 0 else None

        native_step_costs = [
            s.native_cost_usd for s in all_token_steps if s.native_cost_usd is not None
        ]
        disc_cost_sources = (
            run.steps
            if step_has_disc
            else ([run.summary_step] if run.summary_step else run.discard_rows[-1:])
        )
        native_disc_costs = [
            s.native_discarded_cost_usd
            for s in disc_cost_sources
            if s is not None and s.native_discarded_cost_usd is not None
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
            discarded_sampled_only_fraction=disc_sampled_only_frac,
            cache_metric_label="Measured cache hits",
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
                    trained_total += g.unique_final_ob_len + sum(
                        t.steps[-1].ac_len for t in g.trajectories if t.steps
                    )
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
        disc_sampled_only_frac = (disc_sampled / sampled_total) if sampled_total > 0 else None

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
            discarded_sampled_only_fraction=disc_sampled_only_frac,
            cache_metric_label="Est. prefix-cacheable",
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
        cache_metric_label="Est. prefix-cacheable",
    )


def _detect_paired_metric_divergence(
    run: RunData,
    prefilter_steps: list[PrefilterStepSummary],
) -> DiagnosticFinding | None:
    """Detect length inflation (preferring pre-filter lengths) or paired-metric divergence."""
    use_prefilter = len(prefilter_steps) >= 4
    if use_prefilter:
        q = max(1, len(prefilter_steps) // 4)
        ac_first = [s.ac_tokens_per_turn for s in prefilter_steps[:q]]
        ac_last = [s.ac_tokens_per_turn for s in prefilter_steps[-q:]]
        len_source = "pre-filter"
    elif len(run.steps) >= 4:
        q = max(1, len(run.steps) // 4)
        ac_first = [s.ac_tokens_per_turn for s in run.steps[:q] if s.ac_tokens_per_turn is not None]
        ac_last = [s.ac_tokens_per_turn for s in run.steps[-q:] if s.ac_tokens_per_turn is not None]
        len_source = "post-filter"
    else:
        return None

    len_growth_pct: float | None = None
    if ac_first and ac_last and _mean(ac_first) > 1e-9:
        len_growth_pct = ((_mean(ac_last) - _mean(ac_first)) / _mean(ac_first)) * 100.0

    # Check known paired metric families (e.g. outcome vs outcome_facts, success vs success_facts).
    divergent_pair: tuple[str, str, float, float] | None = None
    if len(run.steps) >= 4:
        sq = max(1, len(run.steps) // 4)
        first_slice = run.steps[:sq]
        last_slice = run.steps[-sq:]
        all_keys = {k for s in run.steps for k in s.env_metrics}
        for base_key in ("success", "outcome"):
            facts_key = f"{base_key}_facts"
            if base_key in all_keys and facts_key in all_keys:
                b_first = [
                    s.env_metrics[base_key] for s in first_slice if base_key in s.env_metrics
                ]
                b_last = [s.env_metrics[base_key] for s in last_slice if base_key in s.env_metrics]
                f_first = [
                    s.env_metrics[facts_key] for s in first_slice if facts_key in s.env_metrics
                ]
                f_last = [
                    s.env_metrics[facts_key] for s in last_slice if facts_key in s.env_metrics
                ]
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
                f"Action tokens/turn ({len_source}) grew {len_growth_pct:+.0f}% while "
                f"{bk} ({bd:+.3f}) outpaced {fk} ({fd:+.3f})"
            ),
            detail=(
                f"Comparing the first {q} vs last {q} steps ({len_source}): mean "
                f"ac_tokens_per_turn rose from {_mean(ac_first):.1f} to "
                f"{_mean(ac_last):.1f} ({len_growth_pct:+.1f}%). "
                f"Paired metric `{bk}` gained {bd:+.3f} while `{fk}` moved {fd:+.3f}. "
                "Inspect eval diffs to verify the model is not gaming a string-matching grader "
                "via longer outputs."
            ),
            metrics={
                "len_growth_pct": round(len_growth_pct, 2),
                "source": len_source,
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
            headline=(
                f"Action tokens/turn ({len_source}) increased {len_growth_pct:+.0f}% over training"
            ),
            detail=(
                f"Mean {len_source} ac_tokens_per_turn moved from {_mean(ac_first):.1f} "
                f"(first {q} steps) to {_mean(ac_last):.1f} (last {q} steps). Check whether longer "
                "generations reflect genuine reasoning or verbosity/paste gaming."
            ),
            metrics={"len_growth_pct": round(len_growth_pct, 2), "source": len_source},
        )
    if len_growth_pct is not None:
        if len_growth_pct <= -10.0:
            headline = (
                f"Action tokens/turn ({len_source}) fell {abs(len_growth_pct):.1f}% early vs late"
            )
        else:
            headline = (
                f"Action tokens/turn ({len_source}) stable ({len_growth_pct:+.1f}% early vs late)"
            )
        return DiagnosticFinding(
            check_id="verbosity_drift",
            title="Response Length Stability",
            severity="pass",
            headline=headline,
            detail=(
                f"Mean {len_source} ac_tokens_per_turn moved from {_mean(ac_first):.1f} "
                f"(first {q} steps) to {_mean(ac_last):.1f} (last {q} steps)."
            ),
            metrics={"len_growth_pct": round(len_growth_pct, 2), "source": len_source},
        )
    return None


def analyze_run(run: RunData, rates: TokenCostRates | None = None) -> RunDiagnostics:
    """Compute all health checks, token/cost stats, and timing breakdowns for a run."""
    findings: list[DiagnosticFinding] = []
    remove_constant = bool(run.config.get("remove_constant_reward_groups", False))
    max_tokens_raw = run.config.get("max_tokens")
    max_tokens = max_tokens_raw if isinstance(max_tokens_raw, int) and max_tokens_raw > 0 else None

    # 0. Pre-filter per-step summaries and early-vs-late windows
    (
        prefilter_steps,
        reward_early_mean,
        reward_late_mean,
        reward_window_size,
        reward_source,
        tag_window_summary,
    ) = _compute_prefilter_steps(run, max_tokens)

    # 1. KL0 & sampler-trainer mismatch check (config-aware for off-policy setups)
    kl_v1_vals = [(s.step, abs(s.kl_v1)) for s in run.steps if s.kl_v1 is not None]
    kl_v2_vals = [(s.step, abs(s.kl_v2)) for s in run.steps if s.kl_v2 is not None]
    max_kl_v1 = max((v for _, v in kl_v1_vals), default=None)
    max_kl_v2 = max((v for _, v in kl_v2_vals), default=None)
    nonzero_steps = [step for step, v in kl_v1_vals + kl_v2_vals if v > 1e-12]
    first_nonzero_kl = min(nonzero_steps) if nonzero_steps else None

    mismatch_fracs = [
        s.sample_train_mismatch_frac for s in run.steps if s.sample_train_mismatch_frac is not None
    ]
    mismatch_diffs = [
        s.sample_train_max_abs_diff for s in run.steps if s.sample_train_max_abs_diff is not None
    ]
    max_mismatch_frac = max(mismatch_fracs, default=None)
    max_mismatch_diff = max(mismatch_diffs, default=None)

    num_substeps = run.config.get("num_substeps")
    is_off_policy_cfg = (
        (isinstance(num_substeps, int) and num_substeps > 1)
        or run.config.get("async_config") is not None
        or run.config.get("stream_minibatch_config") is not None
    )

    mismatch_note = ""
    if max_mismatch_frac is not None or max_mismatch_diff is not None:
        parts = []
        if max_mismatch_frac is not None:
            parts.append(f"max mismatch_frac={max_mismatch_frac:.4f}")
        if max_mismatch_diff is not None:
            parts.append(f"max |logp_diff|={max_mismatch_diff:.2e}")
        mismatch_note = " (" + ", ".join(parts) + ")"

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
                    headline=f"Exact KL0 (0.000000 across all steps){mismatch_note}",
                    detail=(
                        f"Both optim/kl_sample_train_v1 and _v2 were 0.0 on all {len(kl_v1_vals)} "
                        "steps — token IDs and sampling logprobs match training forward passes."
                    ),
                    metrics={
                        "max_abs_kl_v1": max_kl_v1,
                        "max_abs_kl_v2": max_kl_v2,
                        "max_sample_train_mismatch_frac": max_mismatch_frac,
                        "max_sample_train_max_abs_diff": max_mismatch_diff,
                    },
                )
            )
        elif is_off_policy_cfg:
            sev = "warn" if worst_kl >= 1e-2 else "info"
            findings.append(
                DiagnosticFinding(
                    check_id="kl0",
                    title="Sampler-Trainer KL Check (Off-Policy Config)",
                    severity=sev,  # type: ignore[arg-type]
                    headline=(
                        f"Expected off-policy KL: max |KL| = {worst_kl:.4f} "
                        f"(first at step {first_nonzero_kl}){mismatch_note}"
                    ),
                    detail=(
                        "Run config uses off-policy updates (num_substeps > 1, async, or "
                        "stream_minibatch), so non-zero sampler-trainer KL is expected."
                    ),
                    step=first_nonzero_kl,
                    metrics={
                        "max_abs_kl_v1": max_kl_v1,
                        "max_abs_kl_v2": max_kl_v2,
                        "off_policy_config": True,
                    },
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
                        f"first at step {first_nonzero_kl}){mismatch_note}"
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
                        f"(started at step {first_nonzero_kl}){mismatch_note}"
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
                        severity=sev,  # type: ignore[arg-type]
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
    if tokens.prefill_multiplier is not None and tokens.prefill_multiplier >= 1.5:
        cache_note = (
            f" ({tokens.cacheable_fraction:.0%} {tokens.cache_metric_label.lower()})"
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
                    f"{tokens.prefill_multiplier:.2f}x unique final-turn prompt tokens "
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

    # 4. Wall-clock timing, sampling idle wait, policy_sample vs env_step breakdown
    total_wall = 0.0
    sampling_wall_total = 0.0
    train_wall_total = 0.0
    ckpt_wall_total = 0.0
    straggler_pairs: list[tuple[int, float]] = []
    idle_fracs: list[float] = []
    policy_sample_means: list[float] = []
    env_step_means: list[float] = []

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

        if (ps := s.time_metrics.get("policy_sample:mean")) is not None:
            policy_sample_means.append(ps)
        if (es := s.time_metrics.get("env_step:mean")) is not None:
            env_step_means.append(es)

        ratio: float | None = None
        if t_rec is not None and t_rec.straggler_ratio is not None:
            ratio = t_rec.straggler_ratio
        else:
            r_max = s.time_metrics.get("do_group_rollout_and_filter_constant_reward:max")
            r_mean = s.time_metrics.get("do_group_rollout_and_filter_constant_reward:mean")
            if r_max is not None and r_mean is not None and r_mean > 1e-9:
                ratio = r_max / r_mean
        if ratio is not None and ratio >= 1.0:
            straggler_pairs.append((s.step, ratio))
            idle_fracs.append(max(0.0, 1.0 - (1.0 / ratio)))

    sampling_frac = (sampling_wall_total / total_wall) if total_wall > 0 else None
    train_frac = (train_wall_total / total_wall) if total_wall > 0 else None
    ckpt_frac = (ckpt_wall_total / total_wall) if total_wall > 0 else None
    sampling_idle_frac = round(statistics.median(idle_fracs), 4) if idle_fracs else None
    policy_sample_mean_s = round(_mean(policy_sample_means), 3) if policy_sample_means else None
    env_step_mean_s = round(_mean(env_step_means), 3) if env_step_means else None

    median_straggler: float | None = None
    max_straggler: float | None = None
    max_straggler_step: int | None = None
    if straggler_pairs:
        median_straggler = statistics.median(r for _, r in straggler_pairs)
        max_straggler_step, max_straggler = max(straggler_pairs, key=lambda p: p[1])
        # Calibrate WARN strictly on straggler idle waste (median ratio >= 2.0 -> >=50% idle wait),
        # NOT raw sampling_wall >= 85% (which is completely normal for synchronous RL).
        sev = "warn" if median_straggler >= 2.0 else "pass"
        idle_pct_str = (
            f"{sampling_idle_frac:.0%} sampling idle wait" if sampling_idle_frac is not None else ""
        )
        breakdown_str = ""
        if policy_sample_mean_s is not None or env_step_mean_s is not None:
            bparts = []
            if policy_sample_mean_s is not None:
                bparts.append(f"policy_sample={policy_sample_mean_s:.2f}s/turn")
            if env_step_mean_s is not None:
                bparts.append(f"env_step={env_step_mean_s:.2f}s/turn")
            breakdown_str = " (" + ", ".join(bparts) + ")"
        findings.append(
            DiagnosticFinding(
                check_id="stragglers",
                title="Straggler & Sampling Idle Time",
                severity=sev,  # type: ignore[arg-type]
                headline=(
                    f"{idle_pct_str} (median straggler {median_straggler:.2f}x, "
                    f"peak {max_straggler:.2f}x at step {max_straggler_step}; "
                    f"sampling {(sampling_frac or 0):.0%} of wall){breakdown_str}"
                ),
                detail=(
                    f"Total wall time {total_wall / 60:.1f} min "
                    f"(sampling {sampling_wall_total / 60:.1f} min, "
                    f"train {train_wall_total / 60:.1f} min, "
                    f"checkpoints {ckpt_wall_total / 60:.1f} min). "
                    "Idle wait = 1 - mean/max rollout duration within each synchronous batch."
                ),
                step=max_straggler_step,
                metrics={
                    "median_straggler_ratio": round(median_straggler, 3),
                    "max_straggler_ratio": round(max_straggler, 3),
                    "sampling_idle_frac": sampling_idle_frac,
                    "sampling_wall_frac": round(sampling_frac or 0.0, 4),
                    "policy_sample_mean_s": policy_sample_mean_s,
                    "env_step_mean_s": env_step_mean_s,
                },
            )
        )

    # 5. Verbosity / reward-hacking drift watch (pre-filter lengths when available)
    drift_finding = _detect_paired_metric_divergence(run, prefilter_steps)
    if drift_finding is not None:
        findings.append(drift_finding)

    # 6. Rollout errors, zero_reward_on_limit clamping, & episode-level truncation check
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

    all_trajs = [
        tr for groups in run.groups_by_step.values() for g in groups for tr in g.trajectories
    ]
    trunc_turns = 0
    all_turns = 0
    trunc_ep_rewards: list[float] = []
    untrunc_ep_rewards: list[float] = []
    end_state_counts: Counter[str] = Counter()

    for tr in all_trajs:
        es = tr.end_state(max_tokens)
        end_state_counts[es] += 1
        if es == "token_limit":
            trunc_ep_rewards.append(tr.total_reward)
        else:
            untrunc_ep_rewards.append(tr.total_reward)
        if max_tokens is not None:
            for turn in tr.steps:
                all_turns += 1
                if turn.ac_len >= max_tokens:
                    trunc_turns += 1

    trunc_turn_frac = (trunc_turns / all_turns) if all_turns > 0 else None
    trunc_ep_rate = (len(trunc_ep_rewards) / len(all_trajs)) if all_trajs else None
    trunc_rew_mean = round(_mean(trunc_ep_rewards), 3) if trunc_ep_rewards else None
    untrunc_rew_mean = round(_mean(untrunc_ep_rewards), 3) if untrunc_ep_rewards else None

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
    if trunc_ep_rate is not None:
        sev = "warn" if trunc_ep_rate >= 0.05 else "pass"
        rew_cmp = ""
        if trunc_rew_mean is not None and untrunc_rew_mean is not None:
            rew_cmp = (
                f" (reward: {trunc_rew_mean:.2f} truncated vs {untrunc_rew_mean:.2f} untruncated)"
            )
        turn_note = (
            f"; {trunc_turn_frac:.1%} of turns ({trunc_turns}/{all_turns}) "
            f"hit max_tokens={max_tokens}"
            if trunc_turn_frac is not None and max_tokens is not None
            else ""
        )
        findings.append(
            DiagnosticFinding(
                check_id="truncation",
                title="Episode Token-Limit Truncation",
                severity=sev,  # type: ignore[arg-type]
                headline=(
                    f"{trunc_ep_rate:.1%} of episodes hit token limit "
                    f"({len(trunc_ep_rewards)}/{len(all_trajs)}){rew_cmp}"
                ),
                detail=(
                    f"Measured across all {len(all_trajs)} sampled episodes pre-filter"
                    f"{turn_note}. Generations cut off mid-thinking or mid-turn often lose "
                    "their final answer or break tool-call parsing."
                ),
                metrics={
                    "trunc_episode_rate": round(trunc_ep_rate, 4),
                    "trunc_episodes": len(trunc_ep_rewards),
                    "total_episodes": len(all_trajs),
                    "trunc_reward_mean": trunc_rew_mean,
                    "untrunc_reward_mean": untrunc_rew_mean,
                    "trunc_turn_frac": round(trunc_turn_frac, 4)
                    if trunc_turn_frac is not None
                    else None,
                },
            )
        )

    # 7. Multi-turn tool-use & prefix-cache diagnostics (pre-filter trajectories first!)
    tool_diag, tool_finding = _compute_tool_diagnostics(
        run,
        prefilter_steps=prefilter_steps,
        trunc_turn_frac=trunc_turn_frac,
        trunc_ep_rate=trunc_ep_rate,
        trunc_rew_mean=trunc_rew_mean,
        untrunc_rew_mean=untrunc_rew_mean,
        end_state_counts=dict(end_state_counts),
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
    avail_tags = sorted(
        {tag for s in run.steps for tag in s.tag_metrics}
        | {tag for ps in prefilter_steps for tag in ps.tag_rewards}
    )

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
        prefilter_steps=prefilter_steps,
        reward_early_mean=reward_early_mean,
        reward_late_mean=reward_late_mean,
        reward_window_size=reward_window_size,
        reward_source=reward_source,
        sampling_idle_fraction=sampling_idle_frac,
        policy_sample_mean_s=policy_sample_mean_s,
        env_step_mean_s=env_step_mean_s,
        max_sample_train_mismatch_frac=max_mismatch_frac,
        max_sample_train_max_abs_diff=max_mismatch_diff,
        tag_window_summary=tag_window_summary,
    )


def _compute_tool_diagnostics(
    run: RunData,
    *,
    prefilter_steps: list[PrefilterStepSummary],
    trunc_turn_frac: float | None,
    trunc_ep_rate: float | None,
    trunc_rew_mean: float | None,
    untrunc_rew_mean: float | None,
    end_state_counts: dict[str, int],
) -> tuple[ToolDiagnostics | None, DiagnosticFinding | None]:
    """Compute multi-turn tool-use, limit-hit, and per-step prefix-cache diagnostics."""
    all_trajs = [
        tr for groups in run.groups_by_step.values() for g in groups for tr in g.trajectories
    ]

    used_prefilter = False

    def _collect(keys: tuple[str, ...]) -> list[float]:
        nonlocal used_prefilter
        if all_trajs:
            vals = [
                v
                for tr in all_trajs
                if (v := _first_present(tr.trajectory_metrics, keys)) is not None
            ]
            if vals:
                used_prefilter = True
                return vals
        return [v for s in run.steps if (v := _first_present(s.env_metrics, keys)) is not None]

    used_vals = _collect(_USED_TOOL_KEYS)
    calls_vals = _collect(_TOOL_CALLS_KEYS)
    err_vals = _collect(_TOOL_ERR_KEYS)
    max_turns_vals = _collect(_MAX_TURNS_KEYS)

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
    source = "prefilter" if used_prefilter else "postfilter"

    # Per-step prefix-cache hit fraction + step-level tool metrics (preferring pre-filter!)
    pf_by_step = {ps.step: ps for ps in prefilter_steps}
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
        ps = pf_by_step.get(s.step)
        row: dict[str, Any] = {
            "step": s.step,
            "cache_hit_frac": round(cached / total_pre, 4) if total_pre > 0 else None,
            "used_tool": (
                ps.used_tool_rate
                if ps is not None and ps.used_tool_rate is not None
                else _first_present(s.env_metrics, _USED_TOOL_KEYS)
            ),
            "tool_calls": (
                ps.tool_calls_per_ep
                if ps is not None and ps.tool_calls_per_ep is not None
                else _first_present(s.env_metrics, _TOOL_CALLS_KEYS)
            ),
            "tool_errors": (
                ps.tool_errors_per_ep
                if ps is not None and ps.tool_errors_per_ep is not None
                else _first_present(s.env_metrics, _TOOL_ERR_KEYS)
            ),
            "hit_max_turns": (
                ps.max_turns_rate
                if ps is not None and ps.max_turns_rate is not None
                else _first_present(s.env_metrics, _MAX_TURNS_KEYS)
            ),
            "trunc_episode_rate": ps.trunc_episode_rate if ps is not None else None,
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
        source=source,
        used_tool_rate=round(used_rate, 4) if used_rate is not None else None,
        mean_tool_calls=round(mean_calls, 3) if mean_calls is not None else None,
        mean_tool_errors=round(mean_errors, 3) if mean_errors is not None else None,
        hit_max_turns_rate=round(hit_max_turns, 4) if hit_max_turns is not None else None,
        hit_max_tokens_rate=round(trunc_turn_frac, 4) if trunc_turn_frac is not None else None,
        trunc_episode_rate=round(trunc_ep_rate, 4) if trunc_ep_rate is not None else None,
        trunc_reward_mean=trunc_rew_mean,
        untrunc_reward_mean=untrunc_rew_mean,
        mean_turns_per_episode=round(mean_turns, 2) if mean_turns is not None else None,
        end_state_counts=end_state_counts,
        per_step_cache_hit=per_step_cache,
    )

    finding: DiagnosticFinding | None = None
    if has_tool_metrics:
        high_errors = mean_errors is not None and mean_errors >= 0.4
        high_max_turns = hit_max_turns is not None and hit_max_turns >= 0.25
        sev = "warn" if (high_errors or high_max_turns) else "pass"
        parts: list[str] = []
        if used_rate is not None:
            parts.append(f"{used_rate:.1%} used tools")
        if mean_calls is not None:
            parts.append(f"{mean_calls:.2f} calls/ep")
        if mean_errors is not None:
            parts.append(f"{mean_errors:.2f} errors/ep")
        if hit_max_turns is not None:
            parts.append(f"{hit_max_turns:.1%} hit max_turns")
        headline = f"({source}) " + ", ".join(parts)
        detail = (
            f"Multi-turn tool execution summary from {source} rollouts "
            f"(mean {mean_turns:.2f} turns/ep). "
            if mean_turns is not None
            else f"Multi-turn tool execution summary ({source}). "
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
                "source": source,
                "used_tool_rate": td.used_tool_rate,
                "mean_tool_calls": td.mean_tool_calls,
                "mean_tool_errors": td.mean_tool_errors,
                "hit_max_turns_rate": td.hit_max_turns_rate,
                "mean_turns_per_episode": td.mean_turns_per_episode,
            },
        )
    return td, finding


def _episode_metric_dict(ep: EvalEpisode) -> dict[str, float]:
    out = dict(ep.metrics)
    out.setdefault("reward", ep.reward)
    return out


def _config_set(episodes: list[EvalEpisode], attr: str) -> set[Any]:
    return {getattr(ep, attr) for ep in episodes if getattr(ep, attr) is not None}


def diff_evals(
    base_episodes: list[EvalEpisode],
    final_episodes: list[EvalEpisode],
    *,
    base_label: str = "base",
    final_label: str = "final",
    primary_metric: str | None = None,
    cluster_by: str | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    loader_warnings: list[str] | None = None,
) -> EvalDiffSummary:
    """Compute paired task-level deltas, 95% bootstrap CIs, end states, and transcript diffs."""
    warnings: list[str] = list(loader_warnings or [])

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

    base_unmatched = len(base_by_task) - len(shared_keys)
    final_unmatched = len(final_by_task) - len(shared_keys)
    if base_unmatched > 0 or final_unmatched > 0:
        warnings.append(
            f"Unmatched tasks ignored: {base_unmatched} {base_label}-only, "
            f"{final_unmatched} {final_label}-only ({len(shared_keys)} paired)."
        )

    # Check config mismatches across sides (max_tokens, effort, max_turns)
    for cfg_field in ("max_tokens", "effort", "max_turns"):
        b_vals_cfg = _config_set(base_episodes, cfg_field)
        f_vals_cfg = _config_set(final_episodes, cfg_field)
        if b_vals_cfg and f_vals_cfg and b_vals_cfg != f_vals_cfg:
            warnings.append(
                f"Eval config differs on {cfg_field}: {base_label}={sorted(b_vals_cfg)} vs "
                f"{final_label}={sorted(f_vals_cfg)}."
            )

    # Discover all shared metrics across paired tasks
    paired_base_eps = [ep for k in shared_keys for ep in base_by_task[k]]
    paired_final_eps = [ep for k in shared_keys for ep in final_by_task[k]]
    base_metric_keys = {k for ep in paired_base_eps for k in _episode_metric_dict(ep)}
    final_metric_keys = {k for ep in paired_final_eps for k in _episode_metric_dict(ep)}
    shared_metrics = sorted(base_metric_keys & final_metric_keys)
    ordered_metrics = [m for m in _PRIMARY_METRIC_PRIORITY if m in shared_metrics]
    ordered_metrics += [m for m in shared_metrics if m not in ordered_metrics]

    if primary_metric is not None:
        if primary_metric not in shared_metrics:
            raise ValueError(
                f"Metric {primary_metric!r} not present in both eval files. "
                f"Available shared metrics: {shared_metrics}"
            )
        chosen_primary = primary_metric
        ordered_metrics = [chosen_primary] + [m for m in ordered_metrics if m != chosen_primary]
    else:
        chosen_primary = ordered_metrics[0] if ordered_metrics else "reward"

    # Warn when pairing training rollout_summaries from different iterations or mismatched tags
    if shared_keys and all(k.startswith("group:") for k in shared_keys):
        b_iters = {
            ep.metadata.get("iteration")
            for k in shared_keys
            for ep in base_by_task[k]
            if ep.metadata.get("iteration") is not None
        }
        f_iters = {
            ep.metadata.get("iteration")
            for k in shared_keys
            for ep in final_by_task[k]
            if ep.metadata.get("iteration") is not None
        }
        tag_mismatches = 0
        for k in shared_keys:
            bt = base_by_task[k][0].metadata.get("tags")
            ft = final_by_task[k][0].metadata.get("tags")
            if bt and ft and bt != ft:
                tag_mismatches += 1
        if (b_iters and f_iters and b_iters != f_iters) or tag_mismatches > 0:
            warnings.append(
                "Paired rollout_summaries come from different iterations or have mismatched group "
                f"tags ({tag_mismatches}/{len(shared_keys)} groups differ): group_idx is per-step "
                "position, not a stable task ID across training steps."
            )

    # Validate --cluster-by upfront using shared strict extractor
    cluster_ids: list[str] | None = None
    num_clusters: int | None = None
    if cluster_by:
        rep_eps = [base_by_task[k][0] for k in shared_keys]
        cluster_ids, num_clusters = extract_cluster_ids(rep_eps, cluster_by)

    attempt_mismatch_tasks = 0
    ungraded_zeroed_base = 0
    ungraded_zeroed_final = 0
    non_outcome_skipped = 0

    tasks: list[EvalTaskDiff] = []
    for idx_k, tk in enumerate(shared_keys):
        b_eps = base_by_task[tk]
        f_eps = final_by_task[tk]
        if len(b_eps) != len(f_eps):
            attempt_mismatch_tasks += 1

        b_metrics: dict[str, float] = {}
        f_metrics: dict[str, float] = {}
        deltas: dict[str, float] = {}
        for m in ordered_metrics:
            is_out = is_outcome_metric(m)
            bv: list[float] = []
            for e in b_eps:
                em = _episode_metric_dict(e)
                if m in em:
                    bv.append(em[m])
                elif is_out:
                    bv.append(0.0)
                    if m == chosen_primary:
                        ungraded_zeroed_base += 1
                else:
                    non_outcome_skipped += 1
            fv: list[float] = []
            for e in f_eps:
                em = _episode_metric_dict(e)
                if m in em:
                    fv.append(em[m])
                elif is_out:
                    fv.append(0.0)
                    if m == chosen_primary:
                        ungraded_zeroed_final += 1
                else:
                    non_outcome_skipped += 1
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
        gold = next(
            (e.gold_answer for e in (*b_eps, *f_eps) if e.gold_answer is not None),
            None,
        )
        cid = cluster_ids[idx_k] if cluster_ids is not None else None

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
                cluster_id=cid,
                gold_answer=gold,
            )
        )

    if attempt_mismatch_tasks > 0:
        warnings.append(
            f"Attempt counts differ between {base_label} and {final_label} on "
            f"{attempt_mismatch_tasks}/{len(tasks)} paired tasks (averaged per task)."
        )
    if ungraded_zeroed_base > 0 or ungraded_zeroed_final > 0:
        warnings.append(
            f"Missing '{chosen_primary}' outcome counted as 0.0 on "
            f"{ungraded_zeroed_base} {base_label} and {ungraded_zeroed_final} "
            f"{final_label} episodes."
        )
    if non_outcome_skipped > 0:
        warnings.append(
            f"Skipped {non_outcome_skipped} missing non-outcome metric value(s) "
            "(not imputed as 0.0)."
        )

    metric_deltas: list[EvalMetricDelta] = []
    for m in ordered_metrics:
        paired_indices = [
            i for i, t in enumerate(tasks) if m in t.base_metrics and m in t.final_metrics
        ]
        b_vals = [tasks[i].base_metrics[m] for i in paired_indices]
        f_vals = [tasks[i].final_metrics[m] for i in paired_indices]
        d_vals = [f - b for b, f in zip(b_vals, f_vals, strict=True)]
        if not d_vals:
            continue
        m_cids = [cluster_ids[i] for i in paired_indices] if cluster_ids is not None else None
        ci_lo, ci_hi = bootstrap_ci_mean_diff(
            d_vals,
            n_boot=n_boot,
            seed=seed,
            cluster_ids=m_cids,
        )
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

    # End-state breakdown and per-end-state success rate (plus completed split)
    base_end_counts = Counter(ep.end_state for ep in paired_base_eps)
    final_end_counts = Counter(ep.end_state for ep in paired_final_eps)

    def _end_state_success(eps: list[EvalEpisode]) -> dict[str, tuple[float, int]]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for ep in eps:
            val = _episode_metric_dict(ep).get(chosen_primary, 0.0)
            buckets[ep.end_state].append(val)
        return {k: (round(_mean(v), 4), len(v)) for k, v in buckets.items()}

    base_es_succ = _end_state_success(paired_base_eps)
    final_es_succ = _end_state_success(paired_final_eps)

    def _completed_decomp(eps: list[EvalEpisode]) -> tuple[float | None, float | None]:
        if not eps:
            return None, None
        completed = [ep for ep in eps if ep.end_state == "completed"]
        comp_rate = len(completed) / len(eps)
        if not completed:
            return round(comp_rate, 4), None
        corr_vals = [_episode_metric_dict(ep).get(chosen_primary, 0.0) for ep in completed]
        return round(comp_rate, 4), round(_mean(corr_vals), 4)

    ans_base, corr_given_ans_base = _completed_decomp(paired_base_eps)
    ans_final, corr_given_ans_final = _completed_decomp(paired_final_eps)

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
        primary_metric=chosen_primary,
        metric_deltas=metric_deltas,
        answer_chars_change_pct=chars_pct,
        turns_change_pct=turns_pct,
        tasks=tasks,
        base_unmatched_tasks=base_unmatched,
        final_unmatched_tasks=final_unmatched,
        base_end_states=dict(base_end_counts),
        final_end_states=dict(final_end_counts),
        base_end_state_success=base_es_succ,
        final_end_state_success=final_es_succ,
        answered_base_rate=ans_base,
        answered_final_rate=ans_final,
        correct_given_answered_base=corr_given_ans_base,
        correct_given_answered_final=corr_given_ans_final,
        cluster_by=cluster_by,
        num_clusters=num_clusters,
        n_boot=n_boot,
        seed=seed,
        warnings=warnings,
    )
