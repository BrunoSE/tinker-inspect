"""Typed data structures for Tinker RL runs, diagnostics, and eval diffs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["pass", "info", "warn", "fail"]
GroupStatus = Literal["mixed", "all_good", "all_bad", "constant", "unknown"]


@dataclass
class TokenCostRates:
    """USD per million tokens matching Tinker's models.json billing columns."""

    prefill: float
    cached_prefill: float
    sample: float
    train: float
    source: str = "custom"

    def estimate_usd(
        self,
        *,
        prefill_uncached: int = 0,
        prefill_cached: int = 0,
        sampled: int = 0,
        trained: int = 0,
    ) -> float:
        return (
            prefill_uncached * self.prefill
            + prefill_cached * self.cached_prefill
            + sampled * self.sample
            + trained * self.train
        ) / 1_000_000.0


@dataclass
class TurnRecord:
    step_idx: int
    ob_len: int
    ac_len: int
    reward: float = 0.0
    episode_done: bool = False
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TrajectoryRecord:
    iteration: int
    group_idx: int
    traj_idx: int
    tags: list[str]
    total_reward: float
    final_reward: float
    trajectory_metrics: dict[str, float]
    steps: list[TurnRecord]
    final_ob_len: int
    sampling_client_step: int | None = None

    @property
    def num_turns(self) -> int:
        return len(self.steps)

    @property
    def sum_ob_len(self) -> int:
        return sum(s.ob_len for s in self.steps)

    @property
    def sum_ac_len(self) -> int:
        return sum(s.ac_len for s in self.steps)

    @property
    def last_ob_len(self) -> int:
        """Observation length on the final turn (contains all prior turns' prefix)."""
        if self.steps:
            return self.steps[-1].ob_len
        return 0

    @property
    def unique_tokens(self) -> int:
        """Unique prefill tokens (final prompt without re-counting prior turns).

        Uses ``last_ob_len`` when ``final_ob_len`` is 0 so sampled action tokens
        are never added to the prefill denominator (keeping prefill_multiplier >= 1.0).
        """
        if self.steps:
            return self.last_ob_len
        return self.final_ob_len

    @property
    def estimated_prefill_cached(self) -> int:
        """Tokens carried forward from the previous turn's prefix (ob_len + ac_len)."""
        if len(self.steps) <= 1:
            return 0
        cached = 0
        for prev, curr in zip(self.steps, self.steps[1:], strict=False):
            prior_prefix = prev.ob_len + prev.ac_len
            cached += min(curr.ob_len, prior_prefix)
        return cached

    @property
    def estimated_prefill_uncached(self) -> int:
        return max(0, self.sum_ob_len - self.estimated_prefill_cached)

    def end_state(self, max_tokens: int | None = None) -> str:
        """Classify how this trajectory ended."""
        m = self.trajectory_metrics
        if (m.get("hit_max_turns", 0.0) or m.get("max_turns_reached", 0.0)) >= 0.5:
            return "turn_limit"
        if m.get("truncated_thinking", 0.0) >= 0.5:
            return "token_limit"
        if (
            isinstance(max_tokens, int)
            and max_tokens > 0
            and any(s.ac_len >= max_tokens for s in self.steps)
        ):
            return "token_limit"
        if "parse_ok" in m and m["parse_ok"] < 0.5:
            return "parse_error"
        return "completed"


@dataclass
class GroupSummary:
    iteration: int
    group_idx: int
    tags: list[str]
    trajectories: list[TrajectoryRecord]
    status: GroupStatus
    kept_after_filter: bool
    mean_reward: float
    min_reward: float
    max_reward: float
    mean_turns: float
    sum_ob_len: int
    sum_ac_len: int
    unique_final_ob_len: int
    estimated_prefill_cached: int
    estimated_prefill_uncached: int
    trunc_episodes: int = 0
    max_turns_episodes: int = 0
    tool_errors_total: float = 0.0


@dataclass
class SpanRecord:
    name: str
    duration: float
    wall_start: float
    wall_end: float


@dataclass
class StepTiming:
    step: int
    spans: list[SpanRecord]
    rollout_durations: list[float]
    sampling_wall: float | None = None
    train_wall: float | None = None
    checkpoint_wall: float | None = None
    total_wall: float | None = None
    straggler_ratio: float | None = None  # max / mean rollout duration


@dataclass
class CheckpointRecord:
    name: str
    batch: int
    state_path: str | None = None
    sampler_path: str | None = None
    cumulative_tokens: dict[str, int] = field(default_factory=dict)


@dataclass
class StepRecord:
    step: int
    raw: dict[str, Any]
    lr: float | None = None
    done_frac: float | None = None
    reward_total: float | None = None
    kl_v1: float | None = None
    kl_v2: float | None = None
    sample_train_mismatch_frac: float | None = None
    sample_train_max_abs_diff: float | None = None
    entropy: float | None = None
    kept_episodes: int | None = None
    kept_turns: int | None = None
    turns_per_episode: float | None = None
    ac_tokens_per_turn: float | None = None
    ob_tokens_per_turn: float | None = None
    total_ac_tokens: int | None = None
    total_ob_tokens: int | None = None
    env_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    tag_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    native_tokens: dict[str, int] = field(default_factory=dict)
    native_discarded_tokens: dict[str, int] = field(default_factory=dict)
    native_cost_usd: float | None = None
    native_discarded_cost_usd: float | None = None
    native_cumulative_cost_usd: float | None = None
    rollout_errors: dict[str, float] = field(default_factory=dict)
    time_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class LogSummary:
    """Counts and sample lines of errors/warnings/retries from logs.log."""

    error_lines: int = 0
    warn_lines: int = 0
    retry_or_rate_limit_lines: int = 0
    samples: list[str] = field(default_factory=list)


@dataclass
class PrefilterStepSummary:
    """Ground-truth per-step statistics computed across ALL sampled episodes pre-filter."""

    step: int
    episodes_sampled: int
    episodes_kept: int
    mean_reward: float
    kept_mean_reward: float | None = None
    mean_turns: float = 0.0
    ac_tokens_per_turn: float = 0.0
    used_tool_rate: float | None = None
    tool_calls_per_ep: float | None = None
    tool_errors_per_ep: float | None = None
    trunc_episode_rate: float | None = None
    max_turns_rate: float | None = None
    parse_error_rate: float | None = None
    tag_rewards: dict[str, tuple[float, int]] = field(default_factory=dict)
    end_states: dict[str, int] = field(default_factory=dict)


@dataclass
class RunData:
    log_dir: str
    name: str
    config: dict[str, Any]
    steps: list[StepRecord]
    checkpoints: list[CheckpointRecord]
    timings: dict[int, StepTiming]
    groups_by_step: dict[int, list[GroupSummary]]
    summary_step: StepRecord | None = None
    discard_rows: list[StepRecord] = field(default_factory=list)
    log_summary: LogSummary | None = None
    logtree_groups: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    code_diff: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class DiagnosticFinding:
    check_id: str
    title: str
    severity: Severity
    headline: str
    detail: str
    step: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenSummary:
    source: Literal["native", "reconstructed", "metrics_only"]
    prefill_total: int = 0
    prefill_cached: int = 0
    prefill_uncached: int = 0
    sampled_total: int = 0
    trained_total: int = 0
    unique_trajectory_tokens: int = 0
    prefill_multiplier: float | None = None
    cacheable_fraction: float | None = None
    kept_prefill: int = 0
    discarded_prefill: int = 0
    kept_sampled: int = 0
    discarded_sampled: int = 0
    discarded_token_fraction: float | None = None
    discarded_sampled_only_fraction: float | None = None
    cache_metric_label: str = "Measured cache hits"
    estimated_cost_usd: float | None = None
    estimated_discarded_cost_usd: float | None = None
    pricing_source: str | None = None


@dataclass
class ToolDiagnostics:
    """Multi-turn tool-use, limit-hit, and prefix-cache diagnostics."""

    source: str = "prefilter"
    used_tool_rate: float | None = None
    mean_tool_calls: float | None = None
    mean_tool_errors: float | None = None
    hit_max_turns_rate: float | None = None
    hit_max_tokens_rate: float | None = None
    trunc_episode_rate: float | None = None
    trunc_reward_mean: float | None = None
    untrunc_reward_mean: float | None = None
    mean_turns_per_episode: float | None = None
    end_state_counts: dict[str, int] = field(default_factory=dict)
    per_step_cache_hit: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunDiagnostics:
    run_name: str
    model_name: str | None
    renderer_name: str | None
    num_steps: int
    total_wall_seconds: float
    sampling_wall_fraction: float | None
    train_wall_fraction: float | None
    checkpoint_wall_fraction: float | None
    max_abs_kl_v1: float | None
    max_abs_kl_v2: float | None
    first_nonzero_kl_step: int | None
    total_groups_sampled: int | None
    total_groups_kept: int | None
    total_groups_all_good: int | None
    total_groups_all_bad: int | None
    total_episodes_sampled: int | None
    total_episodes_kept: int | None
    discarded_group_fraction: float | None
    median_straggler_ratio: float | None
    max_straggler_ratio: float | None
    max_straggler_step: int | None
    tokens: TokenSummary
    findings: list[DiagnosticFinding]
    per_step_drop_stats: list[dict[str, Any]]
    available_metrics: list[str]
    available_tags: list[str]
    tools: ToolDiagnostics | None = None
    prefilter_steps: list[PrefilterStepSummary] = field(default_factory=list)
    reward_early_mean: float | None = None
    reward_late_mean: float | None = None
    reward_window_size: int = 0
    reward_source: str = "postfilter"
    sampling_idle_fraction: float | None = None
    policy_sample_mean_s: float | None = None
    env_step_mean_s: float | None = None
    max_sample_train_mismatch_frac: float | None = None
    max_sample_train_max_abs_diff: float | None = None
    tag_window_summary: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalMessage:
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalEpisode:
    task_key: str
    task_id: str | int | None
    prompt: str
    split: str | None
    ok: bool
    reward: float
    metrics: dict[str, float]
    num_turns: int
    input_tokens: int | None
    output_tokens: int | None
    final_answer_chars: int
    messages: list[EvalMessage]
    errors: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    parse_ok: bool | None = None
    truncated_thinking: bool | None = None
    hit_max_turns: bool | None = None
    gold_answer: str | None = None
    episode_type: str | None = None
    effort: float | None = None
    max_tokens: int | None = None
    max_turns: int | None = None
    graded: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def end_state(self) -> str:
        if not self.ok or self.errors:
            return "error"
        if not self.graded and not self.metrics:
            return "ungraded"
        if self.hit_max_turns or self.stop_reason == "max_turns":
            return "turn_limit"
        if self.truncated_thinking or self.stop_reason in ("length", "max_tokens"):
            return "token_limit"
        if self.parse_ok is False or self.stop_reason == "parse_error":
            return "parse_error"
        if not self.graded:
            return "ungraded"
        if self.stop_reason is not None and self.stop_reason not in (
            "completed",
            "stop",
            "end_turn",
            "eos",
            "",
        ):
            return f"other:{self.stop_reason}"
        return "completed"


@dataclass
class EvalMetricDelta:
    metric: str
    base_mean: float
    final_mean: float
    delta: float
    ci_low: float
    ci_high: float
    wins: int
    regressions: int
    ties: int


@dataclass
class EvalTaskDiff:
    task_key: str
    task_id: str | int | None
    prompt: str
    base_count: int
    final_count: int
    base_metrics: dict[str, float]
    final_metrics: dict[str, float]
    deltas: dict[str, float]
    base_mean_answer_chars: float
    final_mean_answer_chars: float
    answer_chars_ratio: float | None
    base_mean_turns: float
    final_mean_turns: float
    base_episodes: list[EvalEpisode]
    final_episodes: list[EvalEpisode]
    cluster_id: str | None = None
    gold_answer: str | None = None


@dataclass
class EvalDiffSummary:
    base_label: str
    final_label: str
    num_tasks: int
    base_total_episodes: int
    final_total_episodes: int
    base_failed: int
    final_failed: int
    primary_metric: str
    metric_deltas: list[EvalMetricDelta]
    answer_chars_change_pct: float | None
    turns_change_pct: float | None
    tasks: list[EvalTaskDiff]
    base_unmatched_tasks: int = 0
    final_unmatched_tasks: int = 0
    base_end_states: dict[str, int] = field(default_factory=dict)
    final_end_states: dict[str, int] = field(default_factory=dict)
    base_end_state_success: dict[str, tuple[float, int]] = field(default_factory=dict)
    final_end_state_success: dict[str, tuple[float, int]] = field(default_factory=dict)
    answered_base_rate: float | None = None
    answered_final_rate: float | None = None
    correct_given_answered_base: float | None = None
    correct_given_answered_final: float | None = None
    cluster_by: str | None = None
    num_clusters: int | None = None
    n_boot: int = 10_000
    seed: int = 42
    warnings: list[str] = field(default_factory=list)
