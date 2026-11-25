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
    def unique_tokens(self) -> int:
        """Total unique tokens in the trajectory, falling back when final_ob_len is 0."""
        if self.final_ob_len > 0:
            return self.final_ob_len
        if self.steps:
            last = self.steps[-1]
            return last.ob_len + last.ac_len
        return 0

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
    entropy: float | None = None
    kept_episodes: int | None = None
    kept_turns: int | None = None
    turns_per_episode: float | None = None
    ac_tokens_per_turn: float | None = None
    ob_tokens_per_turn: float | None = None
    total_ac_tokens: int | None = None
    total_ob_tokens: int | None = None
    env_metrics: dict[str, float] = field(default_factory=dict)
    tag_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    native_tokens: dict[str, int] = field(default_factory=dict)
    native_discarded_tokens: dict[str, int] = field(default_factory=dict)
    native_cost_usd: float | None = None
    native_discarded_cost_usd: float | None = None
    native_cumulative_cost_usd: float | None = None
    rollout_errors: dict[str, float] = field(default_factory=dict)
    time_metrics: dict[str, float] = field(default_factory=dict)


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
    estimated_cost_usd: float | None = None
    estimated_discarded_cost_usd: float | None = None
    pricing_source: str | None = None


@dataclass
class ToolDiagnostics:
    """Multi-turn tool-use, limit-hit, and prefix-cache diagnostics."""

    used_tool_rate: float | None = None
    mean_tool_calls: float | None = None
    mean_tool_errors: float | None = None
    hit_max_turns_rate: float | None = None
    hit_max_tokens_rate: float | None = None
    mean_turns_per_episode: float | None = None
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
