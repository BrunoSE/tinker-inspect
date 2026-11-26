"""Shared deterministic bootstrap statistics and cluster extraction for eval diffs & series."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

DEFAULT_N_BOOT: int = 10_000
DEFAULT_SEED: int = 42
MIN_CLUSTERS_DEFAULT: int = 5


def extract_cluster_ids(
    task_episodes: Sequence[Any],
    cluster_by: str,
    *,
    min_clusters: int = MIN_CLUSTERS_DEFAULT,
) -> tuple[list[str], int]:
    """Extract and validate cluster IDs for one representative episode per paired task.

    Raises ``ValueError`` if ``cluster_by`` is missing from rows or produces fewer
    than ``min_clusters`` distinct clusters (preventing silent zero-width intervals).
    """
    cids: list[str] = []
    missing_count = 0
    available_fields: set[str] = set()

    for ep in task_episodes:
        meta = getattr(ep, "metadata", None) or {}
        if isinstance(meta, dict):
            available_fields.update(meta.keys())
        for attr in ("episode_type", "split", "stop_reason", "effort", "max_tokens", "max_turns"):
            if getattr(ep, attr, None) is not None:
                available_fields.add(attr)

        val: Any = None
        if isinstance(meta, dict) and cluster_by in meta and meta[cluster_by] is not None:
            val = meta[cluster_by]
        elif hasattr(ep, cluster_by):
            cand = getattr(ep, cluster_by)
            if isinstance(cand, (str, int, float, bool)):
                val = cand
        elif isinstance(getattr(ep, "metrics", None), dict) and cluster_by in ep.metrics:
            val = ep.metrics[cluster_by]

        if val is None or val == "":
            missing_count += 1
        else:
            cids.append(str(val))

    if missing_count > 0:
        avail_str = ", ".join(sorted(available_fields)[:15]) or "none"
        raise ValueError(
            f"--cluster-by {cluster_by!r} was missing on {missing_count}/{len(task_episodes)} "
            f"paired tasks. Available scalar fields: {avail_str}."
        )

    distinct = len(set(cids))
    if distinct < min_clusters:
        raise ValueError(
            f"--cluster-by {cluster_by!r} produced only {distinct} distinct cluster(s) across "
            f"{len(task_episodes)} tasks (minimum {min_clusters} required for cluster bootstrap). "
            "Choose a higher-cardinality grouping field or omit --cluster-by."
        )
    return cids, distinct


def bootstrap_ci_mean_diff(
    diffs: Sequence[float],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    cluster_ids: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Compute a deterministic 95% percentile bootstrap CI for the mean of ``diffs``.

    When ``cluster_ids`` is supplied, resamples clusters (e.g. task patterns or
    families) with replacement and averages all task deltas inside the sampled
    clusters, avoiding artificially narrow intervals when tasks share templates.
    Raises ``ValueError`` if fewer than 2 distinct clusters are supplied.
    """
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(diffs[0]), float(diffs[0]))

    rng = random.Random(seed)
    samples: list[float] = []

    if cluster_ids is not None:
        if len(cluster_ids) != n:
            raise ValueError(
                f"cluster_ids length ({len(cluster_ids)}) must match diffs length ({n})."
            )
        clusters_map: dict[str, list[float]] = defaultdict(list)
        for cid, d in zip(cluster_ids, diffs, strict=True):
            clusters_map[str(cid)].append(float(d))
        clusters = [clusters_map[k] for k in sorted(clusters_map)]
        c_count = len(clusters)
        if c_count < 2:
            raise ValueError(
                f"Cluster bootstrap requires at least 2 distinct clusters, got {c_count}."
            )
        cluster_sums = [sum(vals) for vals in clusters]
        cluster_lens = [len(vals) for vals in clusters]
        for _ in range(n_boot):
            tot = 0.0
            cnt = 0
            for _ in range(c_count):
                idx = rng.randrange(c_count)
                tot += cluster_sums[idx]
                cnt += cluster_lens[idx]
            samples.append(tot / cnt if cnt > 0 else 0.0)
    else:
        vals = [float(d) for d in diffs]
        for _ in range(n_boot):
            tot = 0.0
            for _ in range(n):
                tot += vals[rng.randrange(n)]
            samples.append(tot / n)

    samples.sort()
    lo_idx = max(0, int(0.025 * n_boot))
    hi_idx = min(n_boot - 1, max(0, int(0.975 * n_boot) - 1))
    return (samples[lo_idx], samples[hi_idx])
