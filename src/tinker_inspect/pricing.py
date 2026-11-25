"""Pricing helpers compatible with Tinker's published models.json format."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping

from tinker_inspect import __version__
from tinker_inspect.schema import TokenCostRates

TINKER_PRICING_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"


def _parse_price(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip().lstrip("$").replace(",", ""))
    except ValueError:
        return None


def parse_rates_spec(spec: str) -> TokenCostRates:
    """Parse ``prefill,cached_prefill,sample,train`` USD/1M token rates."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "Expected 4 comma-separated USD/1M rates "
            f"(prefill,cached_prefill,sample,train); got {spec!r}"
        )
    vals = [_parse_price(p) for p in parts]
    if any(v is None or v < 0 for v in vals):
        raise ValueError(f"Invalid rate values in {spec!r}")
    prefill, cached, sample, train = (float(v) for v in vals)  # type: ignore[arg-type]
    return TokenCostRates(
        prefill=prefill,
        cached_prefill=cached,
        sample=sample,
        train=train,
        source="cli:--rates",
    )


def rates_from_entry(
    entry: Mapping[str, object], *, source: str = "models.json"
) -> TokenCostRates | None:
    prefill = _parse_price(entry.get("prefill"))
    cached = _parse_price(entry.get("cached_prefill"))
    sample = _parse_price(entry.get("sample"))
    train = _parse_price(entry.get("train"))
    if None in (prefill, cached, sample, train):
        return None
    return TokenCostRates(
        prefill=prefill,  # type: ignore[arg-type]
        cached_prefill=cached,  # type: ignore[arg-type]
        sample=sample,  # type: ignore[arg-type]
        train=train,  # type: ignore[arg-type]
        source=source,
    )


def fetch_tinker_pricing(
    model_name: str,
    *,
    url: str = TINKER_PRICING_URL,
    timeout: float = 5.0,
) -> TokenCostRates | None:
    """Opt-in fetch of Tinker model pricing from models.json. Never raises on network failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"tinker-inspect/{__version__}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            entries = json.load(resp)
    except Exception:
        return None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("tinker_id") == model_name:
            return rates_from_entry(entry, source="models.json (live)")
    return None
