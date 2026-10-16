"""Canonical run-name encode/decode.

Single source of truth for the prediction folder-name scheme so detectors,
sweeps, and analysis tools never re-implement (or drift from) it.

    <algo>_<mission>_phase_<phase>_<signal_mode>_<channels_label>_q<q>_r<r>_s<s>_k<k>_p<p>

`channels_label` is the literal mode ("light"/"full") except for "manual"/"dev"
runs, where it is the hyphen-joined channel list (e.g. "channel_41-channel_42").
"""
from __future__ import annotations

import re

_PATTERN = re.compile(
    r"^(?P<algo>[^_]+)_(?P<mission>[^_]+)_phase_(?P<phase>\d+)_"
    r"(?P<signal_mode>raw|resampled)_(?P<channels_label>.+)_"
    r"q(?P<query_length>\d+)_r(?P<reference_length>\d+)_s(?P<slide_length>\d+)_"
    r"k(?P<k>[\d.]+)_p(?P<p>[\d.]+)$"
)


def channels_label(channels_mode: str, channels: list[str]) -> str:
    if channels_mode in ("manual", "dev"):
        return "-".join(channels)
    return channels_mode


def build_run_name(config: dict, channels: list[str]) -> str:
    label = channels_label(config["channels_mode"], channels)
    return (
        f"{config['algo_name']}_{config['mission']}_phase_{config['phase']}_"
        f"{config['signal_mode']}_{label}_"
        f"q{config['query_length']}_r{config['reference_length']}_"
        f"s{config['slide_length']}_k{config['k']}_p{config['p']}"
    )


def parse_run_name(name: str) -> dict:
    """Inverse of build_run_name. Numeric fields are returned as ints/floats.

    Note: `channels_label` is returned as-is; for manual/dev runs split on '-'
    to recover the channel list.
    """
    m = _PATTERN.match(name)
    if not m:
        raise ValueError(f"not a recognized run name: {name!r}")
    d = m.groupdict()
    for key in ("query_length", "reference_length", "slide_length"):
        d[key] = int(d[key])
    for key in ("k", "p"):
        d[key] = float(d[key])
    return d
