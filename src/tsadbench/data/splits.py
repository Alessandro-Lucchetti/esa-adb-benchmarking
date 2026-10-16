"""Dataset split abstraction.

A `Split` pins a (mission, signal_mode, phase, channels_mode) selection plus the SMED
hyperparameters, resolves the channel list, and provides typed access to channel
telemetry and ground-truth events. It is the single place new detectors (DAMP, etc.)
go to for data, so they don't re-implement the path/loading conventions.

The SMED core keeps its own internal loading (frozen for bit-exactness); the SMED
adapter only uses Split for the config dict + channel list.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

from tsadbench.io.misc import parse_channels_mode, load_gt_events, project_root
from tsadbench.runner.config import RunConfig


@dataclass
class Split:
    config: RunConfig

    @classmethod
    def from_dict(cls, d: dict) -> "Split":
        return cls(RunConfig(**d))

    @property
    def run_config_dict(self) -> dict:
        return self.config.to_run_dict()

    @cached_property
    def channels(self) -> list[str]:
        return parse_channels_mode(self.run_config_dict, "channel_41")

    # --- data access (for non-SMED detectors) ---
    @property
    def data_dir(self) -> Path:
        return project_root() / "data" / self.config.mission

    def channel_path(self, channel: str) -> Path:
        return self.data_dir / self.config.signal_mode / f"{channel}.parquet"

    def load_channel(self, channel: str) -> pd.DataFrame:
        """Full (timestamp, value) series for a channel, float32 preserved."""
        return pd.read_parquet(self.channel_path(channel))

    def gt_events(self, channels: list[str] | None = None) -> pd.DataFrame:
        """Ground-truth anomaly events (StartTime, EndTime, ...) for the given channels."""
        return load_gt_events(self.data_dir / "labels.csv", channels or self.channels)
