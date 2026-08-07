"""Configuration. Everything is env-overridable so the same code runs against
a laptop mock backend, a local NIM container, or build.nvidia.com.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class NIMSettings:
    """Points at any OpenAI-compatible endpoint: NIM container or NVIDIA API."""

    base_url: str = field(default_factory=lambda: os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    api_key: str | None = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY"))
    vlm_model: str = field(default_factory=lambda: os.getenv("NIM_VLM_MODEL", "nvidia/vila"))
    llm_model: str = field(default_factory=lambda: os.getenv("NIM_LLM_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct"))
    embed_model: str = field(default_factory=lambda: os.getenv("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5"))
    timeout_s: float = field(default_factory=lambda: _env_float("NIM_TIMEOUT_S", 60.0))
    max_retries: int = 3

    @property
    def enabled(self) -> bool:
        """True only when we have a key AND the user has not forced mock mode."""
        if _env_bool("AWVI_FORCE_MOCK", False):
            return False
        return bool(self.api_key)


@dataclass
class PerceptionSettings:
    backend: str = field(default_factory=lambda: os.getenv("AWVI_PERCEPTION_BACKEND", "auto"))
    tracker: str = "nvdcf"
    detector_config: str = "configs/deepstream/pgie_peoplenet.txt"
    min_detection_confidence: float = 0.35
    frame_stride: int = 3
    max_tracks: int = 512


@dataclass
class EscalationSettings:
    """Thresholds for the confidence-based escalation policy.

    These are the knobs that moved the false-positive rate; see
    `eval/ablations.py` for the sweep that produced the defaults.
    """

    suppress_below: float = field(default_factory=lambda: _env_float("AWVI_SUPPRESS_BELOW", 0.36))
    escalate_above: float = field(default_factory=lambda: _env_float("AWVI_ESCALATE_ABOVE", 0.68))
    page_above: float = field(default_factory=lambda: _env_float("AWVI_PAGE_ABOVE", 0.86))
    min_temporal_persistence_s: float = 2.5
    require_vlm_agreement: bool = True
    veto_penalty: float = 0.55
    contradiction_penalty: float = 0.18
    persistence_bonus: float = 0.12
    multi_track_bonus: float = 0.06


@dataclass
class Settings:
    nim: NIMSettings = field(default_factory=NIMSettings)
    perception: PerceptionSettings = field(default_factory=PerceptionSettings)
    escalation: EscalationSettings = field(default_factory=EscalationSettings)
    clip_pre_roll_s: float = 4.0
    clip_post_roll_s: float = 4.0
    keyframes_per_event: int = 4
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("AWVI_DATA_DIR", "data")))
    seed: int = 1337

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        s = cls()
        path = Path(path) if path else CONFIG_DIR / "pipeline.yaml"
        if path.exists():
            raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
            for section, values in raw.items():
                target = getattr(s, section, None)
                if target is None or not isinstance(values, dict):
                    continue
                for k, v in values.items():
                    if hasattr(target, k):
                        setattr(target, k, v)
        return s


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.load()
    return _SETTINGS


def reset_settings() -> None:
    """Test hook."""
    global _SETTINGS
    _SETTINGS = None
