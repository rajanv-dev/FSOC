"""comparison/result_model.py — Data structures for storing and analyzing comparison runs."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FrameRecord:
    frame_idx: int
    t_sec: float
    ground_truth_x: float
    ground_truth_y: float
    estimated_x: float
    estimated_y: float
    pointing_error: float
    locked: bool
    detected: bool
    estimator_name: str
    regime: str
    xgb_conf: float
    uncertainty: float
    nis: float
    latency_ms: float
    fps: float
    preview_frame: np.ndarray | None = None  # RGB/BGR visualization snapshot for playback


@dataclass
class MethodSummary:
    rmse: float = 0.0
    mean_pointing_error: float = 0.0
    max_pointing_error: float = 0.0
    lock_percentage: float = 0.0
    target_loss_rate: float = 0.0
    avg_reacquisition_time: float | None = None
    avg_fps: float = 0.0
    avg_latency_ms: float = 0.0
    total_frames: int = 0
    lost_frames: int = 0
    lock_frames: int = 0


@dataclass
class MethodRunResult:
    method_name: str
    summary: MethodSummary = field(default_factory=MethodSummary)
    frames: list[FrameRecord] = field(default_factory=list)


@dataclass
class DecisionEvent:
    timestamp_sec: float
    frame_idx: int
    regime: str
    estimator: str
    xgb_conf: float
    reason: str


@dataclass
class ComparisonResult:
    scenario_name: str
    duration_sec: float
    timestamp_str: str
    baseline: MethodRunResult
    proposed: MethodRunResult
    improvements: dict[str, dict] = field(default_factory=dict)
    decision_timeline: list[DecisionEvent] = field(default_factory=list)
