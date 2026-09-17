"""comparison/comparison_runner.py — Orchestrates sequential execution of Baseline vs Proposed methods."""
from __future__ import annotations
import time
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from config import CFG
from simulation.environment import SimulationEnvironment
from comparison.scenario_recorder import ScenarioConfig
from comparison.baseline_tracker import BaselineFixedTracker
from comparison.comparison_metrics import build_comparison_result
from comparison.result_model import FrameRecord, ComparisonResult, DecisionEvent

from vision.acquisition import acquire_beacon
from vision.centroid import compute_centroid, compute_snr
from vision.kcf_tracker import KCFTracker
from vision.optical_flow import OpticalFlowTracker
from ai.xgboost_classifier import DisturbanceClassifier, FeatureVector
from estimation.estimator_manager import AdaptiveEstimator
from validation.nis import NISValidator
from recovery.reacquisition import HierarchicalReacquisition
from control.pid_controller import PIDController


class ProposedAdaptiveTracker:
    """Proposed AI Adaptive Tracking Pipeline."""

    def __init__(self):
        self.kcf = KCFTracker()
        self.of = OpticalFlowTracker()
        self.clf = DisturbanceClassifier()
        self.est = AdaptiveEstimator()
        self.nis = NISValidator()
        self.reacq = HierarchicalReacquisition()
        self.pid = PIDController()

        self.acq_done = False
        self.state_x = CFG.camera.width / 2.0
        self.state_y = CFG.camera.height / 2.0
        self.state_vx = 0.0
        self.state_vy = 0.0
        self.predicted_x = self.state_x
        self.predicted_y = self.state_y
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.pointing_error = 0.0
        self.lock = False
        self.detected = False
        self.loss_dur = 0.0
        self.frame_idx = 0
        self.current_regime = "NORMAL"
        self.xgb_conf = 0.95

    def reset(self):
        self.kcf.reset()
        self.of.reset()
        self.est.reset()
        self.reacq.reset()
        self.pid.reset()
        self.acq_done = False
        self.state_x = CFG.camera.width / 2.0
        self.state_y = CFG.camera.height / 2.0
        self.state_vx = self.state_vy = 0.0
        self.predicted_x = self.state_x
        self.predicted_y = self.state_y
        self.prev_vx = self.prev_vy = 0.0
        self.pointing_error = 0.0
        self.lock = False
        self.detected = False
        self.loss_dur = 0.0
        self.frame_idx = 0
        self.current_regime = "NORMAL"
        self.xgb_conf = 0.95

    def process_frame(self, frame: np.ndarray, dt: float,
                      gt_x: float, gt_y: float,
                      generate_overlay: bool = True) -> tuple[FrameRecord, tuple[float, float], DecisionEvent | None]:
        """
        Executes one frame of proposed AI adaptive tracking.
        Returns: (FrameRecord, (pan_cmd, tilt_cmd), transition_event_or_None)
        """
        t0 = time.perf_counter()
        self.frame_idx += 1
        t_sec = self.frame_idx * dt
        W, H = CFG.camera.width, CFG.camera.height
        cx0, cy0 = W // 2, H // 2
        decision_event: DecisionEvent | None = None

        # 1. Initial Acquisition & Hierarchical Reacquisition
        if not self.acq_done:
            cands = acquire_beacon(frame)
            if cands:
                cx, cy, _ = cands[0]
                xc, yc, _ = compute_centroid(frame, cx, cy)
                self.kcf.reinitialise(frame, int(xc), int(yc))
                self.of.initialise(frame, int(xc), int(yc))
                self.est.initialise(xc, yc)
                self.acq_done = True
                self.detected = True
                self.reacq.reset()
                self.state_x, self.state_y = xc, yc
                self.predicted_x, self.predicted_y = xc, yc
            else:
                reacq_found = False
                if self.reacq.is_active:
                    res = self.reacq.attempt(frame, self.predicted_x, self.predicted_y)
                    if res:
                        cx, cy = res
                        xc, yc, _ = compute_centroid(frame, cx, cy)
                        self.kcf.reinitialise(frame, int(xc), int(yc))
                        self.of.initialise(frame, int(xc), int(yc))
                        self.est.initialise(xc, yc)
                        self.acq_done = True
                        self.detected = True
                        self.reacq.reset()
                        self.state_x, self.state_y = xc, yc
                        self.predicted_x, self.predicted_y = xc, yc
                        reacq_found = True
                else:
                    self.reacq.start(self.frame_idx)

                if not reacq_found:
                    self.detected = False
                    self.loss_dur += dt
                    self.lock = False
                    err = float(np.hypot(gt_x - cx0, gt_y - cy0))
                    self.pointing_error = err
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    preview = self._draw_overlay(frame, gt_x, gt_y) if generate_overlay else None

                    rec = FrameRecord(
                        frame_idx=self.frame_idx,
                        t_sec=t_sec,
                        ground_truth_x=gt_x,
                        ground_truth_y=gt_y,
                        estimated_x=self.predicted_x,
                        estimated_y=self.predicted_y,
                        pointing_error=err,
                        locked=False,
                        detected=False,
                        estimator_name="UKF (Reacquisition)" if self.reacq.is_active else self.est.active_name,
                        regime="SEVERE" if self.reacq.is_active else self.current_regime,
                        xgb_conf=self.xgb_conf,
                        uncertainty=self.est.uncertainty,
                        nis=0.0,
                        latency_ms=lat_ms,
                        fps=1.0 / max(dt, 0.001),
                        preview_frame=preview,
                    )
                    return rec, (0.0, 0.0), None

        # 2. Update KCF
        kx, ky, kc = self.kcf.update(frame)
        if not self.kcf.is_initialised or kc < 0.04:
            # 1-Frame Instant Reacquisition from UKF predicted state before declaring loss
            reacq_pt = self.reacq.attempt(frame, self.predicted_x, self.predicted_y)
            if reacq_pt is not None:
                rx, ry = reacq_pt
                self.kcf.reinitialise(frame, int(rx), int(ry))
                self.of.initialise(frame, int(rx), int(ry))
                self.est.initialise(rx, ry)
                kx, ky, kc = rx, ry, 0.80
                self.acq_done = True
                self.detected = True
                self.reacq.reset()
            else:
                self.detected = False
                self.acq_done = False
                self.reacq.start(self.frame_idx)
                self.loss_dur += dt
                self.lock = False
                err = float(np.hypot(gt_x - cx0, gt_y - cy0))
                self.pointing_error = err
                lat_ms = (time.perf_counter() - t0) * 1000.0
                preview = self._draw_overlay(frame, gt_x, gt_y) if generate_overlay else None

                rec = FrameRecord(
                    frame_idx=self.frame_idx,
                    t_sec=t_sec,
                    ground_truth_x=gt_x,
                    ground_truth_y=gt_y,
                    estimated_x=self.predicted_x,
                    estimated_y=self.predicted_y,
                    pointing_error=err,
                    locked=False,
                    detected=False,
                    estimator_name="Hierarchical Reacq",
                    regime="SEVERE",
                    xgb_conf=self.xgb_conf,
                    uncertainty=self.est.uncertainty,
                    nis=0.0,
                    latency_ms=lat_ms,
                    fps=1.0 / max(dt, 0.001),
                    preview_frame=preview,
                )
                return rec, (0.0, 0.0), None

        # 3. Sub-pixel Centroid refinement
        xc, yc, snr = compute_centroid(frame, int(kx), int(ky))
        of_dx, of_dy, of_mag, _ = self.of.update(frame)

        # 4. Feature Extraction & XGBoost Classification
        fv = FeatureVector(
            velocity=float(np.hypot(self.state_vx, self.state_vy)),
            acceleration=float(np.hypot(self.state_vx - self.prev_vx, self.state_vy - self.prev_vy)) / max(dt, 1e-4),
            tracking_confidence=float(kc),
            pointing_error=self.pointing_error,
            optical_flow_mag=of_mag,
            beacon_intensity=float(frame[int(np.clip(yc, 0, H-1)), int(np.clip(xc, 0, W-1))]) if frame.ndim == 2 else 200.0,
            snr=float(snr),
            loss_duration=self.loss_dur,
            prediction_uncertainty=self.est.uncertainty,
            nis=0.0,
        )
        regime, _, xconf = self.clf.classify(fv)
        self.xgb_conf = xconf

        # Detect regime transition event
        if regime != self.current_regime:
            old_r = self.current_regime
            self.current_regime = regime
            self.est.set_regime(regime)
            decision_event = DecisionEvent(
                timestamp_sec=t_sec,
                frame_idx=self.frame_idx,
                regime=regime,
                estimator=self.est.active_name,
                xgb_conf=xconf,
                reason=f"Transition from {old_r} → {regime} (SNR: {snr:.1f}dB, OF: {of_mag:.1f}px)"
            )
        else:
            self.est.set_regime(regime)

        # 5. Adaptive State Estimation
        z = np.array([xc, yc])
        self.est.predict()
        self.est.update(z, of_dx, of_dy)
        sx, sy, svx, svy = self.est.state
        self.state_x, self.state_y = sx, sy
        self.state_vx, self.state_vy = svx, svy
        self.prev_vx, self.prev_vy = svx, svy
        self.predicted_x = sx + svx * dt
        self.predicted_y = sy + svy * dt

        # Innovation validation
        try:
            r, S = self.est.get_innovation(z)
            nis_val, _, _ = self.nis.validate(r, S)
        except Exception:
            nis_val = 0.0

        # 6. Pointing Error & Closed-Loop Control
        est_err = float(np.hypot(sx - cx0, sy - cy0))
        self.pointing_error = est_err
        self.detected = True
        self.loss_dur = 0.0
        self.lock = (kc > 0.04 and est_err < 120.0)

        ex = sx - cx0
        ey = sy - cy0
        pan_cmd, tilt_cmd = self.pid.update(ex, ey, dt, svx, svy)

        lat_ms = (time.perf_counter() - t0) * 1000.0
        preview = self._draw_overlay(frame, gt_x, gt_y) if generate_overlay else None

        rec = FrameRecord(
            frame_idx=self.frame_idx,
            t_sec=t_sec,
            ground_truth_x=gt_x,
            ground_truth_y=gt_y,
            estimated_x=sx,
            estimated_y=sy,
            pointing_error=est_err,
            locked=self.lock,
            detected=True,
            estimator_name=self.est.active_name,
            regime=self.current_regime,
            xgb_conf=xconf,
            uncertainty=self.est.uncertainty,
            nis=nis_val,
            latency_ms=lat_ms,
            fps=1.0 / max(dt, 0.001),
            preview_frame=preview,
        )
        return rec, (pan_cmd, tilt_cmd), decision_event

    def _draw_overlay(self, frame: np.ndarray, gt_x: float, gt_y: float) -> np.ndarray:
        W, H = CFG.camera.width, CFG.camera.height
        cx0, cy0 = W // 2, H // 2
        out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Proposed Method Identifier Badge (Cyan Theme)
        cv2.rectangle(out, (10, 10), (220, 36), (15, 23, 42), -1)
        cv2.rectangle(out, (10, 10), (220, 36), (0, 240, 255), 1)
        cv2.putText(out, "PROPOSED (AI ADAPTIVE)", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)

        # Reticle Center
        cv2.drawMarker(out, (cx0, cy0), (16, 185, 129), cv2.MARKER_CROSS, 16, 1)

        # Regime & Estimator Tag
        reg_col = (16, 185, 129) if self.current_regime == "NORMAL" else \
                  ((255, 240, 0) if self.current_regime == "MILD" else \
                  ((0, 165, 255) if self.current_regime == "MODERATE" else (40, 40, 245)))

        tag_txt = f"{self.current_regime} | {self.est.active_name}"
        cv2.rectangle(out, (10, 42), (240, 62), (10, 15, 25), -1)
        cv2.putText(out, tag_txt, (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.36, reg_col, 1, cv2.LINE_AA)

        if self.detected:
            px = int(np.clip(self.state_x, 0, W - 1))
            py = int(np.clip(self.state_y, 0, H - 1))
            col = (16, 185, 129) if self.lock else (245, 158, 11)
            cv2.circle(out, (px, py), 12, col, 1)
            cv2.drawMarker(out, (px, py), col, cv2.MARKER_CROSS, 10, 1)
            cv2.putText(out, self.est.active_name, (px + 14, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            cv2.line(out, (cx0, cy0), (px, py), (40, 90, 40), 1)
        else:
            cv2.putText(out, "REACQUIRING...", (cx0 - 55, cy0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (40, 40, 245), 1, cv2.LINE_AA)

        return cv2.resize(out, (320, 180), interpolation=cv2.INTER_NEAREST)


class ComparisonWorker(QThread):
    """Background worker thread executing both methods deterministically."""

    progress = Signal(int, str)                # (percentage, status_text)
    finished_comparison = Signal(object)       # ComparisonResult

    def __init__(self, scenario: str = "SEVERE", duration_sec: float = 10.0, seed: int = 42):
        super().__init__()
        self.scenario = scenario
        self.duration_sec = duration_sec
        self.seed = seed
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        print(f"[COMPARISON] Starting Comparison Run: Scenario={self.scenario}, Duration={self.duration_sec}s, Seed={self.seed}")
        total_frames = int(self.duration_sec * CFG.camera.fps)
        dt = 1.0 / CFG.camera.fps

        # -------------------------------------------------------------
        # 1. RUN BASELINE (EXISTING METHOD)
        # -------------------------------------------------------------
        self.progress.emit(0, "Initializing Conventional Baseline Method...")
        env = SimulationEnvironment()
        ScenarioConfig.seed_environment(env, self.seed)
        baseline_tracker = BaselineFixedTracker()
        baseline_frames: list[FrameRecord] = []

        for f_idx in range(total_frames):
            if self._is_cancelled:
                return

            t_sec = f_idx * dt
            ScenarioConfig.apply_scenario(env, self.scenario, elapsed_sec=t_sec, total_sec=self.duration_sec)
            env.step(dt)
            frame = env.render()
            gt_x, gt_y = env.beacon_px, env.beacon_py

            rec, (pan_cmd, tilt_cmd) = baseline_tracker.process_frame(
                frame, dt, gt_x, gt_y, generate_overlay=True
            )
            baseline_frames.append(rec)

            if baseline_tracker.acq_done:
                env.apply_pan_tilt_rate(pan_cmd, tilt_cmd, dt)

            pct = int((f_idx / total_frames) * 50.0)
            if f_idx % 5 == 0 or f_idx == total_frames - 1:
                self.progress.emit(pct, f"Running Conventional Method... ({t_sec:.1f}s / {self.duration_sec:.0f}s)")


        # -------------------------------------------------------------
        # 2. RUN PROPOSED METHOD (AI ADAPTIVE) UNDER EXACT SAME CONDITIONS
        # -------------------------------------------------------------
        self.progress.emit(50, "Resetting Environment with Identical Seed for Proposed Method...")
        env.reset()
        ScenarioConfig.seed_environment(env, self.seed)
        proposed_tracker = ProposedAdaptiveTracker()
        proposed_frames: list[FrameRecord] = []
        decision_timeline: list[DecisionEvent] = []

        for f_idx in range(total_frames):
            if self._is_cancelled:
                return

            t_sec = f_idx * dt
            ScenarioConfig.apply_scenario(env, self.scenario, elapsed_sec=t_sec, total_sec=self.duration_sec)
            env.step(dt)
            frame = env.render()
            gt_x, gt_y = env.beacon_px, env.beacon_py

            rec, (pan_cmd, tilt_cmd), dec_ev = proposed_tracker.process_frame(
                frame, dt, gt_x, gt_y, generate_overlay=True
            )
            proposed_frames.append(rec)
            if dec_ev is not None:
                decision_timeline.append(dec_ev)

            if proposed_tracker.acq_done:
                env.apply_pan_tilt_rate(pan_cmd, tilt_cmd, dt)

            pct = 50 + int((f_idx / total_frames) * 50.0)
            if f_idx % 5 == 0 or f_idx == total_frames - 1:
                self.progress.emit(pct, f"Running AI Adaptive Method... ({t_sec:.1f}s / {self.duration_sec:.0f}s)")


        # -------------------------------------------------------------
        # 3. COMPILE COMPARATIVE METRICS & EMIT RESULT
        # -------------------------------------------------------------
        self.progress.emit(100, "Compiling Comparative Analysis & Generating Charts...")
        result = build_comparison_result(
            scenario_name=self.scenario,
            duration_sec=self.duration_sec,
            baseline_frames=baseline_frames,
            proposed_frames=proposed_frames,
            decision_timeline=decision_timeline
        )
        print("[COMPARISON] Run successfully finished and compiled.")
        self.finished_comparison.emit(result)
