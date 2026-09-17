"""comparison/baseline_tracker.py — Conventional fixed tracking pipeline (Baseline Method).

Pipeline:
  Beacon Acquisition → KCF Tracker → Fixed Kalman Filter → PID Controller → Camera Gimbal

Strict constraints:
  - NO XGBoost disturbance classification
  - NO Adaptive Estimator switching
  - NO UKF switching
  - NO Optical Flow fusion
  - NO AI regime-based tuning
"""
from __future__ import annotations
import time
import cv2
import numpy as np
from config import CFG
from vision.acquisition import acquire_beacon
from vision.centroid import compute_centroid, compute_snr
from vision.kcf_tracker import KCFTracker
from estimation.kalman_filter import KalmanFilter
from control.pid_controller import PIDController
from comparison.result_model import FrameRecord


class BaselineFixedTracker:
    """Conventional tracking pipeline with fixed KCF + Linear Kalman Filter."""

    def __init__(self):
        self.kcf = KCFTracker()
        self.kf = KalmanFilter()
        self.pid = PIDController()
        self.acq_done = False
        self.state_x = CFG.camera.width / 2.0
        self.state_y = CFG.camera.height / 2.0
        self.predicted_x = self.state_x
        self.predicted_y = self.state_y
        self.pointing_error = 0.0
        self.lock = False
        self.detected = False
        self.frame_idx = 0

    def reset(self):
        self.kcf.reset()
        self.kf.reset()
        self.pid.reset()
        self.acq_done = False
        self.state_x = CFG.camera.width / 2.0
        self.state_y = CFG.camera.height / 2.0
        self.predicted_x = self.state_x
        self.predicted_y = self.state_y
        self.pointing_error = 0.0
        self.lock = False
        self.detected = False
        self.frame_idx = 0

    def process_frame(self, frame: np.ndarray, dt: float,
                      gt_x: float, gt_y: float,
                      generate_overlay: bool = True) -> tuple[FrameRecord, tuple[float, float]]:
        """
        Executes one frame of baseline tracking.
        Returns: (FrameRecord, (pan_rate_cmd, tilt_rate_cmd))
        """
        t0 = time.perf_counter()
        self.frame_idx += 1
        t_sec = self.frame_idx * dt
        W, H = CFG.camera.width, CFG.camera.height
        cx0, cy0 = W // 2, H // 2

        # 1. Initial Acquisition
        if not self.acq_done:
            cands = acquire_beacon(frame)
            if cands:
                cx, cy, _ = cands[0]
                xc, yc, _ = compute_centroid(frame, cx, cy)
                self.kcf.reinitialise(frame, int(xc), int(yc))
                self.kf.initialise(xc, yc)
                self.acq_done = True
                self.detected = True
                self.state_x, self.state_y = xc, yc
                self.predicted_x, self.predicted_y = xc, yc
            else:
                self.detected = False
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
                    estimator_name="Fixed Kalman Filter",
                    regime="FIXED",
                    xgb_conf=0.0,
                    uncertainty=self.kf.uncertainty,
                    nis=0.0,
                    latency_ms=lat_ms,
                    fps=1.0 / max(dt, 0.001),
                    preview_frame=preview,
                )
                return rec, (0.0, 0.0)

        # 2. Tracking with KCF
        kx, ky, kc = self.kcf.update(frame)
        if not self.kcf.is_initialised or kc < 0.05:
            # Lost tracking in baseline — cannot adapt, tries simple re-acquisition next frame
            self.detected = False
            self.acq_done = False
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
                estimator_name="Fixed Kalman Filter",
                regime="FIXED",
                xgb_conf=0.0,
                uncertainty=self.kf.uncertainty,
                nis=0.0,
                latency_ms=lat_ms,
                fps=1.0 / max(dt, 0.001),
                preview_frame=preview,
            )
            return rec, (0.0, 0.0)

        # 3. Centroid computation & SNR
        xc, yc, _ = compute_centroid(frame, int(kx), int(ky))
        snr = compute_snr(frame, int(xc), int(yc))

        # 4. Fixed Linear Kalman Filter Predict & Update (No adaptive switching, No optical flow)
        self.kf.predict()
        self.kf.update(np.array([xc, yc]))
        sx, sy, svx, svy = self.kf.state
        self.state_x, self.state_y = sx, sy
        self.predicted_x = sx + svx * dt
        self.predicted_y = sy + svy * dt

        # 5. Pointing Error & Closed-Loop PID
        err = float(np.hypot(sx - cx0, sy - cy0))
        self.pointing_error = err
        self.detected = True
        self.lock = (kc > 0.08 and err < 90.0)

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
            pointing_error=err,
            locked=self.lock,
            detected=True,
            estimator_name="Fixed Kalman Filter",
            regime="FIXED",
            xgb_conf=0.0,
            uncertainty=self.kf.uncertainty,
            nis=0.0,
            latency_ms=lat_ms,
            fps=1.0 / max(dt, 0.001),
            preview_frame=preview,
        )
        return rec, (pan_cmd, tilt_cmd)

    def _draw_overlay(self, frame: np.ndarray, gt_x: float, gt_y: float) -> np.ndarray:
        W, H = CFG.camera.width, CFG.camera.height
        cx0, cy0 = W // 2, H // 2
        out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Baseline Identifier Badge (Amber/Gray Theme)
        cv2.rectangle(out, (10, 10), (220, 36), (15, 23, 42), -1)
        cv2.rectangle(out, (10, 10), (220, 36), (100, 116, 139), 1)
        cv2.putText(out, "CONVENTIONAL (FIXED)", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1, cv2.LINE_AA)

        # Reticle Center
        cv2.drawMarker(out, (cx0, cy0), (148, 163, 184), cv2.MARKER_CROSS, 16, 1)

        # Tracked / Estimated Position
        if self.detected:
            px = int(np.clip(self.state_x, 0, W - 1))
            py = int(np.clip(self.state_y, 0, H - 1))
            col = (16, 185, 129) if self.lock else (245, 158, 11)
            cv2.circle(out, (px, py), 12, col, 1)
            cv2.drawMarker(out, (px, py), col, cv2.MARKER_CROSS, 10, 1)
            cv2.putText(out, "Fixed KF", (px + 14, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            cv2.line(out, (cx0, cy0), (px, py), (70, 70, 90), 1)
        else:
            cv2.putText(out, "TARGET LOST", (cx0 - 45, cy0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (40, 40, 245), 1, cv2.LINE_AA)

        # Fast downscale for smooth playback without high memory consumption
        return cv2.resize(out, (320, 180), interpolation=cv2.INTER_NEAREST)

