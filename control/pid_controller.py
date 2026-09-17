"""control/pid_controller.py — Dual-axis PID for pan/tilt pointing control.

Error is in pixels (beacon_x - frame_cx, beacon_y - frame_cy).
Output is in deg/sec pan/tilt rate commands passed to camera.apply_pan_tilt_rate().
"""
import numpy as np
from config import CFG

class PIDController:
    def __init__(self):
        cfg = CFG.pid
        self._kp = cfg.kp
        self._ki = cfg.ki
        self._kd = cfg.kd
        self._k_ff = getattr(cfg, "k_ff", 0.60)
        self._int_x = self._int_y = 0.0
        self._prev_ex = self._prev_ey = 0.0
        self._int_limit = 20.0

    def update(self, error_x: float, error_y: float,
               dt: float = 1/30,
               vx_px_s: float = 0.0,
               vy_px_s: float = 0.0) -> tuple[float, float]:
        """
        error_x: beacon_px - frame_cx  (+ = beacon right of centre)
        error_y: beacon_py - frame_cy  (+ = beacon below centre)
        vx_px_s: optional estimated beacon velocity in X (px/s)
        vy_px_s: optional estimated beacon velocity in Y (px/s)
        Returns (pan_rate_dps, tilt_rate_dps) to pass to camera.
        """
        cam = CFG.camera
        # Convert pixel error → angular error (degrees)
        ax = error_x / cam.width  * cam.fov_h_deg
        ay = error_y / cam.height * cam.fov_v_deg

        # Convert velocity (px/s) → angular velocity (deg/s) for feedforward
        v_ax = vx_px_s / cam.width  * cam.fov_h_deg
        v_ay = vy_px_s / cam.height * cam.fov_v_deg

        # Integral with anti-windup
        self._int_x = float(np.clip(self._int_x + ax * dt,
                                    -self._int_limit, self._int_limit))
        self._int_y = float(np.clip(self._int_y + ay * dt,
                                    -self._int_limit, self._int_limit))
        # Derivative
        dx = (ax - self._prev_ex) / max(dt, 1e-6)
        dy = (ay - self._prev_ey) / max(dt, 1e-6)
        self._prev_ex, self._prev_ey = ax, ay

        pan_rate  = self._kp * ax + self._ki * self._int_x + self._kd * dx + self._k_ff * v_ax
        tilt_rate = self._kp * ay + self._ki * self._int_y + self._kd * dy + self._k_ff * v_ay

        return float(pan_rate), float(tilt_rate)

    def reset(self):
        self._int_x = self._int_y = 0.0
        self._prev_ex = self._prev_ey = 0.0
