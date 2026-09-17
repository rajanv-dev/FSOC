"""simulation/trajectories.py — Receiver satellite trajectory controller with analytical single-source-of-truth state."""
import math
import random
from dataclasses import dataclass
import numpy as np
from config import CFG

TRAJECTORY_NAMES = ["Circular", "Straight Line", "Figure-8",
                    "Random Walk", "Spiral", "Sinusoidal"]


@dataclass
class ReceiverState:
    x: float               # world x (degrees)
    y: float               # world y (degrees)
    vx: float              # world vx (deg/s)
    vy: float              # world vy (deg/s)
    ax: float              # world ax (deg/s^2)
    ay: float              # world ay (deg/s^2)
    orientation_deg: float # orientation angle (deg)
    t: float               # simulation time (s)


class TrajectoryController:
    def __init__(self):
        self._name = "Circular"
        self._t = 0.0
        self._rng = random.Random(42)
        self.state = ReceiverState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._reset_traj()

    def _reset_traj(self):
        self._t = 0.0
        self._rng = random.Random(42)
        self._compute_state(0.0)

    def set_trajectory(self, name: str):
        self._name = name if name in TRAJECTORY_NAMES else "Circular"
        self._reset_traj()

    def _compute_state(self, dt: float):
        self._t += dt
        t = self._t

        # Trajectory center offset away from Earth (towards deep space)
        tcx = 0.4
        tcy = 0.0

        rx = CFG.camera.fov_h_deg * 0.30   # 30% of FOV radius
        ry = CFG.camera.fov_v_deg * 0.30

        omega = 0.28   # rad/s

        if self._name == "Circular":
            # Orbit centered away from Earth in deep space
            x = tcx + rx * math.cos(omega * t)
            y = tcy + ry * math.sin(omega * t)
            vx = -rx * omega * math.sin(omega * t)
            vy = ry * omega * math.cos(omega * t)
            ax = -rx * (omega ** 2) * math.cos(omega * t)
            ay = -ry * (omega ** 2) * math.sin(omega * t)

        elif self._name == "Straight Line":
            x = tcx + rx * math.sin(omega * t)
            y = tcy + 0.1 * ry * math.cos(0.5 * omega * t)
            vx = rx * omega * math.cos(omega * t)
            vy = -0.05 * ry * omega * math.sin(0.5 * omega * t)
            ax = -rx * (omega ** 2) * math.sin(omega * t)
            ay = -0.025 * ry * (omega ** 2) * math.cos(0.5 * omega * t)

        elif self._name == "Figure-8":
            x = tcx + rx * math.sin(omega * t)
            y = tcy + 0.6 * ry * math.sin(2.0 * omega * t)
            vx = rx * omega * math.cos(omega * t)
            vy = 1.2 * ry * omega * math.cos(2.0 * omega * t)
            ax = -rx * (omega ** 2) * math.sin(omega * t)
            ay = -2.4 * ry * (omega ** 2) * math.sin(2.0 * omega * t)

        elif self._name == "Random Walk":
            dx = 0.15 * math.sin(0.4 * t) + 0.05 * math.sin(1.3 * t)
            dy = 0.12 * math.cos(0.3 * t) + 0.04 * math.sin(1.1 * t)
            x = tcx + dx
            y = tcy + dy
            vx = 0.06 * math.cos(0.4 * t) + 0.065 * math.cos(1.3 * t)
            vy = -0.036 * math.sin(0.3 * t) + 0.044 * math.cos(1.1 * t)
            ax = -0.024 * math.sin(0.4 * t) - 0.0845 * math.sin(1.3 * t)
            ay = -0.0108 * math.cos(0.3 * t) - 0.0484 * math.sin(1.1 * t)

        elif self._name == "Spiral":
            r_scale = 0.5 + 0.5 * math.sin(0.08 * t)
            dr_scale = 0.04 * math.cos(0.08 * t)
            x = tcx + rx * r_scale * math.cos(omega * t)
            y = tcy + ry * r_scale * math.sin(omega * t)
            vx = rx * (dr_scale * math.cos(omega * t) - r_scale * omega * math.sin(omega * t))
            vy = ry * (dr_scale * math.sin(omega * t) + r_scale * omega * math.cos(omega * t))
            ax = -rx * (omega ** 2) * r_scale * math.cos(omega * t)
            ay = -ry * (omega ** 2) * r_scale * math.sin(omega * t)

        else:  # Sinusoidal
            x = tcx + rx * math.sin(omega * t)
            y = tcy + 0.5 * ry * math.sin(2.2 * omega * t)
            vx = rx * omega * math.cos(omega * t)
            vy = 1.1 * ry * omega * math.cos(2.2 * omega * t)
            ax = -rx * (omega ** 2) * math.sin(omega * t)
            ay = -2.42 * ry * (omega ** 2) * math.sin(2.2 * omega * t)

        # Smooth physical disturbances added directly to state
        cfg_d = CFG.disturbance
        if cfg_d.jitter_enabled:
            jx = 0.04 * math.sin(t * 18.0)
            jy = 0.04 * math.cos(t * 22.0)
            x += jx; y += jy
            vx += 0.72 * math.cos(t * 18.0)
            vy += -0.88 * math.sin(t * 22.0)

        # Platform orbital drift / attitude-control residual
        if cfg_d.platform_motion_enabled:
            px = 0.08 * math.sin(t * 2.4)
            py = 0.06 * math.cos(t * 3.1)
            x += px; y += py
            vx += 0.192 * math.cos(t * 2.4)
            vy += -0.186 * math.sin(t * 3.1)

        # HIGH_DYNAMIC_ADAPTIVE: multi-tone sinusoidal jitter (sum of incommensurable
        # frequencies → non-periodic, non-predictable by constant-velocity KF).
        # Amplitude is specified in pixels and converted to world-degrees here.
        if cfg_d.extra_jitter_amplitude_px > 0.0:
            # px → world-deg conversion factor
            px_per_deg_x = CFG.camera.width  / CFG.camera.fov_h_deg
            px_per_deg_y = CFG.camera.height / CFG.camera.fov_v_deg
            amp_x = cfg_d.extra_jitter_amplitude_px / px_per_deg_x
            amp_y = cfg_d.extra_jitter_amplitude_px / px_per_deg_y
            # Three incommensurable sinusoids per axis (structural vibration model)
            w1, w2, w3 = 7.3, 13.7, 23.1   # rad/s
            w4, w5, w6 = 5.9, 11.3, 19.7   # rad/s
            ej_x = amp_x * (math.sin(t * w1) + 0.55 * math.sin(t * w2) + 0.30 * math.sin(t * w3))
            ej_y = amp_y * (math.cos(t * w4) + 0.50 * math.cos(t * w5) + 0.28 * math.sin(t * w6))
            # Corresponding velocity contributions (analytical derivatives)
            evx = amp_x * (w1 * math.cos(t * w1) + 0.55 * w2 * math.cos(t * w2) + 0.30 * w3 * math.cos(t * w3))
            evy = amp_y * (-w4 * math.sin(t * w4) - 0.50 * w5 * math.sin(t * w5) + 0.28 * w6 * math.cos(t * w6))
            x += ej_x; y += ej_y
            vx += evx;  vy += evy

        orientation = (t * 2.0) % 360.0

        self.state = ReceiverState(
            x=float(x), y=float(y),
            vx=float(vx), vy=float(vy),
            ax=float(ax), ay=float(ay),
            orientation_deg=float(orientation),
            t=float(t)
        )

    def step(self, dt: float) -> ReceiverState:
        self._compute_state(dt)
        return self.state

    def reset(self):
        self._reset_traj()

