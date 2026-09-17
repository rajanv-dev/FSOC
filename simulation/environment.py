"""
simulation/environment.py  — FSOC scene compositor (procedural satellite)

Layers:
  0: Deep space (black)
  1: Stars (stationary reference for sender camera view)
  2: Earth sprite (assets/earth.png)
  3: Receiver satellite (procedurally drawn with physical disturbances applied)
     - Jitter (random high-frequency vibration of the receiver satellite)
     - Platform motion (orbital drift/wobble of the receiver satellite)
  4: Sensor effects (camera sensor readout noise & optical haze)

Camera state is ONLY changed by apply_pan_tilt_rate().
"""
import os, sys, math
import numpy as np
import cv2

from config import CFG
from simulation.trajectories import TrajectoryController
from simulation.beacon import BeaconModel
from simulation.space import StarfieldRenderer
from simulation.satellite_renderer import draw_satellite
from camera.sensor_model import SensorModel

from pathlib import Path

# ---------------------------------------------------------------------------
# Earth asset loading — hard-fail with clear message
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ASSET_DIR = _PROJECT_ROOT / "assets"

def _load_earth() -> np.ndarray | None:
    path = _ASSET_DIR / "earth.png"
    if not path.exists():
        print(f"[ERROR] Earth asset NOT FOUND at {path.resolve()}", file=sys.stderr)
        return None
    print(f"EARTH ASSET: FOUND {path.resolve()}", file=sys.stderr)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img

_EARTH_BGR = _load_earth()


# ---------------------------------------------------------------------------
_EARTH_CACHE: dict = {}

def _draw_earth(frame: np.ndarray, cx: int, cy: int, radius_px: int):
    if _EARTH_BGR is not None:
        tw = th = max(8, radius_px * 2)
        cache_key = (tw, th)
        if cache_key in _EARTH_CACHE:
            s_bgr, alpha = _EARTH_CACHE[cache_key]
        else:
            sprite = cv2.resize(_EARTH_BGR, (tw, th), interpolation=cv2.INTER_AREA)
            if sprite.ndim == 3 and sprite.shape[2] == 4:
                s_bgr = sprite[:, :, :3]
                alpha = (sprite[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
            elif sprite.ndim == 3:
                s_bgr = sprite
                alpha = (cv2.cvtColor(s_bgr, cv2.COLOR_BGR2GRAY) > 5).astype(np.float32)[:, :, None]
            else:
                s_bgr = cv2.cvtColor(sprite, cv2.COLOR_GRAY2BGR)
                alpha = (sprite > 5).astype(np.float32)[:, :, None]
            if len(_EARTH_CACHE) > 10:
                _EARTH_CACHE.clear()
            _EARTH_CACHE[cache_key] = (s_bgr, alpha)


        H, W = frame.shape[:2]
        x0, y0 = cx - tw // 2, cy - th // 2
        x1, y1 = x0 + tw,      y0 + th
        sx0 = max(0, -x0);  sy0 = max(0, -y0)
        fx0 = max(0, x0);   fy0 = max(0, y0)
        fx1 = min(W, x1);   fy1 = min(H, y1)
        sx1 = sx0 + fx1 - fx0;  sy1 = sy0 + fy1 - fy0
        if fx1 > fx0 and fy1 > fy0 and sx1 > sx0 and sy1 > sy0:
            if frame.ndim == 3:
                roi   = frame[fy0:fy1, fx0:fx1].astype(np.float32)
                s_roi = s_bgr[sy0:sy1, sx0:sx1].astype(np.float32)
                m_roi = alpha[sy0:sy1, sx0:sx1]
                frame[fy0:fy1, fx0:fx1] = np.clip(
                    roi * (1.0 - m_roi) + s_roi * m_roi, 0, 255
                ).astype(np.uint8)
            else:
                grey = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2GRAY)
                roi   = frame[fy0:fy1, fx0:fx1].astype(np.float32)
                s_roi = grey[sy0:sy1, sx0:sx1].astype(np.float32)
                m_roi = alpha[sy0:sy1, sx0:sx1, 0]
                frame[fy0:fy1, fx0:fx1] = np.clip(
                    roi * (1.0 - m_roi) + s_roi * m_roi, 0, 255
                ).astype(np.uint8)
    else:
        col = (70, 90, 110) if frame.ndim == 3 else 70
        cv2.circle(frame, (cx, cy), radius_px, col, -1)


# ---------------------------------------------------------------------------
# SimulationEnvironment
# ---------------------------------------------------------------------------
class SimulationEnvironment:
    def __init__(self):
        self.trajectory = TrajectoryController()
        self.beacon     = BeaconModel()
        self.stars      = StarfieldRenderer()
        self.sensor     = SensorModel()

        self._rng = np.random.default_rng(101)

        # Explicit camera state — ONLY changed by apply_pan_tilt_rate()
        self._pan_deg:  float = 0.0
        self._tilt_deg: float = 0.0

        # IR aperture pixel position (set during render → used by tracker)
        self.beacon_px: int = CFG.camera.width  // 2
        self.beacon_py: int = CFG.camera.height // 2

        self._first_frame = True

    # ── camera interface ─────────────────────────────────────────────────
    @property
    def pan_deg(self)  -> float: return self._pan_deg
    @property
    def tilt_deg(self) -> float: return self._tilt_deg

    def set_orientation(self, pan: float, tilt: float):
        self._pan_deg, self._tilt_deg = pan, tilt

    def apply_pan_tilt_rate(self, pan_dps: float, tilt_dps: float, dt: float):
        """PID closes the loop here — only place camera orientation changes."""
        mp = CFG.camera.pan_speed_max
        mt = CFG.camera.tilt_speed_max
        self._pan_deg  += float(np.clip( pan_dps,  -mp, mp)) * dt   # pan: + = look right
        self._tilt_deg -= float(np.clip(tilt_dps,  -mt, mt)) * dt   # tilt: - = look down

    # ── projection ───────────────────────────────────────────────────────
    def world_to_pixel(self, wx: float, wy: float) -> tuple[int, int]:
        cam = CFG.camera
        rx  = wx - self._pan_deg
        ry  = wy - self._tilt_deg
        px  = int(cam.width  / 2 + (rx / cam.fov_h_deg) * cam.width)
        py  = int(cam.height / 2 - (ry / cam.fov_v_deg) * cam.height)
        return px, py

    def is_in_fov(self, wx: float, wy: float, margin: float = 1.5) -> bool:
        cam = CFG.camera
        return (abs(wx - self._pan_deg)  <= cam.fov_h_deg / 2 * margin and
                abs(wy - self._tilt_deg) <= cam.fov_v_deg / 2 * margin)

    # ── world step ───────────────────────────────────────────────────────
    def step(self, dt: float):
        st = self.trajectory.step(dt)
        self.beacon.step(dt)

        if self._first_frame:
            self._pan_deg  = st.x
            self._tilt_deg = st.y
            self._first_frame = False

    # ── render ───────────────────────────────────────────────────────────
    def render(self) -> np.ndarray:
        cam = CFG.camera
        W, H = cam.width, cam.height
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        st = self.trajectory.state

        # Layer 1 – stars (stationary background reference)
        star_gray = np.zeros((H, W), dtype=np.uint8)
        star_gray = self.stars.render(star_gray, self._pan_deg, self._tilt_deg)
        frame = cv2.cvtColor(star_gray, cv2.COLOR_GRAY2BGR)

        # Layer 2 – Earth (background, lower-left of boresight)
        epx, epy = self.world_to_pixel(CFG.earth.centre_offset_x,
                                       CFG.earth.centre_offset_y)
        e_r = int((CFG.earth.apparent_radius_deg / cam.fov_h_deg) * W)
        if CFG.earth.visible and e_r > 4:
            _draw_earth(frame, epx, epy, e_r)

        # Layer 3 – Receiver satellite with red light beacon
        sx, sy = self.world_to_pixel(st.x, st.y)
        sat_sz = CFG.satellite.render_size_px
        ir_on = not self.beacon.is_forced_loss
        ir_phase = self.beacon.pulse_phase if ir_on else -math.pi

        bx, by = draw_satellite(frame, sx, sy,
                                size      = sat_sz,
                                angle_deg = st.orientation_deg * 0.25,
                                ir_phase  = ir_phase)

        self.beacon_px = int(np.clip(bx, 0, W - 1))
        self.beacon_py = int(np.clip(by, 0, H - 1))

        # Layer 4 – Camera sensor effects
        frame = self.sensor.apply(frame)
        return frame

    def reset(self):
        self.trajectory.reset()
        self.beacon.reset()
        self.stars.reset()
        self._pan_deg    = 0.0
        self._tilt_deg   = 0.0
        self._first_frame = True
        self._angle_deg   = 0.0
        self._t           = 0.0
        self.beacon_px   = CFG.camera.width  // 2
        self.beacon_py   = CFG.camera.height // 2
