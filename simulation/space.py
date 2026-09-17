"""simulation/space.py — Starfield renderer."""
import numpy as np
from config import CFG

class StarfieldRenderer:
    def __init__(self):
        self._stars = None
        self.reset()

    def reset(self):
        rng = np.random.default_rng(1234)
        n = 350
        self._stars = np.column_stack([
            rng.uniform(-10, 10, n),        # world_x (degrees)
            rng.uniform(-8,   8, n),        # world_y
            rng.integers(20, 200, n),       # grey value
            rng.integers(1, 3, n),          # size (px)
        ])

    def render(self, frame: np.ndarray, pan: float, tilt: float) -> np.ndarray:
        cam = CFG.camera
        W, H = cam.width, cam.height
        for wx, wy, grey, sz in self._stars:
            rx, ry = wx - pan, wy - tilt
            if abs(rx) > cam.fov_h_deg / 2 + 0.2 or abs(ry) > cam.fov_v_deg / 2 + 0.2:
                continue
            px = int(W / 2 + (rx / cam.fov_h_deg) * W)
            py = int(H / 2 - (ry / cam.fov_v_deg) * H)
            if 0 <= px < W and 0 <= py < H:
                frame[max(0,py-int(sz)):py+int(sz)+1,
                      max(0,px-int(sz)):px+int(sz)+1] = int(grey)
        return frame
