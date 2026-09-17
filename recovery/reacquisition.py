"""recovery/reacquisition.py — 4-level hierarchical reacquisition."""
import numpy as np
import cv2

LEVELS = ["Predicted ROI", "Expanded ROI", "Global Scan", "Beacon-Selective"]

class HierarchicalReacquisition:
    def __init__(self):
        self._active = False
        self._level  = 0
        self._start_frame = 0
        self._frames_at_level = 0

    @property
    def is_active(self): return self._active
    @property
    def current_level_name(self): return LEVELS[self._level] if self._active else "--"

    def start(self, frame_idx: int):
        if not self._active:
            self._active = True
            self._level  = 0
            self._start_frame = frame_idx
            self._frames_at_level = 0

    def attempt(self, frame: np.ndarray, pred_x: float,
                pred_y: float) -> tuple[int,int] | None:
        self._frames_at_level += 1
        if self._frames_at_level > 6:
            self._level = min(self._level + 1, len(LEVELS)-1)
            self._frames_at_level = 0

        H, W = frame.shape[:2]
        if self._level == 0:   roi_r = 60
        elif self._level == 1: roi_r = 120
        elif self._level == 2: roi_r = min(W, H) // 2
        else:                  roi_r = min(W, H) // 2

        px, py = int(np.clip(pred_x, 0, W-1)), int(np.clip(pred_y, 0, H-1))
        x0 = max(0, px-roi_r); y0 = max(0, py-roi_r)
        x1 = min(W, px+roi_r); y1 = min(H, py+roi_r)

        roi = frame[y0:y1, x0:x1]
        if roi.size == 0: return None

        from vision.acquisition import acquire_beacon
        cands = acquire_beacon(roi)
        if cands:
            cx_l, cy_l, score = cands[0]
            if score > 0.8:
                self.reset()
                return (x0 + cx_l, y0 + cy_l)
        return None

    def reset(self):
        self._active = False
        self._level  = 0
        self._frames_at_level = 0
