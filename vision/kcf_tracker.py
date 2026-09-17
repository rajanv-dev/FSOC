"""vision/kcf_tracker.py — KCF tracker, OpenCV 4/5 compatible."""
import numpy as np
import cv2
from config import CFG

def _make_kcf():
    if hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    elif hasattr(cv2, 'TrackerKCF'):
        try:
            return cv2.TrackerKCF.create()
        except Exception:
            pass
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
        return cv2.legacy.TrackerKCF_create()
    return None


class KCFTracker:
    def __init__(self):
        self._tracker = None
        self._template = None
        self._bbox = None
        self._initialised = False
        self._psr_hist: list[float] = []

    @property
    def is_initialised(self): return self._initialised
    @property
    def bbox(self): return self._bbox

    def reinitialise(self, frame: np.ndarray, cx: int, cy: int):
        sz = CFG.tracking.kcf_bbox_size; half = sz // 2
        H, W = frame.shape[:2]
        x, y = max(0, cx-half), max(0, cy-half)
        w, h = min(sz, W-x), min(sz, H-y)
        if w < 4 or h < 4: return
        self._bbox = (x, y, w, h)
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim==2 else frame

        try:
            self._tracker = _make_kcf()
            if self._tracker is not None:
                self._tracker.init(bgr, self._bbox)
                self._initialised = True
                self._psr_hist = []
                return
        except Exception:
            self._tracker = None

        # Fallback: Correlation template matching tracker
        self._template = bgr[y:y+h, x:x+w].copy()
        self._initialised = True
        self._psr_hist = []

    def update(self, frame: np.ndarray) -> tuple[float, float, float]:
        if not self._initialised:
            return -1, -1, 0.0
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim==2 else frame

        if self._tracker is not None:
            try:
                ok, bbox = self._tracker.update(bgr)
                if ok:
                    bx,by,bw,bh = [int(v) for v in bbox]
                    self._bbox = (bx,by,bw,bh)
                    cx, cy = bx+bw/2, by+bh/2
                    conf = self._psr(frame, bx, by, bw, bh)
                    self._psr_hist.append(conf)
                    if len(self._psr_hist)>10: self._psr_hist.pop(0)
                    return cx, cy, conf
            except Exception:
                pass

        # Correlation template fallback search in local ROI
        if self._bbox is None or self._template is None:
            self._initialised = False
            return -1, -1, 0.0

        bx, by, bw, bh = self._bbox
        H, W = frame.shape[:2]
        pad = 50
        rx0, ry0 = max(0, bx - pad), max(0, by - pad)
        rx1, ry1 = min(W, bx + bw + pad), min(H, by + bh + pad)
        roi = bgr[ry0:ry1, rx0:rx1]

        if roi.shape[0] < bh or roi.shape[1] < bw:
            self._initialised = False
            return -1, -1, 0.0

        res = cv2.matchTemplate(roi, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < 0.25:
            self._initialised = False
            return -1, -1, 0.0

        nx = rx0 + max_loc[0]
        ny = ry0 + max_loc[1]
        self._bbox = (nx, ny, bw, bh)
        cx, cy = nx + bw / 2.0, ny + bh / 2.0
        conf = float(np.clip(max_val, 0.0, 1.0))
        self._psr_hist.append(conf)
        if len(self._psr_hist) > 10: self._psr_hist.pop(0)
        return cx, cy, conf

    def _psr(self, frame, bx,by,bw,bh):
        H,W=frame.shape[:2]
        x0,y0=max(0,bx),max(0,by); x1,y1=min(W,bx+bw),min(H,by+bh)
        roi=frame[y0:y1,x0:x1].astype(np.float32)
        if roi.size==0: return 0.0
        peak=float(roi.max()); mean=float(roi.mean()); std=float(roi.std())+1e-6
        psr_val = (peak - mean) / std
        return float(np.clip(psr_val / 7.5, 0.0, 1.0))

    def reset(self):
        self._tracker=None; self._template=None; self._bbox=None
        self._initialised=False; self._psr_hist=[]

