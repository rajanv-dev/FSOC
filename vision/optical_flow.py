"""vision/optical_flow.py — Lucas-Kanade optical flow tracker."""
import numpy as np
import cv2
from config import CFG

class OpticalFlowTracker:
    def __init__(self):
        self._prev_frame: np.ndarray | None = None
        self._prev_pts:   np.ndarray | None = None
        self.flow_vectors: list = []

    def initialise(self, frame: np.ndarray, cx: int, cy: int):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self._prev_frame = gray.copy()
        # Seed corners near the tracking point
        roi_pts = np.array([[[cx + dx, cy + dy]] for dx in range(-8,9,4)
                             for dy in range(-8,9,4)], dtype=np.float32)
        self._prev_pts = roi_pts

    def update(self, frame: np.ndarray) -> tuple[float,float,float,list]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._prev_frame is None or self._prev_pts is None or len(self._prev_pts)==0:
            self._prev_frame = gray.copy()
            return 0.0, 0.0, 0.0, []
        lk_params = dict(winSize=CFG.tracking.of_win_size,
                         maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_frame, gray, self._prev_pts, None, **lk_params)
        if next_pts is None:
            self._prev_frame = gray.copy(); return 0.0,0.0,0.0,[]
        good_old = self._prev_pts[status==1]
        good_new = next_pts[status==1]
        vecs = []
        dxs,dys=[],[]
        for (ox,oy),(nx,ny) in zip(good_old.reshape(-1,2), good_new.reshape(-1,2)):
            dxs.append(nx-ox); dys.append(ny-oy)
            vecs.append((float(ox),float(oy),float(nx),float(ny)))
        dx = float(np.median(dxs)) if dxs else 0.0
        dy = float(np.median(dys)) if dys else 0.0
        mag = float(np.hypot(dx,dy))
        self._prev_pts = good_new.reshape(-1,1,2)
        self._prev_frame = gray.copy()
        self.flow_vectors = vecs[:8]
        return dx,dy,mag,vecs

    def reset(self):
        self._prev_frame=None; self._prev_pts=None; self.flow_vectors=[]

