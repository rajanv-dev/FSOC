"""benchmark/metrics.py — Performance metrics tracker."""
import time
import numpy as np
from config import CFG

class PerformanceMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self._t0            = None
        self._acquired      = False
        self._acq_time      = None
        self._errors:  list[float] = []
        self._fps_buf: list[float] = []
        self._lat_buf: list[float] = []
        self._total_frames  = 0
        self._loss_frames   = 0
        self._reacq_times:  list[float] = []
        self._lock_frames   = 0
        self._reacq_t0      = None

    def mark_session_start(self):
        self._t0 = time.perf_counter()

    def mark_acquired(self):
        if not self._acquired and self._t0 is not None:
            self._acq_time = time.perf_counter() - self._t0
            self._acquired = True

    def record_frame(self, bx: float, by: float, detected: bool,
                     fps: float, latency_ms: float):
        W, H = CFG.camera.width, CFG.camera.height
        self._total_frames += 1
        if detected:
            err = float(np.hypot(bx - W/2, by - H/2))
            self._errors.append(err)
            self._lock_frames += 1
            if self._reacq_t0 is not None:
                self._reacq_times.append(time.perf_counter() - self._reacq_t0)
                self._reacq_t0 = None
        else:
            self._loss_frames += 1
            if self._reacq_t0 is None:
                self._reacq_t0 = time.perf_counter()
        self._fps_buf.append(fps)
        self._lat_buf.append(latency_ms)
        if len(self._fps_buf) > 120: self._fps_buf.pop(0)
        if len(self._lat_buf) > 120: self._lat_buf.pop(0)

    @property
    def acquisition_time(self): return self._acq_time
    @property
    def rmse(self):
        return float(np.sqrt(np.mean(np.square(self._errors[-200:])))) if self._errors else 0.0
    @property
    def target_loss_pct(self):
        if self._total_frames == 0: return 0.0
        return 100.0 * self._loss_frames / self._total_frames
    @property
    def avg_reacq_time(self):
        return float(np.mean(self._reacq_times)) if self._reacq_times else None
    @property
    def avg_fps(self): return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0
    @property
    def avg_latency_ms(self): return float(np.mean(self._lat_buf)) if self._lat_buf else 0.0
    @property
    def lock_retention_pct(self):
        if self._total_frames == 0: return 0.0
        return 100.0 * self._lock_frames / self._total_frames
