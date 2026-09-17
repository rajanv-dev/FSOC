"""benchmark/video_input.py — Robust MP4/Video frame reader for benchmark mode."""
import cv2
import numpy as np

class MP4Reader:
    def __init__(self, path: str):
        self._path = path
        self._cap: cv2.VideoCapture | None = None
        self._total_frames = 0
        self._fps = 30.0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._path)
        if self._cap.isOpened():
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                self._fps = fps
            return True
        return False

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def fps(self) -> float:
        return self._fps

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        if not self.is_open:
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop at EOF
            ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        # Resize to standard 640x480 InGaAs format
        if frame.shape[1] != 640 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        return True, frame

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None
