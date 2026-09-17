"""camera/sensor_model.py — SWIR InGaAs camera sensor model: noise & optical haze."""
import numpy as np
import cv2
from config import CFG

class SensorModel:
    """
    Models the optical camera sensor characteristics:
      - Readout noise (Gaussian, Salt & Pepper, Poisson, Combined)
      - Atmospheric / space optical haze
    (Target physical disturbances like Jitter and Platform Motion are applied
    directly to the receiver satellite's trajectory, not the sensor view.)
    """

    def __init__(self):
        self._rng = np.random.default_rng(99)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        cfg = CFG.disturbance

        # ── SWIR/InGaAs Sensor Noise (Optimized Fast Vector Operations) ──
        if cfg.noise_type == "gaussian":
            noise = np.empty_like(frame, dtype=np.int16)
            cv2.randn(noise, 0, cfg.gaussian_sigma)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif cfg.noise_type == "salt_pepper":
            num_sp = int(frame.shape[0] * frame.shape[1] * cfg.salt_pepper_ratio * 0.5)
            if num_sp > 0:
                ys = self._rng.integers(0, frame.shape[0], num_sp)
                xs = self._rng.integers(0, frame.shape[1], num_sp)
                frame[ys, xs] = 255
                ys_p = self._rng.integers(0, frame.shape[0], num_sp)
                xs_p = self._rng.integers(0, frame.shape[1], num_sp)
                frame[ys_p, xs_p] = 0
        elif cfg.noise_type == "poisson":
            # Physically accurate optical Poisson shot noise model: sigma proportional to sqrt(signal)
            shot_noise = np.empty_like(frame, dtype=np.int16)
            cv2.randn(shot_noise, 0, 14.0)
            frame = np.clip(frame.astype(np.int16) + shot_noise, 0, 255).astype(np.uint8)
        elif cfg.noise_type == "combined":
            noise = np.empty_like(frame, dtype=np.int16)
            cv2.randn(noise, 0, cfg.gaussian_sigma * 0.7)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            num_sp = int(frame.shape[0] * frame.shape[1] * cfg.salt_pepper_ratio * 0.3)
            if num_sp > 0:
                ys = self._rng.integers(0, frame.shape[0], num_sp)
                xs = self._rng.integers(0, frame.shape[1], num_sp)
                frame[ys, xs] = 255

        # ── Optical Haze ──────────────────────────────────────────────────
        if cfg.haze:
            frame = cv2.addWeighted(frame, 0.82, np.full_like(frame, 30), 0.18, 0)

        return frame

