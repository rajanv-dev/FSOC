"""
simulation/beacon.py — 1550 nm IR Optical Beacon Model

The beacon is a collimated 1550 nm laser mounted on the receiver satellite.
On the sender's InGaAs/SWIR camera it appears as:
  - A saturated bright core (≈255 digital level)
  - A fast Gaussian falloff
  - A pulsing acquisition ring (so the tracker can see it even at low SNR)
  - Optional fade / forced loss states
"""
import math
import numpy as np
from config import CFG


class BeaconModel:
    def __init__(self):
        self._forced_loss = False
        self._t           = 0.0
        self._fade        = 1.0     # 0.0 = fully off, 1.0 = full power

    # ------------------------------------------------------------------
    @property
    def is_forced_loss(self) -> bool:
        return self._forced_loss

    @property
    def current_intensity(self) -> float:
        """Peak digital level of the IR beacon hot-spot (0–255)."""
        base    = float(CFG.beacon.intensity) * self._fade
        # Subtle amplitude flicker to simulate atmospheric / pointing jitter
        flicker = 1.0 + 0.03 * math.sin(self._t * 9.7) \
                      + 0.02 * math.sin(self._t * 23.1)
        return float(np.clip(base * flicker, 0, 255))

    @property
    def pulse_phase(self) -> float:
        """0–2π acquisition-ring pulse phase (for visual overlay)."""
        return self._t * 4.0  # 4 rad/s pulsing

    def force_loss(self, active: bool):
        self._forced_loss = active
        if active:
            self._fade = 0.0
        else:
            self._fade = 1.0   # instant restore (PID must reacquire)

    def step(self, dt: float):
        self._t += dt
        # Apply scenario-controlled fade factor (1.0 = full power, neutral for existing scenarios).
        # Only apply when not in forced_loss — forced_loss controls _fade independently.
        from config import CFG
        if not self._forced_loss:
            self._fade = CFG.disturbance.beacon_fade_factor

    def reset(self):
        self._forced_loss = False
        self._t           = 0.0
        self._fade        = 1.0
        # Reset the shared config field so consecutive runs are independent
        from config import CFG
        CFG.disturbance.beacon_fade_factor = 1.0
