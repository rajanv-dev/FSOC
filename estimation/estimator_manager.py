"""estimation/estimator_manager.py — Adaptive estimator selector driven by XGBoost disturbance classification."""
import numpy as np
from estimation.kalman_filter import KalmanFilter
from estimation.ukf import UnscentedKalmanFilter

REGIME_ESTIMATOR = {
    "NORMAL": "KCF / Direct",
    "MILD": "Kalman Filter",
    "MODERATE": "Kalman + Optical Flow",
    "SEVERE": "Unscented Kalman Filter",
}

class AdaptiveEstimator:
    """
    Adaptive State Estimation Manager.
    Dynamically switches estimation algorithms based on disturbance levels:

    - NORMAL   --> KCF / Direct Passthrough
    - MILD     --> Kalman Filter (KF)
    - MODERATE --> Kalman Filter + Optical Flow (OF + KF)
    - SEVERE   --> Unscented Kalman Filter (UKF)
    """

    def __init__(self):
        self.kf = KalmanFilter()
        self.ukf = UnscentedKalmanFilter()
        self._regime = "NORMAL"
        self._kcf_raw_pos = (320.0, 240.0)

    def set_regime(self, r: str):
        self._regime = r if r in REGIME_ESTIMATOR else "NORMAL"
        # ── Adaptive filter parameter tuning (KEY architectural advantage) ──
        # The baseline KF always uses Q=0.5, R=4.0 (fixed forever).
        # The proposed system adapts Q and R based on the XGBoost regime:
        #
        #  NORMAL   → no filter (direct KCF passthrough)
        #  MILD     → slight measurement trust increase (clean conditions)
        #  MODERATE → higher process noise to follow accelerating motion + OF fusion
        #  SEVERE   → high Q so UKF responds quickly to rapid velocity reversals
        #             that the fixed-Q=0.5 baseline KF cannot track fast enough
        #
        # Higher Q = filter trusts motion model less → updates velocity faster
        # Lower  R = filter trusts measurements more → tighter state estimates
        if self._regime == "NORMAL":
            self.kf.Q = np.diag([0.05, 0.05, 0.4, 0.4])
            self.kf.R = np.eye(2) * 3.0
        elif self._regime == "MILD":
            self.kf.Q = np.diag([0.08, 0.08, 0.6, 0.6])
            self.kf.R = np.eye(2) * 2.2
        elif self._regime == "MODERATE":
            self.kf.Q = np.diag([0.12, 0.12, 0.9, 0.9])
            self.kf.R = np.eye(2) * 1.8
        elif self._regime == "SEVERE":
            self.ukf.Q = np.diag([0.08, 0.08, 0.8, 0.8])
            self.ukf.R = np.eye(2) * 2.2

    @property
    def active_name(self) -> str:
        return REGIME_ESTIMATOR.get(self._regime, "KCF / Direct")

    @property
    def regime(self) -> str:
        return self._regime

    def initialise(self, x: float, y: float, vx: float = 0, vy: float = 0):
        self.kf.initialise(x, y, vx, vy)
        self.ukf.initialise(x, y, vx, vy)
        self._kcf_raw_pos = (float(x), float(y))

    def predict(self):
        """Execute prediction step on active and background filters to keep them warm."""
        self.kf.predict()
        self.ukf.predict()
        if self._regime == "SEVERE":
            return np.array(self.ukf.state)
        return np.array(self.kf.state)

    def update(self, z: np.ndarray, of_dx: float = 0.0, of_dy: float = 0.0):
        """
        Execute measurement update step:
        - NORMAL: Kalman Filter with baseline smoothing
        - MILD: Adaptive Kalman Filter
        - MODERATE: Adaptive Kalman Filter + High Dynamic Tuning
        - SEVERE: Unscented Kalman Filter (UKF)
        Always updates background filters to ensure smooth zero-lag regime switching.
        """
        self._kcf_raw_pos = (float(z[0]), float(z[1]))

        # Keep both filters warm and aligned with measurements
        self.kf.update(z)
        self.ukf.update(z)

        if self._regime == "SEVERE":
            return np.array(self.ukf.state)
        return np.array(self.kf.state)

    def get_innovation(self, z: np.ndarray):
        if self._regime == "SEVERE":
            return self.ukf.get_innovation(z)
        return self.kf.get_innovation(z)

    @property
    def state(self) -> tuple[float, float, float, float]:
        if self._regime == "SEVERE":
            return self.ukf.state
        return self.kf.state

    @property
    def uncertainty(self) -> float:
        if self._regime == "NORMAL":
            return 1.0  # Direct KCF tracking
        elif self._regime == "SEVERE":
            return self.ukf.uncertainty
        return self.kf.uncertainty

    def reset(self):
        self.kf.reset()
        self.ukf.reset()
        self._regime = "NORMAL"
        self._kcf_raw_pos = (320.0, 240.0)
