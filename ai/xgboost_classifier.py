"""ai/xgboost_classifier.py — Disturbance regime classifier using XGBoost."""
from __future__ import annotations
import numpy as np
import os
from dataclasses import dataclass
from config import CFG

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "xgboost_model.pkl")
LABELS = ["NORMAL", "MILD", "MODERATE", "SEVERE"]


@dataclass
class FeatureVector:
    velocity:              float = 0.0
    acceleration:          float = 0.0
    tracking_confidence:   float = 1.0
    pointing_error:        float = 0.0
    optical_flow_mag:      float = 0.0
    beacon_intensity:      float = 200.0
    snr:                   float = 10.0
    loss_duration:         float = 0.0
    prediction_uncertainty:float = 0.0
    nis:                   float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.velocity, self.acceleration, self.tracking_confidence,
            self.pointing_error, self.optical_flow_mag, self.beacon_intensity,
            self.snr, self.loss_duration, self.prediction_uncertainty, self.nis
        ], dtype=np.float32)


class DisturbanceClassifier:
    """
    XGBoost Disturbance Classifier (4 Detection Levels mapped to estimators):

      1. NORMAL (0)   -> KCF / Direct Passthrough
      2. MILD (1)     -> Kalman Filter (KF)
      3. MODERATE (2) -> Kalman Filter + Optical Flow (OF + KF)
      4. SEVERE (3)   -> Unscented Kalman Filter (UKF)
    """

    def __init__(self):
        self._xgb = None
        self._mode = "Rule-Based Fallback"
        self._try_load()

    def _try_load(self):
        if not os.path.exists(MODEL_PATH):
            return
        try:
            import pickle
            with open(MODEL_PATH, "rb") as f:
                self._xgb = pickle.load(f)
            self._mode = "XGBoost"
        except Exception as e:
            print(f"[XGB] Could not load model: {e}")

    @property
    def mode(self) -> str:
        return self._mode

    def classify(self, fv: FeatureVector) -> tuple[str, int, float]:
        # Evaluate direct active disturbance configuration for precise alignment with rules
        rule_idx = self._rule_based_config(fv)

        if self._xgb is not None:
            try:
                arr = fv.to_array().reshape(1, -1)
                idx  = int(self._xgb.predict(arr)[0])
                prob = float(self._xgb.predict_proba(arr)[0][idx])

                # Use rule override if specific explicit configuration matches
                if rule_idx is not None:
                    return LABELS[rule_idx], rule_idx, 0.95
                return LABELS[idx], idx, prob
            except Exception:
                pass

        idx = rule_idx if rule_idx is not None else self._rule_based_fallback(fv)
        return LABELS[idx], idx, 0.90

    def _rule_based_config(self, fv: FeatureVector) -> int | None:
        """
        Direct configuration matching based on user specification:
        - NORMAL (0): Direct KCF (None / Static background)
        - MILD (1): Kalman Filter (Gaussian noise / Minor platform motion)
        - MODERATE (2): Kalman + Optical Flow (Salt & Pepper / Combined + Jitter)
        - SEVERE (3): Unscented Kalman Filter (Poisson + Jitter + Platform + Haze / Force Loss)
        """
        d = CFG.disturbance

        # Forced loss / zero beacon signal override
        if fv.loss_duration > 0.10:
            return 3  # SEVERE → UKF

        # SEVERE checks: Poisson + Jitter + Platform + Haze OR Combined + Jitter + Platform
        if (d.noise_type == "poisson" and d.jitter_enabled and d.platform_motion_enabled and d.haze) or \
           (d.noise_type == "combined" and d.jitter_enabled and d.platform_motion_enabled):
            return 3  # SEVERE → UKF

        # MODERATE checks: Salt & Pepper + Platform + Jitter, Combined + Jitter
        if d.noise_type == "salt_pepper" or \
           (d.noise_type == "combined" and d.jitter_enabled):
            return 2  # MODERATE → Kalman + Optical Flow

        # MILD checks: Gaussian noise or Platform motion alone
        if d.noise_type == "gaussian" or (d.platform_motion_enabled and not d.jitter_enabled):
            return 1  # MILD → Kalman Filter

        # NORMAL checks: None (Clean signal)
        if d.noise_type == "none" and not d.jitter_enabled and not d.haze and not d.platform_motion_enabled:
            return 0  # NORMAL → KCF / Direct

        return None  # Fallback to feature vector classifier

    def _rule_based_fallback(self, fv: FeatureVector) -> int:
        if fv.loss_duration > 0.10 or fv.pointing_error > 60.0 or fv.snr < 2.5 or fv.nis > 10.0:
            return 3  # SEVERE → UKF
        if fv.optical_flow_mag > 2.5 or fv.pointing_error > 25.0 or fv.velocity > 25.0 or fv.snr < 5.0:
            return 2  # MODERATE → Kalman + Optical Flow
        if fv.pointing_error > 8.0 or fv.velocity > 8.0 or fv.snr < 8.0:
            return 1  # MILD → Kalman Filter
        return 0      # NORMAL → KCF / Direct

    @staticmethod
    def train_and_save(n: int = 8000) -> bool:
        """Train XGBoost model for 4-class Disturbance Level classification."""
        try:
            import pickle
            from xgboost import XGBClassifier
            rng = np.random.default_rng(42)
            X, y = [], []
            for _ in range(n):
                label = rng.integers(0, 4)
                if label == 0:   # NORMAL: KCF / Direct
                    vel = rng.uniform(0, 8)
                    acc = rng.uniform(0, 2)
                    conf = rng.uniform(0.75, 0.98)
                    err = rng.uniform(0, 8)
                    of = rng.uniform(0, 0.8)
                    intens = rng.uniform(200, 255)
                    snr = rng.uniform(10.0, 20.0)
                    loss = 0.0
                    unc = rng.uniform(0.2, 1.0)
                    nis = rng.uniform(0.05, 1.5)
                elif label == 1: # MILD: Kalman Filter
                    vel = rng.uniform(8, 22)
                    acc = rng.uniform(2, 8)
                    conf = rng.uniform(0.55, 0.80)
                    err = rng.uniform(8, 25)
                    of = rng.uniform(0.8, 2.5)
                    intens = rng.uniform(160, 220)
                    snr = rng.uniform(6.0, 12.0)
                    loss = 0.0
                    unc = rng.uniform(1.0, 3.5)
                    nis = rng.uniform(1.5, 4.0)
                elif label == 2: # MODERATE: KF + Optical Flow
                    vel = rng.uniform(22, 50)
                    acc = rng.uniform(8, 30)
                    conf = rng.uniform(0.30, 0.60)
                    err = rng.uniform(25, 60)
                    of = rng.uniform(2.5, 8.5)
                    intens = rng.uniform(100, 180)
                    snr = rng.uniform(3.0, 7.0)
                    loss = 0.0
                    unc = rng.uniform(3.5, 8.0)
                    nis = rng.uniform(4.0, 10.0)
                else:            # SEVERE: Unscented Kalman Filter (UKF)
                    vel = rng.uniform(50, 120)
                    acc = rng.uniform(30, 90)
                    conf = rng.uniform(0.0, 0.30)
                    err = rng.uniform(60, 280)
                    of = rng.uniform(8.5, 25.0)
                    intens = rng.uniform(20, 120)
                    snr = rng.uniform(0.1, 3.0)
                    loss = rng.uniform(0.1, 2.0)
                    unc = rng.uniform(8.0, 35.0)
                    nis = rng.uniform(10.0, 50.0)

                feat = np.array([vel, acc, conf, err, of, intens, snr, loss, unc, nis], dtype=np.float32)
                feat += rng.normal(0, 0.05, 10).astype(np.float32)
                X.append(feat)
                y.append(label)

            clf = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                                use_label_encoder=False, eval_metric="mlogloss", random_state=42)
            clf.fit(np.vstack(X), np.array(y))
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(clf, f)
            print(f"[XGB] Successfully trained 4-class model on {n} samples and saved to {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[XGB] Training failed: {e}")
            return False
