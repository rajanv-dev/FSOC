"""validation/nis.py — Normalised Innovation Squared validator."""
import numpy as np
from scipy import stats

class NISValidator:
    def __init__(self, dof: int = 2, confidence: float = 0.95):
        self.threshold = float(stats.chi2.ppf(confidence, df=dof))
        self._history: list[float] = []

    def validate(self, residual: np.ndarray,
                 S: np.ndarray) -> tuple[float, bool, str]:
        r = np.atleast_1d(np.squeeze(np.asarray(residual, dtype=np.float64))).ravel()
        try:
            nis = float(r @ np.linalg.inv(S) @ r)
        except Exception:
            nis = 0.0
        self._history.append(nis)
        if len(self._history) > 60: self._history.pop(0)
        valid = nis <= self.threshold
        return nis, valid, ("VALID" if valid else "INVALID")
