"""estimation/kalman_filter.py — Linear Kalman Filter [x, y, vx, vy]."""
import numpy as np

class KalmanFilter:
    def __init__(self, dt: float = 1/30, q: float = 0.5, r: float = 4.0):
        self._initialised = False
        self.dt = dt
        # State [x, y, vx, vy]
        self.x_hat = np.zeros((4, 1))
        self.P = np.eye(4) * 100
        # Transition
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        self.Q = np.eye(4) * q
        self.R = np.eye(2) * r

    def initialise(self, x, y, vx=0, vy=0):
        self.x_hat = np.array([[x],[y],[vx],[vy]], dtype=np.float64)
        self.P = np.eye(4) * 50
        self._initialised = True

    def predict(self) -> np.ndarray:
        self.x_hat = self.F @ self.x_hat
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x_hat.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z).flatten()[:2].reshape(2,1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ (z - self.H @ self.x_hat)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x_hat.copy()

    def get_innovation(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_flat = np.asarray(z).flatten()[:2]
        z_pred = (self.H @ self.x_hat).flatten()
        r = z_flat - z_pred
        S = self.H @ self.P @ self.H.T + self.R
        return r, S

    @property
    def state(self):
        x = self.x_hat.flatten()
        return float(x[0]), float(x[1]), float(x[2]), float(x[3])

    @property
    def uncertainty(self) -> float:
        return float(np.sqrt(self.P[0,0] + self.P[1,1]))

    @property
    def innovation_covariance(self) -> np.ndarray:
        return self.H @ self.P @ self.H.T + self.R

    def reset(self):
        self.__init__()
