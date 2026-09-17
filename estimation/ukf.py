"""estimation/ukf.py — Unscented Kalman Filter with regularisation."""
import numpy as np

class UnscentedKalmanFilter:
    def __init__(self):
        self.n   = 4
        alpha, beta, kappa = 0.8, 2.0, 2.0
        lam = alpha**2 * (self.n + kappa) - self.n
        self.Wm = np.full(2*self.n+1, 0.5/(self.n+lam))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam/(self.n+lam)
        self.Wc[0] = lam/(self.n+lam) + (1-alpha**2+beta)
        self._sqrt_factor = self.n + lam
        self.x_hat = np.zeros(self.n)
        self.P  = np.eye(self.n)*100
        self.Q  = np.eye(self.n)*1.5
        self.R  = np.eye(2)*1.0
        self.H  = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        self._initialised = False

    def initialise(self, x, y, vx=0, vy=0):
        self.x_hat = np.array([x,y,float(vx),float(vy)])
        self.P = np.eye(self.n)*50
        self._initialised = True

    def _regularise(self):
        self.P = (self.P+self.P.T)/2 + np.eye(self.n)*1e-5

    def _sigma_points(self):
        self._regularise()
        try:
            L = np.linalg.cholesky(self.P*self._sqrt_factor)
        except np.linalg.LinAlgError:
            self.P += np.eye(self.n)*0.1
            L = np.linalg.cholesky(self.P*self._sqrt_factor)
        pts=[self.x_hat]
        for i in range(self.n): pts.append(self.x_hat+L[:,i])
        for i in range(self.n): pts.append(self.x_hat-L[:,i])
        return np.array(pts)

    def _fx(self, s, dt=1/30):
        return np.array([s[0]+s[2]*dt, s[1]+s[3]*dt, s[2], s[3]])

    def _hx(self, s):
        return s[:2]

    def predict(self):
        sigma = self._sigma_points()
        sigma_p = np.array([self._fx(s) for s in sigma])
        self.x_hat = np.sum(self.Wm[:,None]*sigma_p, axis=0)
        self.P = self.Q.copy()
        for i,s in enumerate(sigma_p):
            d=(s-self.x_hat).reshape(-1,1); self.P+=self.Wc[i]*(d@d.T)
        self._regularise()
        return self.x_hat.copy()

    def update(self, z: np.ndarray):
        z=np.asarray(z).flatten()[:2]
        sigma=self._sigma_points()
        z_sigma=np.array([self._hx(s) for s in sigma])
        z_pred=np.sum(self.Wm[:,None]*z_sigma, axis=0)
        S=self.R.copy()
        Pxz=np.zeros((self.n,2))
        for i,s in enumerate(sigma):
            dz=(z_sigma[i]-z_pred).reshape(-1,1)
            dx=(s-self.x_hat).reshape(-1,1)
            S+=self.Wc[i]*(dz@dz.T); Pxz+=self.Wc[i]*(dx@dz.T)
        try: K=Pxz@np.linalg.inv(S)
        except: K=np.zeros((self.n,2))
        self.x_hat+=K@(z-z_pred)
        self.P-=K@S@K.T; self._regularise()
        return self.x_hat.copy()

    def get_innovation(self, z: np.ndarray):
        z_flat=np.asarray(z).flatten()[:2]
        z_pred=(self.H@self.x_hat).flatten()
        r=z_flat-z_pred
        S=self.H@self.P@self.H.T+self.R
        return r,S

    @property
    def state(self): return tuple(float(v) for v in self.x_hat)
    @property
    def uncertainty(self): return float(np.sqrt(self.P[0,0]+self.P[1,1]))
    @property
    def innovation_covariance(self): return self.H@self.P@self.H.T+self.R

    def reset(self): self.__init__()
