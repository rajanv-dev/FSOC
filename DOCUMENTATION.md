# Complete Source Code Technical Documentation

## 1. Executive Technical Overview

The **FSOC AI Virtual Camera Tracking System** is a mission-critical software suite designed for Free-Space Optical Communication (FSOC) and satellite laser beam fine-pointing. High-precision laser pointing requires sub-pixel beam alignment under extreme orbital dynamic rates, platform vibration, atmospheric turbulence, cloud haze, and target optical fading.

This repository provides an end-to-end simulation, vision, state estimation, machine learning classification, control engineering, and interactive visualization framework.

---

## 2. System Architecture & Data Flow

```
+-----------------------------------------------------------------------------------+
|                            PHYSICS & SENSOR SIMULATION                            |
|                                                                                   |
|   Orbit Engine -------> Atmospheric Disturbance -------> Virtual InGaAs Sensor    |
|   (LEO/GEO/Jitter)      (Gaussian/Poisson/Haze)         (640x480 @ 30 FPS)      |
+------------------------------------------+----------------------------------------+
                                           | Raw Image Frame (Grayscale / BGR)
                                           v
+-----------------------------------------------------------------------------------+
|                             VISION & FEATURE EXTRACTION                           |
|                                                                                   |
|   Intensity Centroid      KCF Tracker            Lucas-Kanade Optical Flow        |
|   (Weighted Center)       (Bounding Box)         (Feature Vector Motion)          |
+------------------------------------------+----------------------------------------+
                                           | Measured Position z_k = [x, y]^T
                                           v
+-----------------------------------------------------------------------------------+
|                            AI DISTURBANCE CLASSIFICATION                          |
|                                                                                   |
|   Feature Vector Generation  -------------->  XGBoost Classifier Model           |
|   [vel, acc, err, snr, nis, ...]                (NORMAL, MILD, MODERATE, SEVERE)  |
+------------------------------------------+----------------------------------------+
                                           | Regime Classification & Confidence
                                           v
+-----------------------------------------------------------------------------------+
|                             ADAPTIVE STATE ESTIMATION                             |
|                                                                                   |
|   Regime = NORMAL   ---> Direct Passthrough / KCF                                 |
|   Regime = MILD     ---> Standard Linear Kalman Filter (KF)                       |
|   Regime = MODERATE ---> Kalman Filter + Optical Flow                             |
|   Regime = SEVERE   ---> Unscented Kalman Filter (UKF)                            |
+------------------------------------------+----------------------------------------+
                                           | State Estimate x_hat = [x, y, vx, vy]^T
                                           v
+-----------------------------------------------------------------------------------+
|                             GIMBAL CONTROL SUBSYSTEM                              |
|                                                                                   |
|   Pixel Error (e_x, e_y) + Velocity (vx, vy) ---> Feedforward PID Controller       |
|                                                   Pan/Tilt Angular Rates (deg/s)  |
+-----------------------------------------------------------------------------------+
```

---

## 3. Detailed Component & Module Specifications

### 3.1 Main Entry Point & Configuration

#### `main.py`
- **Purpose**: Main executable entry point for launching the PySide6 user interfaces or running command-line workflows.
- **CLI Arguments**:
  - `--train-xgb`: Executes XGBoost synthetic training script and exits (`DisturbanceClassifier.train_and_save(8000)`).
  - `--compare`: Launches directly into the `ComparisonDashboard` UI.
  - `--scenario <NAME>`: Specifies the initial scenario (`NORMAL`, `MILD`, `MODERATE`, `SEVERE`, `PROGRESSIVE`, `HIGH_DYNAMIC_ADAPTIVE`).
  - `--video / --mp4 <PATH>`: Direct video input mode bypassing the synthetic sensor simulator.

#### `config.py`
Central dataclass hierarchy storing all operational parameters:
- `CameraConfig`: Sensor resolution ($640 \times 480$), horizontal FOV ($4.0^\circ$), vertical FOV ($3.0^\circ$), FPS ($30$), max pan/tilt velocity ($16.0^\circ/\text{s}$).
- `BeaconConfig`: Peak SWIR intensity ($230$), spot size ($12\text{ px}$), wavelength ($1550\text{ nm}$).
- `SatelliteConfig`: Model dimensions, solar panel span ($28\text{ px}$), render sprite scale ($90\text{ px}$).
- `EarthConfig`: Planetary limb visibility, world degree offsets.
- `DisturbanceConfig`: Noise types (`none`, `salt_pepper`, `gaussian`, `poisson`, `combined`), Gaussian $\sigma$, platform jitter amplitude, atmospheric haze toggles.
- `TrackingConfig`: KCF search bounding box size ($60\text{ px}$), Lucas-Kanade max features ($20$).
- `PIDConfig`: $K_p = 8.5$, $K_i = 0.25$, $K_d = 0.30$, $K_{ff} = 0.85$ (velocity feedforward gain).
- `TargetsConfig`: Operational thresholds ($\text{RMSE} \le 10\text{ px}$, target loss $< 5\%$, reacquisition time $\le 1.0\text{s}$).

---

### 3.2 Machine Learning & AI Subsystem (`ai/`)

#### `ai/xgboost_classifier.py`
Classifies environmental disturbance intensity in real-time into 4 discrete regimes mapped to state estimators.

##### Feature Vector (`FeatureVector`):
$$f = \begin{bmatrix} v & a & c_{\text{track}} & e_{\text{point}} & v_{\text{flow}} & I_{\text{beacon}} & \text{SNR} & t_{\text{loss}} & \sigma_{\text{pred}} & \text{NIS} \end{bmatrix}^T \in \mathbb{R}^{10}$$

##### Disturbance Regimes & Estimator Mapping:
1. **NORMAL (Label 0)**: Direct KCF / Visual Passthrough
2. **MILD (Label 1)**: Linear Kalman Filter (KF)
3. **MODERATE (Label 2)**: Linear Kalman Filter + Optical Flow Assistance
4. **SEVERE (Label 3)**: Unscented Kalman Filter (UKF)

##### Classification Architecture:
- Primary: Pre-trained `XGBClassifier` ($150$ gradient boosted trees, max depth $4$).
- Rule-Based Override (`_rule_based_config`): Handles explicit scenario presets and forced signal loss ($t_{\text{loss}} > 0.10\text{s}$).
- Rule-Based Fallback (`_rule_based_fallback`): Heuristic thresholds used if XGBoost binary model is missing.

---

### 3.3 Computer Vision Subsystem (`vision/`)

#### `vision/centroid.py`
Sub-pixel intensity center of mass calculation for optical beacon spot localization.

##### Mathematical Formula:
Given ROI pixel intensities $I(x, y)$ above threshold $T$:
$$\bar{x} = \frac{\sum_{x, y} x \cdot I(x,y)}{\sum_{x, y} I(x,y)}, \quad \bar{y} = \frac{\sum_{x, y} y \cdot I(x,y)}{\sum_{x, y} I(x,y)}$$

#### `vision/kcf_tracker.py`
OpenCV Kernelized Correlation Filter wrapper providing bounding-box tracking across consecutive frames.

#### `vision/optical_flow.py`
Lucas-Kanade Pyramidal Optical Flow feature velocity estimator. Tracks corner points to estimate regional pixel motion vectors $(\Delta x, \Delta y)$.

#### `vision/acquisition.py`
Multi-phase target acquisition manager:
- `SEARCHING`: Scanning frame for intensity peaks exceeding acquisition threshold.
- `ACQUIRING`: Verifying centroid temporal stability over consecutive frames.
- `TRACKING`: Active target lock confirmed.

---

### 3.4 State Estimation Subsystem (`estimation/`)

#### `estimation/kalman_filter.py`
Linear Discrete-Time Kalman Filter for constant-velocity target kinematics.
- **State Vector**: $x_k = \begin{bmatrix} x & y & v_x & v_y \end{bmatrix}^T$
- **State Transition**:
  $$F = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
- **Measurement Matrix**: $H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$

#### `estimation/ukf.py`
Unscented Kalman Filter utilizing the Merwe Scaled Unscented Transform to handle non-linear trajectories and severe maneuver dynamics.
- **Sigma Point Generation**:
  $$x^{(0)} = \hat{x}, \quad x^{(i)} = \hat{x} + \sqrt{(n + \lambda) P}_i, \quad x^{(i+n)} = \hat{x} - \sqrt{(n + \lambda) P}_i$$
  where $\lambda = \alpha^2 (n + \kappa) - n$, $\alpha = 0.8$, $\beta = 2.0$, $\kappa = 2.0$.
- **Cholesky Regularization**: Enforces positive semi-definiteness:
  $$P_{\text{reg}} = \frac{P + P^T}{2} + 10^{-5} \cdot I_n$$

#### `estimation/estimator_manager.py`
Dynamic Filter Switcher:
- Receives regime classification from AI subsystem.
- Seamlessly transfers covariance matrices and state vectors between KF and UKF.
- Computes Normalized Innovation Squared ($\text{NIS}$):
  $$\text{NIS} = r^T S^{-1} r, \quad r = z - H \hat{x}, \quad S = H P H^T + R$$

---

### 3.5 Control Engineering Subsystem (`control/`)

#### `control/pid_controller.py`
Dual-axis Pan/Tilt gimbal closed-loop controller with anti-windup integration and velocity feedforward.

##### Mathematical Formula:
Pixel errors $(e_x, e_y)$ are converted to FOV angular offsets $(\theta_x, \theta_y)$:
$$\theta_x = \frac{e_x}{W} \cdot \text{FOV}_h, \quad \theta_y = \frac{e_y}{H} \cdot \text{FOV}_v$$

Angular velocity command with feedforward:
$$u_{\text{pan}}(t) = K_p \theta_x(t) + K_i \int \theta_x(\tau) d\tau + K_d \frac{d\theta_x(t)}{dt} + K_{ff} v_{\theta, x}$$
$$u_{\text{tilt}}(t) = K_p \theta_y(t) + K_i \int \theta_y(\tau) d\tau + K_d \frac{d\theta_y(t)}{dt} + K_{ff} v_{\theta, y}$$

---

### 3.6 Target Recovery Subsystem (`recovery/`)

#### `recovery/reacquisition.py`
State machine for automated beam recovery following signal interruption or severe atmospheric cloud obscuration:
- `LOCK_MAINTAINED`: Operational tracking.
- `SPIRAL_SEARCH`: Archimedean spiral search centered on last known target trajectory projection.
- `RASTER_SCAN`: Grid search covering full field-of-view when target position uncertainty expands beyond threshold.

---

### 3.7 Simulation Subsystem (`simulation/` & `camera/`)

- `simulation/trajectories.py`: Simulates high-dynamic satellite orbits (LEO circular/elliptical, GEO drift, sinusoidal platform vibration).
- `simulation/environment.py`: Synthetic frame generator applying Gaussian noise, Salt & Pepper impulse noise, Poisson photon noise, platform jitter, and cloud haze fading.
- `simulation/beacon.py`: SWIR 1550nm laser beam Gaussian intensity profile renderer.
- `simulation/space.py`: Deep space background with star fields and Earth limb rendering.
- `simulation/satellite_renderer.py`: Satellite body, solar array, and optical aperture sprite rendering.
- `camera/sensor_model.py`: Camera projection engine translating world degree coordinates to sensor pixel space with gimbal pan/tilt feedback.

---

### 3.8 Head-to-Head Comparison Framework (`comparison/`)

- `comparison/baseline_tracker.py`: Implements standard non-adaptive fixed baseline tracker (fixed KCF/Centroid without AI adaptation or UKF switching).
- `comparison/comparison_runner.py`: Synchronous dual-execution harness executing baseline and proposed AI-adaptive trackers side-by-side on identical simulated trajectories.
- `comparison/scenario_recorder.py`: Time-series frame telemetry recorder.
- `comparison/comparison_metrics.py`: Computes comparative performance statistics:
  - $\text{RMSE} = \sqrt{\frac{1}{N}\sum_{k=1}^N \|e_k\|^2}$
  - Lock Rate ($\%$)
  - Target Loss Rate ($\%$)
  - Reacquisition Time ($\text{s}$)
  - Transition Counts ($\text{KF} \rightarrow \text{UKF}$)

---

### 3.9 Graphical User Interface (`gui/`)

- `gui/dashboard.py`: Full PySide6 main application window featuring:
  - Live virtual camera viewport with target reticle and prediction bounding boxes.
  - Telemetry scopes: Tracking error vs time, estimation uncertainty, NIS plot.
  - Manual disturbance injection controls (sliders for noise, jitter, haze).
  - Controller gain tuning (PID parameters).
- `gui/comparison_dashboard.py`: Side-by-side comparative UI presenting live synchronized visual feeds of Baseline vs Proposed systems along with metric summary cards.
- `gui/splash.py`: Main launch window for mode selection.

---

## 4. Verification & Diagnostics

- `headless_validation.py`: CLI automated runner testing scenario execution across predefined seeds.
- `comparison_diagnostic.py`: Automated diagnostic verification tool validating system lock rate, NIS bounds, and filter stability.

---

*Documentation compiled for FSOC AI Virtual Camera Tracking System codebase.*
