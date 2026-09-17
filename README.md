# FSOC AI Virtual Camera Tracking System

An advanced, AI-driven Virtual Camera Tracking System for **Free-Space Optical Communication (FSOC)** and Satellite Laser Beam Fine-Point Tracking. 

This platform simulates high-dynamic satellite orbital trajectories, severe environmental atmospheric disturbances (cloud haze, platform vibration, beacon intensity fading, sensor noise), and evaluates state-of-the-art vision and estimation algorithms (KCF, Lucas-Kanade Optical Flow, Kalman Filter, Unscented Kalman Filter) integrated with machine learning (XGBoost disturbance classification) and feedforward PID control.

---

## 🌟 Key Features

* **Real-Time Orbit & Sensor Simulation**:
  * Physics-based satellite trajectory generation (LEO, GEO, High-Dynamic, Progressive, & Severe Vibration).
  * Synthetic InGaAs / SWIR Virtual Sensor with adjustable FOV, resolution, and frame rates.
  * Realistic Space & Earth Limb background rendering (`satellite.png`, `earth.png`).
  * Dynamic beacon optical fading ($1550\text{ nm}$ SWIR) and multi-tone sinusoidal platform jitter.

* **Multi-Stage Adaptive Vision Pipeline**:
  * **Acquisition Engine**: Weighted Intensity Centroiding and multi-frame optical confirmation.
  * **Visual Tracking**: Kernelized Correlation Filter (KCF) and Lucas-Kanade Optical Flow feature tracking.
  * **Target Loss & Reacquisition**: Automated target loss detection with spiral/grid re-acquisition scanning.

* **Adaptive State Estimation**:
  * **Standard Kalman Filter (KF)**: Constant acceleration model for low-to-moderate dynamic motion.
  * **Unscented Kalman Filter (UKF)**: Non-linear state estimation handling abrupt dynamic maneuvers.
  * **NIS Consistency Monitoring**: Normalized Innovation Squared ($NIS$) monitoring for estimation filter health.

* **AI-Driven Disturbance Classification**:
  * **XGBoost Disturbance Classifier**: Classifies real-time noise and vibration severity (`NORMAL`, `MILD`, `MODERATE`, `SEVERE`, `HIGH_DYNAMIC_ADAPTIVE`).
  * Dynamically transitions tracking strategies (Direct/KCF $\rightarrow$ KF $\rightarrow$ UKF) to optimize tracking precision under extreme jitter or signal loss.

* **Feedforward PID Pan-Tilt Control**:
  * Dual-axis Pan/Tilt gimbal closed-loop control with proportional, integral, derivative gains, and zero-lag velocity feedforward (`k_ff`).

* **Dual PySide6 Desktop GUI & Benchmarking**:
  * **Interactive Main Dashboard**: Live visual canvas, trajectory scope, real-time error plots, disturbance control sliders, and manual/AI mode toggles.
  * **Head-to-Head Comparison Dashboard**: Side-by-side performance evaluation comparing the **Baseline System** against the **Proposed AI-Adaptive System**.
  * **MP4 Video Ingestion**: Benchmark direct video inputs bypassing the synthetic camera simulator.

---

## 🏗 System Architecture

```
                       ┌──────────────────────────────────────┐
                       │  Synthetic Environment & Orbit Engine │
                       │    (LEO / GEO / Vibration / Fading)  │
                       └──────────────────┬───────────────────┘
                                          │ Frame Feed (SWIR)
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Vision Processing Pipeline                            │
│  ┌───────────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐  │
│  │ Brightness Centroiding│   │ KCF Visual Tracker   │   │ Lucas-Kanade Flow   │  │
│  └───────────────────────┘   └──────────────────────┘   └─────────────────────┘  │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │ Centroid (x, y)
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            AI & Estimation Subsystem                             │
│  ┌──────────────────────────────┐          ┌──────────────────────────────────┐  │
│  │ XGBoost Classifier           │ ───────► │ Adaptive Estimator Manager       │  │
│  │ (Disturbance Severity Eval)  │          │ (Switching Direct / KF / UKF)   │  │
│  └──────────────────────────────┘          └──────────────────────────────────┘  │
└─────────────────────────────────────────┬────────────────────────────────────────┘
                                          │ State Estimate [x, y, vx, vy, ax, ay]
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Gimbal Control Subsystem                             │
│       Feedforward PID Controller (Kp, Ki, Kd, K_ff) ──► Camera Pan/Tilt Motion   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory & File Structure

```
SIH/
├── main.py                      # Main application entry point (GUI launcher & CLI)
├── config.py                    # Central system configuration & hyperparameters
├── headless_validation.py       # Automated headless validation execution script
├── comparison_diagnostic.py     # Diagnostic validator for performance comparison runs
│
├── ai/                          # Machine Learning Subsystem
│   └── xgboost_classifier.py    # XGBoost model training & dynamic disturbance inference
│
├── estimation/                  # State Estimation Subsystem
│   ├── estimator_manager.py     # Dynamic switching manager between KF & UKF
│   ├── kalman_filter.py         # Standard Linear Kalman Filter implementation
│   └── ukf.py                   # Unscented Kalman Filter implementation
│
├── vision/                      # Computer Vision Subsystem
│   ├── acquisition.py           # Target detection & acquisition manager
│   ├── centroid.py              # Sub-pixel weighted intensity centroid algorithm
│   ├── kcf_tracker.py           # Kernelized Correlation Filter tracker wrapper
│   └── optical_flow.py          # Lucas-Kanade Optical Flow tracking engine
│
├── simulation/                  # Physics & Space Simulation Subsystem
│   ├── beacon.py                # SWIR optical beacon model & fading simulation
│   ├── environment.py           # Atmospheric noise, haze, & jitter dynamics
│   ├── satellite_renderer.py    # Synthetic satellite visual renderer
│   ├── space.py                 # Background space rendering (stars & Earth limb)
│   └── trajectories.py          # Satellite trajectory equations (LEO, GEO, Jitter)
│
├── control/                     # Control Engineering Subsystem
│   └── pid_controller.py        # Proportional-Integral-Derivative + Velocity Feedforward
│
├── recovery/                    # Target Recovery Subsystem
│   └── reacquisition.py         # Spiral & raster scan reacquisition state machine
│
├── comparison/                  # Evaluation Framework
│   ├── baseline_tracker.py      # Standard non-adaptive baseline system implementation
│   ├── comparison_metrics.py     # Metric calculator (RMSE, Loss %, Lock Rate, Reacq Time)
│   ├── comparison_runner.py     # Head-to-head simulation runner
│   ├── result_model.py          # Structured results data models
│   └── scenario_recorder.py     # Real-time state recorder for evaluation runs
│
├── camera/                      # Sensor Subsystem
│   └── sensor_model.py          # Virtual InGaAs camera projection & FOV conversion
│
├── gui/                         # Graphical User Interfaces (PySide6)
│   ├── dashboard.py             # Interactive main tracking dashboard
│   ├── comparison_dashboard.py  # Head-to-head validation comparison UI
│   └── splash.py                # Mode selection splash screen
│
├── validation/                  # Statistical Verification
│   └── nis.py                   # Normalized Innovation Squared (NIS) consistency metrics
│
├── benchmark/                   # Performance Metrics & External Inputs
│   ├── metrics.py               # Target accuracy metrics calculation
│   └── video_input.py           # Video file stream reader (.mp4 ingest)
│
└── assets/                      # Graphic Assets
    ├── earth.png                # High-res Earth limb graphic
    └── satellite.png            # Satellite model texture
```

---

## 🛠 Installation & Requirements

### Prerequisites
* Python **3.9+** (Python 3.10 / 3.11 recommended)

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd SIH
   ```

2. **Create and Activate a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install numpy scipy opencv-python PySide6 xgboost matplotlib pillow
   ```

---

## 🚀 Usage Guide

### 1. Launch Main Application (GUI)
Run the application launcher to open the interactive splash screen:
```bash
python main.py
```
From the splash screen, select between **Simulation Mode**, **Benchmark Mode**, or **Head-to-Head Comparison Mode**.

### 2. Direct Launch into Comparison Mode
To launch directly into the Head-to-Head Comparison Dashboard:
```bash
python main.py --compare --scenario SEVERE
```
*Available scenarios*: `NORMAL`, `MILD`, `MODERATE`, `SEVERE`, `PROGRESSIVE`, `HIGH_DYNAMIC_ADAPTIVE`.

### 3. Ingest MP4 Video Input
To run the visual tracking pipeline against a custom video file (`.mp4`):
```bash
python main.py --video path/to/video.mp4
```

### 4. Train the AI Disturbance Classifier
To train (or retrain) the XGBoost Disturbance Classifier on synthetic disturbance feature vectors:
```bash
python main.py --train-xgb
```
This generates and saves the model artifact to `models/xgboost_disturbance.json`.

### 5. Run Headless Diagnostics & Benchmarks (CLI)
To run automated validation without opening any graphical user interface:
```bash
python headless_validation.py
```
Or run the comparison diagnostic test suite directly:
```bash
python comparison_diagnostic.py
```

---

## ⚙️ Configuration (`config.py`)

All core parameters are organized in dataclasses within `config.py` for easy customization:

| Config Section | Parameters | Description |
| :--- | :--- | :--- |
| `CameraConfig` | `width=640`, `height=480`, `fps=30`, `pan_speed_max=16.0` | Camera sensor parameters & gimbal limits |
| `BeaconConfig` | `intensity=230`, `default_size_px=12`, `wavelength_nm=1550` | SWIR laser beacon optical properties |
| `DisturbanceConfig` | `noise_type`, `gaussian_sigma`, `jitter_max_px`, `haze` | Atmospheric noise & platform jitter setup |
| `PIDConfig` | `kp=8.5`, `ki=0.25`, `kd=0.30`, `k_ff=0.85` | Pan/Tilt closed-loop controller gains |
| `TargetsConfig` | `tracking_error_px=10.0`, `target_loss_pct=5.0` | Problem Statement operational targets |

---

## 📊 Evaluation & Verification Metrics

The platform evaluates system performance across key operational metrics:

* **RMSE (Pixel / Angular Error)**: Root Mean Square Error of beam alignment relative to target centroid.
* **Mean Error (px)**: Average tracking offset across the trajectory duration.
* **Lock Rate (%)**: Percentage of total runtime maintaining active lock within threshold ($\le 10\text{ px}$).
* **Loss Rate (%)**: Percentage of frames where target signal was completely lost ($> 5\text{s}$ or out-of-frame).
* **Reacquisition Time**: Time taken to detect and re-lock target after signal loss ($\le 1.0\text{s}$ goal).
* **NIS Consistency**: Normalized Innovation Squared checks validating estimation filter variance ($95\%$ confidence bounds).

---

## 📜 License

This repository is maintained for Smart India Hackathon (SIH) / Free-Space Optical Communication (FSOC) research and development.
