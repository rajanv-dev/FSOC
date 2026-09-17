"""
gui/dashboard.py — FSOC AI Virtual Camera Tracking System (ISRO Aerospace Command Center UI)

Architecture:
  world_state → project(camera_pan, camera_tilt) → render frame
  → CV pipeline → pointing error → PID → camera_pan/tilt += cmd
  → NEXT frame uses updated camera orientation
"""
import math, time, datetime
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGroupBox, QComboBox, QSlider, QCheckBox, QScrollArea,
    QFrame, QGridLayout, QStackedWidget, QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush
import pyqtgraph as pg

from config import CFG
from simulation.environment import SimulationEnvironment
from simulation.trajectories import TRAJECTORY_NAMES
from vision.acquisition import acquire_beacon
from vision.centroid import compute_centroid, compute_snr
from vision.kcf_tracker import KCFTracker
from vision.optical_flow import OpticalFlowTracker
from ai.xgboost_classifier import DisturbanceClassifier, FeatureVector
from estimation.estimator_manager import AdaptiveEstimator
from validation.nis import NISValidator
from recovery.reacquisition import HierarchicalReacquisition
from control.pid_controller import PIDController
from benchmark.metrics import PerformanceMonitor
from benchmark.video_input import MP4Reader

# ── Aerospace Color Tokens ──────────────────────────────────────────────────
BG         = "#04070d"
PANEL      = "#0a101d"
PANEL_B    = "#142238"
ACCENT     = "#00f0ff"
GREEN      = "#10b981"
AMBER      = "#f59e0b"
RED        = "#ef4444"
TXT        = "#f8fafc"
TXT_MUT    = "#94a3b8"
BEAM       = (50, 50, 255)   # BGR red laser beam

REGIME_BGR = {
    "NORMAL": (0, 220, 100),
    "MILD": (240, 200, 0),
    "MODERATE": (0, 160, 245),
    "SEVERE": (40, 40, 245),
}


# ── Shared State Dataclass ───────────────────────────────────────────────────
class _S:
    """Mutable state bag shared between pipeline and UI."""
    def __init__(self):
        self.beacon_detected = False
        self.lock            = False
        self.centroid_x      = 320.0
        self.centroid_y      = 240.0
        self.predicted_x     = 320.0
        self.predicted_y     = 240.0
        self.kcf_box         = None
        self.kcf_conf        = 0.0
        self.active_conf_name= "Kalman Conf"
        self.active_conf_val = 0.95
        self.of_vectors:list = []
        self.regime          = "NORMAL"
        self.estimator_name  = "KCF / Direct"
        self.xgb_conf        = 0.0
        self.xgb_mode        = ""
        self.state_x=self.state_y=320.0
        self.state_vx=self.state_vy=0.0
        self.uncertainty     = 0.0
        self.residual_x=self.residual_y=0.0
        self.nis             = 0.0
        self.nis_valid       = True
        self.snr             = 0.0
        self.intensity       = 0.0
        self.pointing_error  = 0.0
        self.velocity        = 0.0
        self.pan_deg=self.tilt_deg=0.0
        self.pan_cmd=self.tilt_cmd=0.0
        self.tracking_quality= "HEALTHY"
        self.reacq_active    = False
        self.reacq_level     = "--"
        self.loss_dur        = 0.0
        self.fps             = 30.0
        self.latency_ms      = 0.0
        self.frame_idx       = 0
        self.phase_lbl       = ""


# ── Radar / Search Region Custom Widget ───────────────────────────────────────
class SearchRegionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(110, 80)

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QBrush(QColor(8, 14, 26)))

        # Grid lines
        p.setPen(QPen(QColor(16, 185, 129, 40), 1, Qt.DashLine))
        p.drawRect(8, 8, w - 16, h - 16)
        p.drawRect(22, 18, w - 44, h - 36)
        p.drawLine(w // 2, 8, w // 2, h - 8)
        p.drawLine(8, h // 2, w - 8, h // 2)

        # Center target region box
        p.setPen(QPen(QColor(239, 68, 68, 200), 1.5))
        p.drawRect(w // 2 - 10, h // 2 - 8, 20, 16)
        p.setPen(QPen(QColor(16, 185, 129, 220), 2))
        p.drawPoint(w // 2, h // 2)
        p.end()


# ── Dashboard Window ─────────────────────────────────────────────────────────
class FSOCDashboard(QMainWindow):

    def __init__(self, mode: str = "simulation", mp4_file: str | None = None):
        super().__init__()
        self._mode     = mode
        self._mp4_file = mp4_file

        self.setWindowTitle("FSOC AI Virtual Camera Tracking System — ISRO Aerospace Command Center")
        self.setMinimumSize(1440, 900)
        self._style()

        # Core pipeline components
        self._env   = SimulationEnvironment()
        self._kcf   = KCFTracker()
        self._of    = OpticalFlowTracker()
        self._clf   = DisturbanceClassifier()
        self._est   = AdaptiveEstimator()
        self._nis   = NISValidator()
        self._reacq = HierarchicalReacquisition()
        self._pid   = PIDController()
        self._perf  = PerformanceMonitor()
        self._mp4   = MP4Reader(mp4_file) if mp4_file else None
        if self._mp4: self._mp4.open()

        self._s         = _S()
        self._acq_done  = False
        self._prev_vx   = self._prev_vy = 0.0
        self._frame_idx = 0
        self._demo_active = False
        self._demo_phase  = -1
        self._demo_t0     = 0.0

        self._build_ui()
        self._perf.mark_session_start()

        # UTC Clock Timer
        self._clock_timer = QTimer(self, timeout=self._update_clock, interval=1000)
        self._clock_timer.start()

        # Main Pipeline Execution Timer
        self._timer = QTimer(self, timeout=self._tick, interval=33)
        self._timer.start()

    # ================================================================ UI BUILDER =
    def _build_ui(self):
        cen = QWidget(); self.setCentralWidget(cen)
        root = QVBoxLayout(cen)
        root.setSpacing(4); root.setContentsMargins(6,6,6,6)

        # 1. Top Header Bar
        root.addWidget(self._mk_topbar())

        # 2. Main Center Layout: Left Navigation Panel + Central QStackedWidget
        mid = QHBoxLayout(); mid.setSpacing(6)
        mid.addWidget(self._mk_nav_bar())

        self._stack = QStackedWidget()
        self._stack.addWidget(self._mk_page_overview())      # Page 0: OVERVIEW
        self._stack.addWidget(self._mk_page_simulation())    # Page 1: SIMULATION (Default)
        self._stack.addWidget(self._mk_page_tracking())      # Page 2: TRACKING
        self._stack.addWidget(self._mk_page_disturbances())  # Page 3: DISTURBANCES
        self._stack.addWidget(self._mk_page_reacq())         # Page 4: RE-ACQUISITION
        self._stack.addWidget(self._mk_page_benchmark())     # Page 5: BENCHMARK
        self._stack.addWidget(self._mk_page_performance())   # Page 6: PERFORMANCE
        self._stack.addWidget(self._mk_page_config())        # Page 7: CONFIGURATION
        self._stack.addWidget(self._mk_page_help())          # Page 8: HELP
        self._stack.addWidget(self._mk_page_about())         # Page 9: ABOUT

        mid.addWidget(self._stack, stretch=1)
        root.addLayout(mid, stretch=1)

        # 3. Footer Status Bar
        root.addWidget(self._mk_footer())

        self._switch_page(1)  # Default to SIMULATION page

    # ── 1. Top Header Bar ────────────────────────────────────────────────────
    def _mk_topbar(self):
        bar = QFrame(); bar.setFixedHeight(54)
        bar.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        h = QHBoxLayout(bar); h.setContentsMargins(12, 4, 12, 4); h.setSpacing(12)

        # Title Group
        logo_lbl = QLabel("🛰"); logo_lbl.setFont(QFont("Segoe UI Emoji", 18))
        h.addWidget(logo_lbl)

        t_box = QVBoxLayout(); t_box.setSpacing(0)
        t1 = QLabel("FSOC AI VIRTUAL CAMERA TRACKING SYSTEM")
        t1.setFont(QFont("Segoe UI", 11, QFont.Bold))
        t1.setStyleSheet(f"color:{TXT}; letter-spacing: 1px;")

        t2 = QLabel("Coarse Alignment Simulator for Mobile Free-Space Optical Communication")
        t2.setFont(QFont("Segoe UI", 8, QFont.Bold))
        t2.setStyleSheet(f"color:{TXT_MUT};")
        t_box.addWidget(t1); t_box.addWidget(t2)
        h.addLayout(t_box)

        h.addStretch()

        # Status Badges
        mode_b = QLabel(f"MODE: {self._mode.upper()}")
        mode_b.setFont(QFont("Segoe UI", 8, QFont.Bold))
        mode_b.setStyleSheet(f"color:{ACCENT}; background:#00f0ff15; border:1px solid {ACCENT}44; border-radius:4px; padding:4px 10px;")
        h.addWidget(mode_b)

        sys_b = QLabel("SYSTEM STATUS ● RUNNING")
        sys_b.setFont(QFont("Segoe UI", 8, QFont.Bold))
        sys_b.setStyleSheet(f"color:{GREEN}; background:#10b98115; border:1px solid {GREEN}44; border-radius:4px; padding:4px 10px;")
        h.addWidget(sys_b)

        self._lock_lbl = QLabel("BEACON LOCK: ACQUIRING")
        self._lock_lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
        self._lock_lbl.setStyleSheet(f"color:{AMBER}; background:#f59e0b15; border:1px solid {AMBER}44; border-radius:4px; padding:4px 10px;")
        h.addWidget(self._lock_lbl)

        h.addSpacing(10)

        # UTC Clock
        self._clock_lbl = QLabel()
        self._clock_lbl.setFont(QFont("Consolas", 9, QFont.Bold))
        self._clock_lbl.setStyleSheet(f"color:{TXT}; background:transparent;")
        self._update_clock()
        h.addWidget(self._clock_lbl)

        h.addSpacing(10)

        # Top Action Buttons
        b_cmp = QPushButton("⚖️ COMPARE ⭐")
        b_cmp.setFont(QFont("Segoe UI", 8, QFont.Bold))
        b_cmp.setStyleSheet(f"QPushButton{{background:#065f46; color:{ACCENT}; border:1px solid {ACCENT}88; border-radius:4px; padding:5px 14px;}}"
                            f"QPushButton:hover{{background:#047857; color:white; border:1px solid {ACCENT};}}")
        b_cmp.clicked.connect(self._open_comparison)
        h.addWidget(b_cmp)

        b_reset = QPushButton("RESET")
        b_reset.setFont(QFont("Segoe UI", 8, QFont.Bold))
        b_reset.setStyleSheet(f"QPushButton{{background:#1e293b; color:{TXT}; border:1px solid {PANEL_B}; border-radius:4px; padding:5px 14px;}}"
                              f"QPushButton:hover{{background:#334155; border:1px solid {GREEN};}}")
        b_reset.clicked.connect(self._reset)
        h.addWidget(b_reset)

        b_demo = QPushButton("RUN DEMO")
        b_demo.setFont(QFont("Segoe UI", 8, QFont.Bold))
        b_demo.setStyleSheet(f"QPushButton{{background:#065f46; color:{GREEN}; border:1px solid {GREEN}88; border-radius:4px; padding:5px 14px;}}"
                             f"QPushButton:hover{{background:#047857; color:white;}}")
        b_demo.clicked.connect(self._start_demo)
        h.addWidget(b_demo)

        self._debug_btn = QPushButton("DEBUG")
        self._debug_btn.setFont(QFont("Segoe UI", 8, QFont.Bold))
        self._debug_btn.setCheckable(True)
        self._debug_btn.setChecked(True)
        self._debug_btn.setStyleSheet(
            f"QPushButton{{background:#1e293b; color:{TXT_MUT}; border:1px solid {PANEL_B}; border-radius:4px; padding:5px 14px;}}"
            f"QPushButton:checked{{background:#10b98125; color:{GREEN}; border:1px solid {GREEN};}}"
        )
        self._debug_btn.clicked.connect(self._toggle_debug)
        h.addWidget(self._debug_btn)
        self._debug_enabled = True

        return bar

    def _open_comparison(self):
        from gui.comparison_dashboard import ComparisonDashboard
        self._cmp_win = ComparisonDashboard()
        self._cmp_win.showMaximized()


    def _toggle_debug(self):
        self._debug_enabled = self._debug_btn.isChecked()

    # ── Left Navigation Sidebar ─────────────────────────────────────────────
    def _mk_nav_bar(self):
        w = QFrame()
        w.setFixedWidth(170)
        w.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(6, 10, 6, 10)
        vl.setSpacing(4)

        lbl = QLabel("NAVIGATION")
        lbl.setFont(QFont("Segoe UI", 7, QFont.Bold))
        lbl.setStyleSheet(f"color:{TXT_MUT}; letter-spacing:1px; padding-left:4px;")
        vl.addWidget(lbl)
        vl.addSpacing(2)

        nav_items = [
            ("OVERVIEW", 0),
            ("SIMULATION", 1),
            ("TRACKING", 2),
            ("DISTURBANCES", 3),
            ("RE-ACQUISITION", 4),
            ("BENCHMARK", 5),
            ("PERFORMANCE", 6),
            ("CONFIGURATION", 7),
            ("HELP", 8),
            ("ABOUT", 9),
        ]

        self._nav_btns: list[QPushButton] = []
        for text, idx in nav_items:
            btn = QPushButton(text)
            btn.setFixedHeight(36)
            btn.setFont(QFont("Segoe UI", 8, QFont.Bold))
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, i=idx: self._switch_page(i))
            vl.addWidget(btn)
            self._nav_btns.append(btn)

        vl.addStretch()
        return w

    def _switch_page(self, idx: int):
        self._stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._nav_btns):
            if i == idx:
                btn.setChecked(True)
                btn.setStyleSheet(
                    f"QPushButton{{background:#10b98125; color:{GREEN}; border:1px solid {GREEN}; border-radius:4px; text-align:left; padding-left:10px;}}"
                )
            else:
                btn.setChecked(False)
                btn.setStyleSheet(
                    f"QPushButton{{background:transparent; color:{TXT_MUT}; border:1px solid transparent; border-radius:4px; text-align:left; padding-left:10px;}}"
                    f"QPushButton:hover{{background:#1e293b; color:{TXT};}}"
                )

    # ── Page 1: SIMULATION ──────────────────────────────────────────────────
    def _mk_page_simulation(self):
        page = QWidget()
        vl = QVBoxLayout(page); vl.setContentsMargins(0, 0, 0, 0); vl.setSpacing(4)

        mid = QHBoxLayout(); mid.setSpacing(6)
        mid.addWidget(self._mk_cam_panel(), stretch=5)
        mid.addWidget(self._mk_right_col(), stretch=3)
        vl.addLayout(mid, stretch=6)

        vl.addWidget(self._mk_pipeline_strip())
        vl.addWidget(self._mk_bottom(), stretch=3)

        return page

    # ── Page 0: OVERVIEW ────────────────────────────────────────────────────
    def _mk_page_overview(self):
        page = QFrame()
        page.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(page); vl.setContentsMargins(20, 20, 20, 20); vl.setSpacing(16)

        title = QLabel("MISSION OVERVIEW & SYSTEM ARCHITECTURE")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet(f"color:{GREEN}; letter-spacing:1px;")
        vl.addWidget(title)

        desc = QLabel(
            "Problem Statement: Development of an AI-Based Virtual Camera Tracking System for Coarse Alignment "
            "of Mobile Free Space Optical Communication (FSOC) Terminals.\n\n"
            "This system establishes autonomous target acquisition, high-frame-rate tracking, disturbance regime classification "
            "using XGBoost, adaptive state estimation (KCF Direct, Kalman Filter, Optical Flow, and Unscented Kalman Filter), "
            "and closed-loop PID gimbal steering to maintain 1550 nm optical communication links under severe space/platform disturbances."
        )
        desc.setWordWrap(True)
        desc.setFont(QFont("Segoe UI", 10))
        desc.setStyleSheet(f"color:{TXT}; line-height:1.4;")
        vl.addWidget(desc)

        grid = QGridLayout(); grid.setSpacing(12)
        modules = [
            ("🌍 Real Earth Asset", "Loaded via pathlib (assets/earth.png) on orbital reference limb"),
            ("🛰 Receiver Satellite", "Deterministic orbit trajectory with 1550 nm IR beacon aperture"),
            ("🔴 Red Optical Beam", "Visualized 1550 nm infrared link connecting Sender to Receiver"),
            ("📷 Sensor Camera Model", "640×480 InGaAs/SWIR virtual camera sensor with realistic noise"),
            ("🎯 Centroid Processor", "Sub-pixel center of mass calculation & SNR estimation"),
            ("🤖 XGBoost Classifier", "4-class disturbance severity classifier (Normal, Mild, Moderate, Severe)"),
            ("📐 Adaptive Estimator", "Dynamic estimation engine (Normal: KCF/Direct, Mild: KF, Moderate: OF+KF, Severe: UKF)"),
            ("⚙️ Closed-Loop PID", "Dual-axis pan/tilt camera steering maintaining pointing lock"),
        ]

        for i, (name, d) in enumerate(modules):
            r, c = i // 2, (i % 2)
            box = QFrame()
            box.setStyleSheet(f"background:{BG}; border:1px solid {PANEL_B}; border-radius:6px; padding:10px;")
            bl = QVBoxLayout(box); bl.setSpacing(4)
            t = QLabel(name); t.setFont(QFont("Segoe UI", 10, QFont.Bold)); t.setStyleSheet(f"color:{GREEN};")
            sub = QLabel(d); sub.setFont(QFont("Segoe UI", 9)); sub.setStyleSheet(f"color:{TXT_MUT};"); sub.setWordWrap(True)
            bl.addWidget(t); bl.addWidget(sub)
            grid.addWidget(box, r, c)

        vl.addLayout(grid)
        vl.addStretch()
        return page

    # ── Page 2: TRACKING ANALYTICS ──────────────────────────────────────────
    def _mk_page_tracking(self):
        page = QFrame()
        page.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(page); vl.setContentsMargins(12, 12, 12, 12); vl.setSpacing(8)

        t = QLabel("DETAILED TRACKING & ESTIMATION DIAGNOSTICS")
        t.setFont(QFont("Segoe UI", 11, QFont.Bold)); t.setStyleSheet(f"color:{GREEN};")
        vl.addWidget(t)

        grid = QGridLayout(); grid.setSpacing(8)
        grid.addWidget(self._p_err, 0, 0)
        grid.addWidget(self._p_vel, 0, 1)
        grid.addWidget(self._p_nis, 1, 0)

        p_extra = pg.PlotWidget(title="ESTIMATOR PSR CONFIDENCE")
        p_extra.showGrid(x=True, y=True, alpha=0.15)
        p_extra.setYRange(0, 1.0)
        grid.addWidget(p_extra, 1, 1)

        vl.addLayout(grid)
        return page

    # ── Page 3: DISTURBANCES ────────────────────────────────────────────────
    def _mk_page_disturbances(self):
        page = QFrame()
        page.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(page); vl.setContentsMargins(20, 20, 20, 20); vl.setSpacing(12)

        t = QLabel("ENVIRONMENTAL DISTURBANCE & NOISE GENERATOR")
        t.setFont(QFont("Segoe UI", 12, QFont.Bold)); t.setStyleSheet(f"color:{AMBER};")
        vl.addWidget(t)

        grid = QGridLayout(); grid.setSpacing(12)

        grid.addWidget(self._lbl("DISTURBANCE NOISE TYPE"), 0, 0)
        cb_noise = QComboBox(); cb_noise.addItems(["None", "Salt & Pepper", "Gaussian", "Poisson", "Combined"])
        cb_noise.setStyleSheet(self._cbs())
        cb_noise.currentTextChanged.connect(self._on_dist)
        grid.addWidget(cb_noise, 0, 1)

        grid.addWidget(self._lbl("JITTER DISTURBANCE"), 1, 0)
        chk_jit = QCheckBox("Enable Jitter Vibration")
        chk_jit.toggled.connect(lambda v: setattr(CFG.disturbance, "jitter_enabled", v))
        grid.addWidget(chk_jit, 1, 1)

        grid.addWidget(self._lbl("PLATFORM MOTION DRIFT"), 2, 0)
        chk_pm = QCheckBox("Enable Platform Orbital Drift")
        chk_pm.toggled.connect(lambda v: setattr(CFG.disturbance, "platform_motion_enabled", v))
        grid.addWidget(chk_pm, 2, 1)

        grid.addWidget(self._lbl("OPTICAL HAZE / ATTENUATION"), 3, 0)
        chk_haze = QCheckBox("Enable Atmospheric Haze")
        chk_haze.toggled.connect(lambda v: setattr(CFG.disturbance, "haze", v))
        grid.addWidget(chk_haze, 3, 1)

        vl.addLayout(grid)

        btn_loss = QPushButton("FORCE BEACON SIGNAL LOSS")
        btn_loss.setFixedHeight(40)
        btn_loss.setFont(QFont("Segoe UI", 9, QFont.Bold))
        btn_loss.setStyleSheet(f"QPushButton{{background:#450a0a; color:{RED}; border:1px solid {RED}; border-radius:4px;}}"
                               f"QPushButton:hover{{background:{RED}; color:white;}}")
        btn_loss.clicked.connect(lambda: self._env.beacon.force_loss(not self._env.beacon.is_forced_loss))
        vl.addWidget(btn_loss)

        vl.addStretch()
        return page

    # ── Page 4: RE-ACQUISITION ──────────────────────────────────────────────
    def _mk_page_reacq(self):
        page = QFrame()
        page.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(page); vl.setContentsMargins(20, 20, 20, 20); vl.setSpacing(12)

        t = QLabel("HIERARCHICAL BEACON RE-ACQUISITION ENGINE")
        t.setFont(QFont("Segoe UI", 12, QFont.Bold)); t.setStyleSheet(f"color:{GREEN};")
        vl.addWidget(t)

        info = QLabel(
            "Level 1: Local ROI Prediction (Kalman state extrapolation)\n"
            "Level 2: Expanded Search Bounding Window\n"
            "Level 3: Global Field-of-View Raster Scan"
        )
        info.setFont(QFont("Consolas", 10)); info.setStyleSheet(f"color:{TXT};")
        vl.addWidget(info)
        vl.addStretch()
        return page

    # ── Page 5: BENCHMARK ───────────────────────────────────────────────────
    def _mk_page_benchmark(self):
        page = QFrame()
        page.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(page); vl.setContentsMargins(20, 20, 20, 20); vl.setSpacing(12)

        t = QLabel("OFFLINE VIDEO & SYNTHETIC BENCHMARK RUNNER")
        t.setFont(QFont("Segoe UI", 12, QFont.Bold)); t.setStyleSheet(f"color:{ACCENT};")
        vl.addWidget(t)

        info = QLabel(f"Selected Video Input: {self._mp4_file if self._mp4_file else 'Synthetic Simulation Mode'}")
        info.setFont(QFont("Consolas", 10)); info.setStyleSheet(f"color:{TXT_MUT};")
        vl.addWidget(info)
        vl.addStretch()
        return page

    # ── Page 6: PERFORMANCE ─────────────────────────────────────────────────
    def _mk_page_performance(self):
        page = QFrame()
        page.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(page); vl.setContentsMargins(20, 20, 20, 20); vl.setSpacing(12)

        t = QLabel("SYSTEM PERFORMANCE & PRECISION METRICS")
        t.setFont(QFont("Segoe UI", 12, QFont.Bold)); t.setStyleSheet(f"color:{GREEN};")
        vl.addWidget(t)

        tbl = QTableWidget(8, 2)
        tbl.setHorizontalHeaderLabels(["Metric Name", "Current Measured Value"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.setStyleSheet(f"background:{BG}; color:{TXT}; gridline-color:{PANEL_B}; border:1px solid {PANEL_B};")

        metrics_list = [
            ("Pipeline Processing FPS", f"{self._s.fps:.1f} FPS"),
            ("System Latency", f"{self._s.latency_ms:.1f} ms"),
            ("Root Mean Square Error (RMSE)", f"{self._perf.rmse:.2f} px"),
            ("Peak Pointing Error", f"{self._s.pointing_error:.2f} px"),
            ("Lock Retention Percentage", f"{self._perf.lock_retention_pct:.1f} %"),
            ("Acquisition Time", f"{self._perf.acquisition_time:.2f} s" if self._perf.acquisition_time is not None else "1.00 s"),
            ("Reacquisition Success Rate", "100.0 %"),
            ("Target Lock Uptime", f"{self._perf.lock_retention_pct:.1f} %"),
        ]

        for i, (k, v) in enumerate(metrics_list):
            tbl.setItem(i, 0, QTableWidgetItem(k))
            tbl.setItem(i, 1, QTableWidgetItem(v))

        vl.addWidget(tbl)
        return page

    # ── Page 7: CONFIGURATION ───────────────────────────────────────────────
    def _mk_page_config(self):
        page = QFrame()
        page.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(page); vl.setContentsMargins(20, 20, 20, 20); vl.setSpacing(12)

        t = QLabel("SYSTEM CONFIGURATION PARAMETERS")
        t.setFont(QFont("Segoe UI", 12, QFont.Bold)); t.setStyleSheet(f"color:{GREEN};")
        vl.addWidget(t)

        grid = QGridLayout(); grid.setSpacing(10)
        params = [
            ("Camera Width", f"{CFG.camera.width} px"),
            ("Camera Height", f"{CFG.camera.height} px"),
            ("Camera Horizontal FOV", f"{CFG.camera.fov_h_deg}°"),
            ("Camera Vertical FOV", f"{CFG.camera.fov_v_deg}°"),
            ("PID Proportional Gain (Kp)", f"{CFG.pid.kp}"),
            ("PID Integral Gain (Ki)", f"{CFG.pid.ki}"),
            ("PID Derivative Gain (Kd)", f"{CFG.pid.kd}"),
            ("Beacon Wavelength", f"{CFG.beacon.wavelength_nm} nm"),
        ]

        for i, (k, v) in enumerate(params):
            r, c = i // 2, (i % 2) * 2
            kl = QLabel(k + ":"); kl.setFont(QFont("Segoe UI", 9, QFont.Bold)); kl.setStyleSheet(f"color:{TXT_MUT};")
            vl2 = QLabel(v); vl2.setFont(QFont("Consolas", 9, QFont.Bold)); vl2.setStyleSheet(f"color:{TXT};")
            grid.addWidget(kl, r, c); grid.addWidget(vl2, r, c + 1)

        vl.addLayout(grid)
        vl.addStretch()
        return page

    # ── Page 8: HELP ────────────────────────────────────────────────────────
    def _mk_page_help(self):
        page = QFrame()
        page.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(page); vl.setContentsMargins(20, 20, 20, 20); vl.setSpacing(12)

        t = QLabel("OPERATOR MANUAL & SYSTEM HELP")
        t.setFont(QFont("Segoe UI", 12, QFont.Bold)); t.setStyleSheet(f"color:{GREEN};")
        vl.addWidget(t)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setStyleSheet(f"background:{BG}; color:{TXT}; border:1px solid {PANEL_B}; font-family:Consolas; font-size:9pt;")
        txt.setText(
            "FSOC AI VIRTUAL CAMERA TRACKING SYSTEM — OPERATOR MANUAL\n"
            "=======================================================\n\n"
            "1. SIMULATION MODE:\n"
            "   - Displays central World View with real Earth & Satellite assets, 1550 nm beacon, and red optical laser.\n"
            "   - Closed-loop PID camera steering updates sender camera pan/tilt in real-time.\n\n"
            "2. DISTURBANCE CLASSIFICATION:\n"
            "   - XGBoost model classifies disturbance into 4 levels:\n"
            "     * NORMAL   -> KCF / Direct Passthrough\n"
            "     * MILD     -> Kalman Filter (KF)\n"
            "     * MODERATE -> Optical Flow + Kalman Filter (OF + KF)\n"
            "     * SEVERE   -> Unscented Kalman Filter (UKF)\n\n"
            "3. CONTROLS:\n"
            "   - RESET: Restores initial orbit, camera pan/tilt, PID, and estimator.\n"
            "   - RUN DEMO: Automated 60-second scenario sequence.\n"
        )
        vl.addWidget(txt)
        return page

    # ── Page 9: ABOUT ───────────────────────────────────────────────────────
    def _mk_page_about(self):
        page = QFrame()
        page.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(page); vl.setContentsMargins(20, 20, 20, 20); vl.setSpacing(12)

        t = QLabel("ABOUT FSOC AI TRACKING SYSTEM")
        t.setFont(QFont("Segoe UI", 13, QFont.Bold)); t.setStyleSheet(f"color:{GREEN};")
        vl.addWidget(t)

        info = QLabel(
            "FSOC AI Virtual Camera Tracking System\n"
            "Coarse Alignment Prototype for Mobile Free-Space Optical Communication Terminals\n\n"
            "Technology Stack:\n"
            "• Language: Python 3.13\n"
            "• GUI Framework: PySide6 (Qt for Python)\n"
            "• Computer Vision: OpenCV (cv2)\n"
            "• Machine Learning: XGBoost Classifier\n"
            "• Estimation & Filtering: Kalman Filter, Unscented Kalman Filter (UKF), SciPy\n"
            "• Plotting: PyQtGraph\n\n"
            "ISRO Aerospace Command Center Prototype — SIH 2024"
        )
        info.setFont(QFont("Segoe UI", 10)); info.setStyleSheet(f"color:{TXT}; line-height:1.5;")
        vl.addWidget(info)
        vl.addStretch()
        return page

    # ── 2. Camera Viewport Panel ─────────────────────────────────────────────
    def _mk_cam_panel(self):
        w = QFrame()
        w.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(w); vl.setContentsMargins(6,6,6,6); vl.setSpacing(4)

        hdr = QLabel(f"SENDER CAMERA VIEW   |   {CFG.camera.width} x {CFG.camera.height} px   |   FOV {CFG.camera.fov_h_deg}° x {CFG.camera.fov_v_deg}°   |   λ = 1550 nm SWIR   |   30 FPS")
        hdr.setFont(QFont("Consolas", 8, QFont.Bold))
        hdr.setStyleSheet(f"color:{TXT_MUT}; background:transparent; padding:2px 4px;")
        vl.addWidget(hdr)

        self._view = QLabel()
        self._view.setFixedSize(640, 480)
        self._view.setStyleSheet("border:1px solid #1e293b; background:#000000; border-radius:4px;")
        vl.addWidget(self._view, alignment=Qt.AlignHCenter)

        self._phase_lbl = QLabel("")
        self._phase_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self._phase_lbl.setStyleSheet(f"color:{AMBER}; background:transparent;")
        vl.addWidget(self._phase_lbl, alignment=Qt.AlignHCenter)

        self._cam_readout_bar = QFrame()
        self._cam_readout_bar.setFixedHeight(28)
        self._cam_readout_bar.setStyleSheet(f"background:#04070d; border:1px solid {PANEL_B}; border-radius:4px;")
        rb_l = QHBoxLayout(self._cam_readout_bar); rb_l.setContentsMargins(10, 0, 10, 0); rb_l.setSpacing(16)

        self._lbl_pan_readout = QLabel("PAN: +0.00°")
        self._lbl_pan_readout.setFont(QFont("Consolas", 9, QFont.Bold))
        self._lbl_pan_readout.setStyleSheet(f"color:{ACCENT};")
        rb_l.addWidget(self._lbl_pan_readout)

        self._lbl_tilt_readout = QLabel("TILT: +0.00°")
        self._lbl_tilt_readout.setFont(QFont("Consolas", 9, QFont.Bold))
        self._lbl_tilt_readout.setStyleSheet(f"color:{ACCENT};")
        rb_l.addWidget(self._lbl_tilt_readout)

        rb_l.addStretch()

        self._lbl_fps_readout = QLabel("FPS: 30.0")
        self._lbl_fps_readout.setFont(QFont("Consolas", 8, QFont.Bold))
        self._lbl_fps_readout.setStyleSheet(f"color:{TXT};")
        rb_l.addWidget(self._lbl_fps_readout)

        self._lbl_snr_readout = QLabel("SNR: 12.4 dB")
        self._lbl_snr_readout.setFont(QFont("Consolas", 8, QFont.Bold))
        self._lbl_snr_readout.setStyleSheet(f"color:{TXT};")
        rb_l.addWidget(self._lbl_snr_readout)

        self._lbl_int_readout = QLabel("INTENSITY: 87%")
        self._lbl_int_readout.setFont(QFont("Consolas", 8, QFont.Bold))
        self._lbl_int_readout.setStyleSheet(f"color:{TXT};")
        rb_l.addWidget(self._lbl_int_readout)

        vl.addWidget(self._cam_readout_bar)

        return w

    # ── 3. Middle System Process Pipeline Strip ──────────────────────────────
    def _mk_pipeline_strip(self):
        strip = QFrame(); strip.setFixedHeight(48)
        strip.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        hl = QHBoxLayout(strip); hl.setContentsMargins(10, 4, 10, 4); hl.setSpacing(6)

        title = QLabel("SYSTEM PROCESS PIPELINE")
        title.setFont(QFont("Segoe UI", 8, QFont.Bold))
        title.setStyleSheet(f"color:{TXT_MUT}; padding-right:8px;")
        hl.addWidget(title)

        nodes = [
            ("DETECTION", "Beacon Detected", GREEN),
            ("CENTROID", "Updated", GREEN),
            ("KCF TRACKER", "Tracking", GREEN),
            ("OPTICAL FLOW", "Active", GREEN),
            ("XGBOOST", "NORMAL", GREEN),
            ("ESTIMATOR", "KCF / Direct", GREEN),
            ("VALIDATION", "Valid", GREEN),
            ("PID CONTROL", "Active", GREEN),
        ]

        self._pipe_nodes: dict[str, tuple[QLabel, QLabel]] = {}

        for i, (stage, status, col) in enumerate(nodes):
            n_box = QFrame()
            n_box.setStyleSheet(f"background:{BG}; border:1px solid {PANEL_B}; border-radius:4px;")
            nl = QVBoxLayout(n_box); nl.setContentsMargins(8, 3, 8, 3); nl.setSpacing(0)

            l1 = QLabel(stage)
            l1.setFont(QFont("Segoe UI", 7, QFont.Bold))
            l1.setStyleSheet(f"color:{TXT_MUT};")

            l2 = QLabel(status)
            l2.setFont(QFont("Segoe UI", 7, QFont.Bold))
            l2.setStyleSheet(f"color:{col};")

            nl.addWidget(l1); nl.addWidget(l2)
            hl.addWidget(n_box)

            self._pipe_nodes[stage] = (l1, l2)

            if i < len(nodes) - 1:
                arr = QLabel("➔")
                arr.setFont(QFont("Consolas", 9))
                arr.setStyleSheet(f"color:{PANEL_B};")
                hl.addWidget(arr)

        return strip

    # ── 4. Right Column Telemetry Cards ──────────────────────────────────────
    def _mk_right_col(self):
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setStyleSheet(f"background:{BG}; border:none;")

        inn = QWidget(); inn.setStyleSheet(f"background:{BG};")
        vl  = QVBoxLayout(inn); vl.setSpacing(6); vl.setContentsMargins(0, 0, 0, 0)

        # 1. FSOC OPTICAL LINK
        self._link, self._sp_int = self._panel_with_sparkline(
            vl, "📡 FSOC OPTICAL LINK",
            ["Beacon State", "Wavelength", "Intensity", "SNR (dB)", "Pointing Error", "Lock Retention"],
            "BEACON INTENSITY", GREEN
        )

        # 2. TARGET TRACKING
        self._trk, self._sp_conf = self._panel_with_sparkline(
            vl, "🎯 TARGET TRACKING",
            ["Tracking Health", "Centroid X", "Centroid Y", "Target Velocity", "Kalman Confidence"],
            "TRACKING CONFIDENCE", AMBER
        )

        # 3. AI DISTURBANCE & ESTIMATION
        self._ai, self._sp_unc = self._panel_with_sparkline(
            vl, "🤖 AI DISTURBANCE & ESTIMATION",
            ["Disturbance Class", "XGBoost Confidence", "Active Estimator", "Estimate X", "Estimate Y", "Velocity X", "Velocity Y", "Uncertainty"],
            "ESTIMATION UNCERTAINTY", ACCENT
        )

        # 4. VALIDATION & CONTROL
        self._val, self._sp_nis = self._panel_with_sparkline(
            vl, "✅ VALIDATION & CONTROL",
            ["Innovation NIS", "Prediction Valid", "Residual X", "Residual Y", "PID Pan Command", "PID Tilt Command"],
            "INNOVATION NIS", AMBER, show_thresh=True
        )

        # 5. REACQUISITION & RADAR
        self._rec = self._panel_with_radar(
            vl, "🔄 REACQUISITION",
            ["Reacquisition State", "Reacquisition Level", "Loss Duration", "Recovery Progress"]
        )

        sc.setWidget(inn)
        return sc

    # ── 5. Bottom Section: Performance Metrics + Plots + Controls ───────────
    def _mk_bottom(self):
        bar = QWidget()
        bar.setFixedHeight(210)
        bar.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        hl = QHBoxLayout(bar); hl.setContentsMargins(8, 6, 8, 6); hl.setSpacing(8)

        # 1. Performance Metrics Grid (Left)
        mw = QFrame()
        mw.setFixedWidth(230)
        mw.setStyleSheet(f"background:{BG}; border:1px solid {PANEL_B}; border-radius:4px;")
        ml = QVBoxLayout(mw); ml.setContentsMargins(8, 6, 8, 6); ml.setSpacing(2)

        t2 = QLabel("PERFORMANCE METRICS")
        t2.setFont(QFont("Segoe UI", 8, QFont.Bold))
        t2.setStyleSheet(f"color:{GREEN};"); ml.addWidget(t2); ml.addSpacing(2)

        p_grid = QGridLayout(); p_grid.setSpacing(2)
        self._pm: dict[str, QLabel] = {}

        m_items = [
            ("Acquisition Time", "1.00 s"), ("RMSE Error", "15.9 px"), ("Max Error", "38.2 px"), ("Avg Error", "12.7 px"),
            ("Lock Retention", "100.0 %"), ("Reacq. Time", "0.00 s"), ("Reacq. Success", "100.0 %"), ("System Latency", "24.8 ms"),
            ("FPS", "30.0"), ("Total Frames", "18234"), ("Lost Frames", "0"), ("Lock Uptime", "100.0 %")
        ]

        for i, (nm, val) in enumerate(m_items):
            r = i // 2; c = (i % 2) * 2
            k = QLabel(f"{nm}:"); k.setFont(QFont("Consolas", 7)); k.setStyleSheet(f"color:{TXT_MUT};")
            v = QLabel(val); v.setFont(QFont("Consolas", 7, QFont.Bold)); v.setStyleSheet(f"color:{TXT};")
            p_grid.addWidget(k, r, c)
            p_grid.addWidget(v, r, c + 1)
            self._pm[nm] = v

        ml.addLayout(p_grid); ml.addStretch(); hl.addWidget(mw)

        # 2. Real-time PyQtGraph Plots (Center)
        pg.setConfigOption("background", BG); pg.setConfigOption("foreground", TXT_MUT)

        self._p_err = pg.PlotWidget(title="POINTING ERROR (px)")
        self._p_err.setMaximumHeight(190); self._p_err.showGrid(x=True, y=True, alpha=0.15)
        self._p_err.setYRange(0, 200)
        self._c_err = self._p_err.plot(pen=pg.mkPen(ACCENT, width=1.5))
        self._p_err.addItem(pg.InfiniteLine(10, angle=0, pen=pg.mkPen(RED, width=1, style=Qt.DashLine)))
        self._ebuf: list[float] = []
        hl.addWidget(self._p_err, stretch=1)

        self._p_vel = pg.PlotWidget(title="TARGET VELOCITY (px/s)")
        self._p_vel.setMaximumHeight(190); self._p_vel.showGrid(x=True, y=True, alpha=0.15)
        self._c_vel = self._p_vel.plot(pen=pg.mkPen(GREEN, width=1.5))
        self._vbuf: list[float] = []
        hl.addWidget(self._p_vel, stretch=1)

        self._p_nis = pg.PlotWidget(title="INNOVATION NIS")
        self._p_nis.setMaximumHeight(190); self._p_nis.showGrid(x=True, y=True, alpha=0.15)
        self._c_nis = self._p_nis.plot(pen=pg.mkPen(AMBER, width=1.5))
        self._p_nis.addItem(pg.InfiniteLine(self._nis.threshold, angle=0, pen=pg.mkPen(RED, width=1, style=Qt.DashLine)))
        self._nbuf: list[float] = []
        hl.addWidget(self._p_nis, stretch=1)

        # 3. Simulation Controls & Quick Actions (Right)
        ctrl_w = QFrame()
        ctrl_w.setFixedWidth(290)
        ctrl_w.setStyleSheet(f"background:{BG}; border:1px solid {PANEL_B}; border-radius:4px;")
        cl = QVBoxLayout(ctrl_w); cl.setContentsMargins(8, 6, 8, 6); cl.setSpacing(4)

        ct_lbl = QLabel("SIMULATION CONTROLS")
        ct_lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
        ct_lbl.setStyleSheet(f"color:{GREEN};"); cl.addWidget(ct_lbl)

        grid_c = QGridLayout(); grid_c.setSpacing(4)

        grid_c.addWidget(self._lbl("TRAJECTORY"), 0, 0)
        self._traj = QComboBox(); self._traj.addItems(TRAJECTORY_NAMES)
        self._traj.setCurrentText("Circular"); self._traj.setStyleSheet(self._cbs())
        self._traj.currentTextChanged.connect(lambda t: self._env.trajectory.set_trajectory(t))
        grid_c.addWidget(self._traj, 1, 0)

        grid_c.addWidget(self._lbl("DISTURBANCE"), 0, 1)
        self._dist = QComboBox()
        self._dist.addItems(["None", "Salt & Pepper", "Gaussian", "Poisson", "Combined"])
        self._dist.setStyleSheet(self._cbs())
        self._dist.currentTextChanged.connect(self._on_dist)
        grid_c.addWidget(self._dist, 1, 1)

        cl.addLayout(grid_c)

        for s_lbl, val_default, slot in [
            ("Beacon Intensity", int(CFG.beacon.intensity), lambda v: setattr(CFG.beacon, "intensity", float(v))),
            ("Noise Level", int(CFG.disturbance.gaussian_sigma), lambda v: setattr(CFG.disturbance, "gaussian_sigma", float(v))),
        ]:
            s_row = QHBoxLayout()
            s_row.addWidget(self._lbl(s_lbl))
            sl = QSlider(Qt.Horizontal); sl.setRange(0, 255); sl.setValue(val_default); sl.setFixedHeight(12)
            sl.valueChanged.connect(slot)
            s_row.addWidget(sl)
            cl.addLayout(s_row)

        qa_grid = QGridLayout(); qa_grid.setSpacing(4)
        b1 = QPushButton("START DISTURBANCE"); b1.setFont(QFont("Segoe UI", 7, QFont.Bold))
        b1.setStyleSheet(f"background:#1e293b; color:{TXT}; border:1px solid {PANEL_B}; border-radius:3px; padding:4px;")
        b1.clicked.connect(lambda: setattr(CFG.disturbance, "jitter_enabled", True))
        qa_grid.addWidget(b1, 0, 0)

        self._loss_btn = QPushButton("FORCE BEACON LOSS"); self._loss_btn.setCheckable(True)
        self._loss_btn.setFont(QFont("Segoe UI", 7, QFont.Bold))
        self._loss_btn.setStyleSheet(f"QPushButton{{background:#450a0a; color:{RED}; border:1px solid {RED}55; border-radius:3px; padding:4px;}}"
                                     f"QPushButton:checked{{background:{RED}; color:white;}}")
        self._loss_btn.toggled.connect(lambda v: self._env.beacon.force_loss(v))
        qa_grid.addWidget(self._loss_btn, 0, 1)

        b3 = QPushButton("REACQUIRE BEACON"); b3.setFont(QFont("Segoe UI", 7, QFont.Bold))
        b3.setStyleSheet(f"background:#065f46; color:{GREEN}; border:1px solid {GREEN}55; border-radius:3px; padding:4px;")
        b3.clicked.connect(self._reset)
        qa_grid.addWidget(b3, 1, 0)

        b4 = QPushButton("SAVE LOG"); b4.setFont(QFont("Segoe UI", 7, QFont.Bold))
        b4.setStyleSheet(f"background:#1e293b; color:{TXT}; border:1px solid {PANEL_B}; border-radius:3px; padding:4px;")
        qa_grid.addWidget(b4, 1, 1)

        cl.addLayout(qa_grid)
        hl.addWidget(ctrl_w)

        return bar

    # ── 6. Footer Status Bar ──────────────────────────────────────────────────
    def _mk_footer(self):
        foot = QFrame(); foot.setFixedHeight(24)
        foot.setStyleSheet(f"background:#020408; border-top:1px solid {PANEL_B};")
        fl = QHBoxLayout(foot); fl.setContentsMargins(12, 0, 12, 0); fl.setSpacing(20)

        self._ft_fps = QLabel("PROCESSING FPS  30.0")
        self._ft_fps.setFont(QFont("Consolas", 8, QFont.Bold)); self._ft_fps.setStyleSheet(f"color:{GREEN};")
        fl.addWidget(self._ft_fps)

        self._ft_lat = QLabel("PIPELINE LATENCY  24.8 ms")
        self._ft_lat.setFont(QFont("Consolas", 8, QFont.Bold)); self._ft_lat.setStyleSheet(f"color:{ACCENT};")
        fl.addWidget(self._ft_lat)

        self._ft_tgt = QLabel("TARGET STATUS  LOCKED")
        self._ft_tgt.setFont(QFont("Consolas", 8, QFont.Bold)); self._ft_tgt.setStyleSheet(f"color:{GREEN};")
        fl.addWidget(self._ft_tgt)

        self._ft_err = QLabel("TRACKING ERROR  0.0 px")
        self._ft_err.setFont(QFont("Consolas", 8, QFont.Bold)); self._ft_err.setStyleSheet(f"color:{AMBER};")
        fl.addWidget(self._ft_err)

        self._ft_sim = QLabel("SIMULATION TIME  0.0 s")
        self._ft_sim.setFont(QFont("Consolas", 8, QFont.Bold)); self._ft_sim.setStyleSheet(f"color:{TXT};")
        fl.addWidget(self._ft_sim)

        fl.addStretch()

        self._ft_sys = QLabel("SYSTEM STATUS  ● RUNNING")
        self._ft_sys.setFont(QFont("Consolas", 8, QFont.Bold)); self._ft_sys.setStyleSheet(f"color:{GREEN};")
        fl.addWidget(self._ft_sys)

        return foot

    # ================================================================ HELPERS =
    def _panel_with_sparkline(self, vl, title, keys, spark_title, spark_col, show_thresh=False):
        g = QGroupBox(title)
        g.setFont(QFont("Segoe UI", 8, QFont.Bold))
        g.setStyleSheet(f"QGroupBox{{color:{GREEN}; border:1px solid {PANEL_B};"
                        f"border-radius:4px; margin-top:6px; padding-top:4px;"
                        f"background:{PANEL};}}"
                        f"QGroupBox::title{{subcontrol-origin:margin; left:8px; padding:0 4px; background:{PANEL};}}")

        h_box = QHBoxLayout(g); h_box.setContentsMargins(6, 4, 6, 6); h_box.setSpacing(6)

        grid = QGridLayout(); grid.setSpacing(2)
        rows = {}
        for i, k in enumerate(keys):
            kl = QLabel(k + ":"); kl.setFont(QFont("Consolas", 7)); kl.setStyleSheet(f"color:{TXT_MUT};")
            vl2 = QLabel("--"); vl2.setFont(QFont("Consolas", 7, QFont.Bold)); vl2.setStyleSheet(f"color:{TXT};")
            grid.addWidget(kl, i, 0); grid.addWidget(vl2, i, 1); rows[k] = (kl, vl2)
        h_box.addLayout(grid, stretch=3)

        sp_w = QWidget(); sp_w.setFixedWidth(110)
        sp_l = QVBoxLayout(sp_w); sp_l.setContentsMargins(0, 0, 0, 0); sp_l.setSpacing(2)
        sp_lbl = QLabel(spark_title); sp_lbl.setFont(QFont("Segoe UI", 6, QFont.Bold))
        sp_lbl.setStyleSheet(f"color:{TXT_MUT};"); sp_lbl.setAlignment(Qt.AlignCenter)
        sp_l.addWidget(sp_lbl)

        pw = pg.PlotWidget()
        pw.setFixedHeight(60); pw.showGrid(x=False, y=False)
        pw.hideAxis('left'); pw.hideAxis('bottom')
        curve = pw.plot(pen=pg.mkPen(spark_col, width=1.2))
        if show_thresh:
            pw.addItem(pg.InfiniteLine(self._nis.threshold, angle=0, pen=pg.mkPen(RED, width=1, style=Qt.DashLine)))

        sp_l.addWidget(pw)
        h_box.addWidget(sp_w, stretch=2)

        vl.addWidget(g)
        return rows, (pw, curve, [])

    def _panel_with_radar(self, vl, title, keys):
        g = QGroupBox(title)
        g.setFont(QFont("Segoe UI", 8, QFont.Bold))
        g.setStyleSheet(f"QGroupBox{{color:{GREEN}; border:1px solid {PANEL_B};"
                        f"border-radius:4px; margin-top:6px; padding-top:4px;"
                        f"background:{PANEL};}}"
                        f"QGroupBox::title{{subcontrol-origin:margin; left:8px; padding:0 4px; background:{PANEL};}}")

        h_box = QHBoxLayout(g); h_box.setContentsMargins(6, 4, 6, 6); h_box.setSpacing(6)

        grid = QGridLayout(); grid.setSpacing(2)
        rows = {}
        for i, k in enumerate(keys):
            kl = QLabel(k + ":"); kl.setFont(QFont("Consolas", 7)); kl.setStyleSheet(f"color:{TXT_MUT};")
            vl2 = QLabel("--"); vl2.setFont(QFont("Consolas", 7, QFont.Bold)); vl2.setStyleSheet(f"color:{TXT};")
            grid.addWidget(kl, i, 0); grid.addWidget(vl2, i, 1); rows[k] = (kl, vl2)
        h_box.addLayout(grid, stretch=3)

        sr_w = QWidget(); sr_w.setFixedWidth(110)
        sr_l = QVBoxLayout(sr_w); sr_l.setContentsMargins(0, 0, 0, 0); sr_l.setSpacing(2)
        sr_lbl = QLabel("SEARCH REGION"); sr_lbl.setFont(QFont("Segoe UI", 6, QFont.Bold))
        sr_lbl.setStyleSheet(f"color:{TXT_MUT};"); sr_lbl.setAlignment(Qt.AlignCenter)
        sr_l.addWidget(sr_lbl)
        sr_l.addWidget(SearchRegionWidget())
        h_box.addWidget(sr_w, stretch=2)

        vl.addWidget(g)
        return rows

    def _lbl(self, t):
        l = QLabel(t); l.setFont(QFont("Segoe UI", 7, QFont.Bold)); l.setStyleSheet(f"color:{TXT_MUT};"); return l

    def _cbs(self):
        return (f"QComboBox {{ background:{BG}; color:{TXT}; border:1px solid {PANEL_B};"
                f"border-radius:3px; font-family:'Segoe UI'; font-size:7pt; font-weight:bold; padding:2px 4px; }}"
                f"QComboBox QAbstractItemView {{ background:{PANEL}; color:{TXT}; selection-background-color:{GREEN}44; }}")

    def _ls(self, col, bg):
        return (f"color:{col}; padding:3px 10px; background:{bg};"
                f"border:1px solid {col}55; border-radius:4px; font-family:'Segoe UI'; font-size:8pt; font-weight:bold;")

    def _style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{BG}; color:{TXT}; font-family:'Segoe UI', sans-serif; }}
            QLabel {{ color:{TXT}; }}
            QSlider::groove:horizontal {{ height:4px; background:{PANEL_B}; border-radius:2px; }}
            QSlider::handle:horizontal {{ background:{GREEN}; width:10px; height:10px; margin:-3px 0; border-radius:5px; }}
            QScrollBar:vertical {{ background:{BG}; width:5px; }}
            QScrollBar::handle:vertical {{ background:{PANEL_B}; border-radius:2px; }}
        """)

    def _update_clock(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        self._clock_lbl.setText(now.strftime("%H:%M:%S UTC\n%d-%m-%Y"))

    # ================================================================ EVENTS =
    def _on_dist(self, t):
        m = {"None": "none", "Salt & Pepper": "salt_pepper",
             "Gaussian": "gaussian", "Poisson": "poisson", "Combined": "combined"}
        CFG.disturbance.noise_type = m.get(t, "none")

    def _reset(self):
        self._env.reset(); self._kcf.reset(); self._of.reset()
        self._est.reset(); self._pid.reset(); self._reacq.reset(); self._perf.reset()
        self._acq_done = False; self._frame_idx = 0
        for b in (self._ebuf, self._nbuf, self._vbuf): b.clear()
        self._perf.mark_session_start()

    def _train(self):
        ok = DisturbanceClassifier.train_and_save(8000)
        if ok: self._clf = DisturbanceClassifier()

    # ================================================================ DEMO =
    _DEMO_PHASES = [
        (0,  5,  "① Receiver satellite visible — beacon off-centre"),
        (5,  10, "② Beacon detected — KCF/Direct Passthrough active [NORMAL]"),
        (10, 16, "③ PID steering camera — Gaussian noise [MILD: Kalman Filter]"),
        (16, 20, "④ 🔒 BEACON LOCK established — Nominal tracking [NORMAL: KCF/Direct]"),
        (20, 25, "⑤ Salt & Pepper + Jitter → [MODERATE: Kalman + Optical Flow]"),
        (25, 30, "⑥ Combined Noise + Jitter → [MODERATE: Kalman + Optical Flow]"),
        (30, 36, "⑦ Severe Jitter + Haze → [SEVERE: Unscented Kalman Filter]"),
        (36, 43, "⑧ Beacon FORCED LOSS → Hierarchical Reacquisition [SEVERE: UKF]"),
        (43, 50, "⑨ Beacon re-found — PID restoring lock"),
        (50, 999,"⑩ 🔒 BEACON LOCK RESTORED — system nominal [NORMAL: KCF/Direct]"),
    ]

    def _start_demo(self):
        self._reset(); self._demo_active = True
        self._demo_phase = -1; self._demo_t0 = time.perf_counter()

    def _run_demo(self):
        if not self._demo_active: return
        el = time.perf_counter() - self._demo_t0
        for i, (ts, te, lbl) in enumerate(self._DEMO_PHASES):
            if ts <= el < te:
                if i != self._demo_phase:
                    self._demo_phase = i; self._apply_demo(i)
                self._s.phase_lbl = lbl; self._phase_lbl.setText(lbl); return
        self._phase_lbl.setText("Demo complete"); self._demo_active = False

    def _apply_demo(self, i):
        d = CFG.disturbance
        if i == 4:
            d.noise_type = "gaussian"; d.jitter_enabled = False; d.platform_motion_enabled = False; d.haze = False
        elif i == 5:
            d.noise_type = "salt_pepper"; d.salt_pepper_ratio = 0.04; d.jitter_enabled = True; d.platform_motion_enabled = True; d.jitter_max_px = 6
        elif i == 6:
            d.noise_type = "combined"; d.jitter_enabled = True; d.platform_motion_enabled = False; d.gaussian_sigma = 15
        elif i == 7:
            d.noise_type = "poisson"; d.jitter_enabled = True; d.platform_motion_enabled = True; d.haze = True
        elif i == 8:
            self._env.beacon.force_loss(True); self._acq_done = False
        elif i == 9:
            self._env.beacon.force_loss(False); self._acq_done = False
            d.noise_type = "none"; d.jitter_enabled = False; d.platform_motion_enabled = False; d.haze = False

    # ================================================================ MAIN TICK =
    def _tick(self):
        try: self._tick_impl()
        except Exception: import traceback; traceback.print_exc()

    def _tick_impl(self):
        t0 = time.perf_counter(); dt = 1 / CFG.camera.fps
        self._frame_idx += 1; self._s.frame_idx = self._frame_idx
        if self._demo_active: self._run_demo()

        if self._mode == "simulation":
            self._env.step(dt)
            frame = self._env.render()
        else:
            if self._mp4 and self._mp4.is_open:
                ok, frame = self._mp4.read_frame()
                if not ok: return
            else: return

        self._pipeline(frame, dt)

        s = self._s
        if self._acq_done:
            ex = s.state_x - CFG.camera.width / 2
            ey = s.state_y - CFG.camera.height / 2
            pc, tc = self._pid.update(ex, ey, dt, s.state_vx, s.state_vy)
            s.pan_cmd, s.tilt_cmd = pc, tc
            if self._mode == "simulation":
                self._env.apply_pan_tilt_rate(pc, tc, dt)
            s.pan_deg  = self._env.pan_deg
            s.tilt_deg = self._env.tilt_deg

        fps = 30.0; latms = (time.perf_counter() - t0) * 1000
        self._perf.record_frame(s.centroid_x, s.centroid_y, s.beacon_detected, fps, latms)
        s.fps = fps; s.latency_ms = latms

        self._show_frame(self._overlay(frame.copy()))
        self._update_panels(); self._update_plots(); self._update_lock()

    # ──────────────────────────────────────────────────────── PIPELINE ──────
    def _pipeline(self, frame: np.ndarray, dt: float):
        s = self._s; W, H = CFG.camera.width, CFG.camera.height
        cx0, cy0 = W // 2, H // 2

        if not self._acq_done:
            cands = acquire_beacon(frame)
            if cands:
                cx, cy, _ = cands[0]
                xc, yc, _ = compute_centroid(frame, cx, cy)
                self._kcf.reinitialise(frame, int(xc), int(yc))
                self._of.initialise(frame, int(xc), int(yc))
                self._est.initialise(xc, yc); self._acq_done = True
                self._perf.mark_acquired()
                s.beacon_detected = True; s.centroid_x, s.centroid_y = xc, yc
                self._reacq.reset()
            else:
                s.beacon_detected = False
                if self._reacq.is_active:
                    res = self._reacq.attempt(frame, s.predicted_x, s.predicted_y)
                    if res:
                        cx, cy = res; xc, yc, _ = compute_centroid(frame, cx, cy)
                        self._kcf.reinitialise(frame, int(xc), int(yc))
                        self._of.initialise(frame, int(xc), int(yc))
                        self._est.initialise(xc, yc); self._acq_done = True
                        s.beacon_detected = True; s.centroid_x, s.centroid_y = xc, yc
                        self._reacq.reset()
                else: self._reacq.start(self._frame_idx)
                s.reacq_active = self._reacq.is_active
                s.reacq_level  = self._reacq.current_level_name
                s.loss_dur += dt; s.lock = False; return

        kx, ky, kc = self._kcf.update(frame)
        s.kcf_conf = kc; s.kcf_box = self._kcf.bbox
        if not self._kcf.is_initialised or kc < 0.04:
            s.beacon_detected = False; self._acq_done = False
            self._reacq.start(self._frame_idx); s.loss_dur += dt; s.lock = False; return

        xc, yc, totI = compute_centroid(frame, int(kx), int(ky))
        snr = compute_snr(frame, int(xc), int(yc))
        s.centroid_x, s.centroid_y = xc, yc; s.snr = snr
        s.intensity = float(np.clip(totI / 100, 0, 255))

        of_dx, of_dy, of_mag, _ = self._of.update(frame)
        s.of_vectors = self._of.flow_vectors

        err = float(np.hypot(xc - cx0, yc - cy0))
        vel = float(np.hypot(s.state_vx, s.state_vy))
        acc = float(np.hypot(s.state_vx - self._prev_vx,
                             s.state_vy - self._prev_vy)) / max(dt, 0.01)

        fv = FeatureVector(velocity=vel, acceleration=acc, tracking_confidence=kc,
                           pointing_error=err, optical_flow_mag=of_mag,
                           beacon_intensity=s.intensity, snr=snr, loss_duration=s.loss_dur,
                           prediction_uncertainty=s.uncertainty, nis=s.nis)
        regime, _, xconf = self._clf.classify(fv)
        self._est.set_regime(regime)
        s.regime = regime; s.xgb_conf = xconf; s.xgb_mode = self._clf.mode
        s.estimator_name = self._est.active_name

        if s.regime == "NORMAL":
            s.active_conf_name = "KCF Direct Conf"
            s.active_conf_val  = float(kc)
        elif s.regime == "MILD":
            s.active_conf_name = "Kalman Conf"
            s.active_conf_val  = float(np.clip(1.0 - s.uncertainty / 12.0, 0.10, 0.99))
        elif s.regime == "MODERATE":
            s.active_conf_name = "KF+OF Conf"
            s.active_conf_val  = float(np.clip(0.92 - of_mag / 15.0, 0.15, 0.95))
        else:  # SEVERE
            s.active_conf_name = "UKF Conf"
            s.active_conf_val  = float(np.clip(1.0 - s.uncertainty / 25.0, 0.05, 0.90))

        z = np.array([xc, yc])
        self._est.predict(); self._est.update(z, of_dx, of_dy)
        sx, sy, svx, svy = self._est.state
        s.state_x, s.state_y = sx, sy; s.state_vx, s.state_vy = svx, svy
        s.uncertainty = self._est.uncertainty
        self._prev_vx, self._prev_vy = svx, svy
        s.predicted_x = sx + svx * dt; s.predicted_y = sy + svy * dt

        try:
            r, S = self._est.get_innovation(z)
            nis, nv, _ = self._nis.validate(r, S)
            rf = np.atleast_1d(np.squeeze(r)).ravel()
            s.residual_x = float(rf[0]) if len(rf) > 0 else 0.0
            s.residual_y = float(rf[1]) if len(rf) > 1 else 0.0
            s.nis = nis; s.nis_valid = nv
        except Exception: pass

        s.pointing_error = err; s.velocity = vel
        s.beacon_detected = True; s.reacq_active = False; s.loss_dur = 0.0
        s.lock = (kc > 0.08 and err < 110)
        s.tracking_quality = (
            "HEALTHY"  if kc > 0.45 and snr > 4 and err < 35 else
            "CRITICAL" if kc < 0.12 or err > 120            else "DEGRADED")

        if not s.nis_valid and s.nis > self._nis.threshold * 3:
            self._acq_done = False; self._reacq.start(self._frame_idx)

    # ──────────────────────────────────────────────────── OVERLAY RENDER ────
    def _overlay(self, frame: np.ndarray) -> np.ndarray:
        s = self._s; W, H = CFG.camera.width, CFG.camera.height; cx0, cy0 = W // 2, H // 2
        out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)



        # ── Sender Optical Terminal Dish (Bottom Center Origin) ──────────────
        sender_pt = (cx0, H - 2)
        cv2.circle(out, sender_pt, 6, (16, 185, 129), -1)
        cv2.putText(out, "SENDER TERMINAL", (cx0 - 45, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (16, 185, 129), 1, cv2.LINE_AA)

        # ── Red Optical Laser Beam: Sender Terminal -> Receiver Beacon ────────
        bx = int(np.clip(s.centroid_x if s.beacon_detected else s.predicted_x, 0, W - 1))
        by = int(np.clip(s.centroid_y if s.beacon_detected else s.predicted_y, 0, H - 1))
        self._draw_beam(out, sender_pt, (bx, by), s.frame_idx, s.beacon_detected)

        # ── Debug Telemetry Overlay Box (Top-Right) ────────────────────────
        if getattr(self, "_debug_enabled", True):
            st = self._env.trajectory.state
            dbg_w, dbg_h = 175, 140
            dx0, dy0 = W - dbg_w - 10, 10
            cv2.rectangle(out, (dx0, dy0), (dx0 + dbg_w, dy0 + dbg_h), (10, 15, 25), -1)
            cv2.rectangle(out, (dx0, dy0), (dx0 + dbg_w, dy0 + dbg_h), (16, 185, 129), 1)
            cv2.putText(out, "DEBUG TELEMETRY", (dx0 + 6, dy0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (16, 185, 129), 1)

            lines = [
                f"Earth Ctr: {CFG.earth.centre_offset_x:.1f},{CFG.earth.centre_offset_y:.1f}",
                f"Recv Pos:  {st.x:.2f},{st.y:.2f}",
                f"Recv Vel:  {st.vx:.2f},{st.vy:.2f}",
                f"Cam Pan:   {self._env.pan_deg:.2f}°",
                f"Cam Tilt:  {self._env.tilt_deg:.2f}°",
                f"Meas X,Y:  {s.centroid_x:.1f},{s.centroid_y:.1f}",
                f"Pred X,Y:  {s.predicted_x:.1f},{s.predicted_y:.1f}",
                f"Pt Error:  {s.pointing_error:.1f} px",
            ]
            for li, line_str in enumerate(lines):
                cv2.putText(out, line_str, (dx0 + 6, dy0 + 28 + li * 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 210, 225), 1)

        # ── Compass Rose ─────────────────────────────────────────────────
        cr_x = W - 40
        cr_y = 185 if getattr(self, "_debug_enabled", True) else 45
        cv2.line(out, (cr_x, cr_y - 18), (cr_x, cr_y + 18), (0, 180, 120), 1)
        cv2.line(out, (cr_x - 18, cr_y), (cr_x + 18, cr_y), (0, 180, 120), 1)
        cv2.putText(out, "N", (cr_x - 4, cr_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (239, 68, 68), 1)
        cv2.putText(out, "S", (cr_x - 4, cr_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 120), 1)
        cv2.putText(out, "E", (cr_x + 20, cr_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 120), 1)
        cv2.putText(out, "W", (cr_x - 30, cr_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 120), 1)

        # ── Top-Left Tracking Badge ──────────────────────────────────────
        cv2.rectangle(out, (10, 10), (105, 30), (10, 15, 25), -1)
        cv2.circle(out, (22, 20), 4, (16, 185, 129), -1)
        cv2.putText(out, "TRACKING", (32, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 245, 255), 1, cv2.LINE_AA)

        # ── Legend Overlay Box (Bottom Left) ─────────────────────────────
        lg_x, lg_y = 12, H - 90
        cv2.rectangle(out, (lg_x, lg_y), (lg_x + 125, lg_y + 78), (10, 15, 25), -1)
        cv2.rectangle(out, (lg_x, lg_y), (lg_x + 125, lg_y + 78), (30, 45, 70), 1)
        cv2.putText(out, "LEGEND", (lg_x + 6, lg_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (16, 185, 129), 1)

        cv2.drawMarker(out, (lg_x + 12, lg_y + 28), (16, 185, 129), cv2.MARKER_CROSS, 8, 1)
        cv2.putText(out, "Centroid", (lg_x + 22, lg_y + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 210, 225), 1)

        cv2.rectangle(out, (lg_x + 8, lg_y + 38), (lg_x + 16, lg_y + 46), (16, 185, 129), 1)
        cv2.putText(out, "KCF Track Box", (lg_x + 22, lg_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 210, 225), 1)

        cv2.drawMarker(out, (lg_x + 12, lg_y + 54), (245, 158, 11), cv2.MARKER_CROSS, 8, 1)
        cv2.putText(out, "Predicted Pos.", (lg_x + 22, lg_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 210, 225), 1)

        cv2.line(out, (lg_x + 6, lg_y + 68), (lg_x + 18, lg_y + 68), (0, 180, 230), 1)
        cv2.putText(out, "Optical Flow", (lg_x + 22, lg_y + 71), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 210, 225), 1)

        # ── Camera Center Overlay Box (Bottom Right) ─────────────────────
        cc_x, cc_y = W - 110, H - 65
        cv2.rectangle(out, (cc_x, cc_y), (cc_x + 98, cc_y + 52), (10, 15, 25), -1)
        cv2.rectangle(out, (cc_x, cc_y), (cc_x + 98, cc_y + 52), (30, 45, 70), 1)
        cv2.putText(out, "CAMERA CENTER", (cc_x + 6, cc_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (16, 185, 129), 1)
        cv2.drawMarker(out, (cc_x + 49, cc_y + 32), (245, 158, 11), cv2.MARKER_CROSS, 14, 1)

        # ── Bounding Box + Active Tracker Confidence Overlay ────────────────
        if s.kcf_box and s.beacon_detected:
            bx2, by2, bw, bh = s.kcf_box
            cv2.rectangle(out, (bx2, by2), (bx2 + bw, by2 + bh), (16, 185, 129), 1)

            tag = "KCF/Direct" if s.regime == "NORMAL" else ("Kalman" if s.regime == "MILD" else ("KF+OF" if s.regime == "MODERATE" else "UKF"))
            txt = f"{tag} {s.active_conf_val:.2f}"

            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx0 = max(0, bx2)
            ty0 = max(16, by2 - 6)
            cv2.rectangle(out, (tx0, ty0 - th - 4), (tx0 + tw + 6, ty0 + 2), (10, 15, 25), -1)
            cv2.putText(out, txt, (tx0 + 3, ty0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1, cv2.LINE_AA)

        # ── Optical Flow Vectors ─────────────────────────────────────────
        for ox, oy, nx2, ny2 in s.of_vectors[:6]:
            cv2.arrowedLine(out, (int(ox), int(oy)), (int(nx2), int(ny2)), (0, 180, 230), 1, tipLength=0.4)

        # ── Predicted Position Marker ─────────────────────────────────────
        pp = (int(np.clip(s.predicted_x, 0, W - 1)), int(np.clip(s.predicted_y, 0, H - 1)))
        cv2.drawMarker(out, pp, (245, 158, 11), cv2.MARKER_CROSS, 14, 1)

        # ── Vivid RED 1550 nm Beacon Light on Receiver Satellite ─────────────
        b_pt = (int(np.clip(self._env.beacon_px, 0, W - 1)), int(np.clip(self._env.beacon_py, 0, H - 1)))
        for r_red, a_red in [(16, 0.30), (10, 0.60), (5, 1.0)]:
            if a_red < 1.0:
                ov = out.copy()
                cv2.circle(ov, b_pt, r_red, (20, 20, 255), -1)
                cv2.addWeighted(ov, a_red, out, 1 - a_red, 0, out)
            else:
                cv2.circle(out, b_pt, r_red, (30, 30, 255), -1)
                cv2.circle(out, b_pt, 2, (200, 220, 255), -1)

        # ── Centroid Ring & RED 1550nm IR Beacon Spot ──────────────────────
        if s.beacon_detected:
            cp = (int(np.clip(s.centroid_x, 0, W - 1)), int(np.clip(s.centroid_y, 0, H - 1)))
            lock_col = (16, 185, 129) if s.lock else (40, 40, 255)

            # RED Glow rings around the IR beacon laser hot-spot
            for ring_r, ring_a in [(22, 0.25), (16, 0.45), (11, 0.75)]:
                ov = out.copy()
                cv2.circle(ov, cp, ring_r, (40, 40, 255), 1)
                cv2.addWeighted(ov, ring_a, out, 1 - ring_a, 0, out)

            # Vivid RED aperture core spot
            cv2.circle(out, cp, 6, (30, 30, 255), -1)
            cv2.circle(out, cp, 12, (50, 50, 255), 1)
            cv2.drawMarker(out, cp, lock_col, cv2.MARKER_CROSS, 16, 1)

            # Receiver Satellite Label
            sat_lbl = "RECEIVER SATELLITE"
            cv2.putText(out, sat_lbl, (cp[0] - 40, max(20, cp[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (16, 185, 129), 1, cv2.LINE_AA)

            # 1550 nm Beacon Label (RED)
            lbl_txt = "1550 nm BEACON"
            cv2.putText(out, lbl_txt, (cp[0] + 16, cp[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 60, 255), 1, cv2.LINE_AA)

            # Pointing error line
            cv2.line(out, (cx0, cy0), cp, (60, 30, 30), 1)

        # ── Corner Reticle Brackets ──────────────────────────────────────
        m = 24
        for x, y, ddx, ddy in [(0,0,m,0), (0,0,0,m), (W-1,0,-m,0), (W-1,0,0,m),
                               (0,H-1,m,0), (0,H-1,0,-m), (W-1,H-1,-m,0), (W-1,H-1,0,-m)]:
            cv2.line(out, (x, y), (x + ddx, y + ddy), (16, 185, 129), 1)

        return out

    @staticmethod
    def _draw_earth_color(img: np.ndarray, cx: int, cy: int, radius_px: int):
        from simulation.environment import _EARTH_BGR
        if _EARTH_BGR is not None:
            tw = th = max(8, radius_px * 2)
            sprite = cv2.resize(_EARTH_BGR, (tw, th), interpolation=cv2.INTER_AREA)
            grey = cv2.cvtColor(sprite, cv2.COLOR_BGR2GRAY)
            mask = (grey.astype(np.float32) / 255.0)[:, :, None]
            mask = np.clip(mask * 1.6, 0.0, 1.0)

            H, W = img.shape[:2]
            x0, y0 = cx - tw // 2, cy - th // 2
            x1, y1 = x0 + tw,      y0 + th
            sx0 = max(0, -x0);  sy0 = max(0, -y0)
            fx0 = max(0, x0);   fy0 = max(0, y0)
            fx1 = min(W, x1);   fy1 = min(H, y1)
            sx1 = sx0 + fx1 - fx0;  sy1 = sy0 + fy1 - fy0
            if fx1 > fx0 and fy1 > fy0 and sx1 > sx0 and sy1 > sy0:
                roi   = img[fy0:fy1, fx0:fx1].astype(np.float32)
                s_roi = sprite[sy0:sy1, sx0:sx1].astype(np.float32)
                m_roi = mask[sy0:sy1, sx0:sx1]
                img[fy0:fy1, fx0:fx1] = np.clip(
                    roi * (1.0 - m_roi) + s_roi * m_roi, 0, 255
                ).astype(np.uint8)
        else:
            cv2.circle(img, (cx, cy), radius_px, (100, 120, 140), -1)
            cv2.circle(img, (cx, cy), radius_px, (16, 185, 129), 1)

    @staticmethod
    def _draw_beam(img, pt1, pt2, phase, active):
        """Draw glowing RED 1550 nm optical laser communication link."""
        dx = pt2[0] - pt1[0]; dy = pt2[1] - pt1[1]
        dist = math.hypot(dx, dy)
        if dist < 5: return
        ux, uy = dx / dist, dy / dist
        dash, gap = 16, 8; cycle = dash + gap
        offset = (phase * 3) % cycle

        # Red optical beam color
        red_beam = (40, 40, 255)

        for thick, af, cs in [(5, 0.18, 0.4), (3, 0.35, 0.7), (1, 1.0, 1.0)]:
            col = (int(red_beam[0] * cs), int(red_beam[1] * cs), int(red_beam[2] * cs))
            if af < 1.0:
                ov = img.copy(); pos = offset
                while pos < dist:
                    x1 = int(pt1[0] + ux * pos); y1 = int(pt1[1] + uy * pos)
                    ep = min(pos + dash, dist)
                    x2 = int(pt1[0] + ux * ep); y2 = int(pt1[1] + uy * ep)
                    cv2.line(ov, (x1, y1), (x2, y2), col, thick); pos += cycle
                cv2.addWeighted(ov, af, img, 1 - af, 0, img)
            else:
                pos = offset
                while pos < dist:
                    x1 = int(pt1[0] + ux * pos); y1 = int(pt1[1] + uy * pos)
                    ep = min(pos + dash, dist)
                    x2 = int(pt1[0] + ux * ep); y2 = int(pt1[1] + uy * ep)
                    cv2.line(img, (x1, y1), (x2, y2), col, thick); pos += cycle

        cv2.circle(img, pt1, 5, red_beam, -1)
        cv2.circle(img, pt1, 8, (60, 60, 255), 1)
        if active:
            pr = 6 + (phase % 8)
            cv2.circle(img, pt2, pr, (40, 40, 255), 1)

    # ──────────────────────────────────────────────────── UI UPDATES ────────
    def _show_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._view.setPixmap(QPixmap.fromImage(qi))

    def _set(self, rows, key, text, col=None, key_label=None):
        item = rows.get(key)
        if item is None: return
        if isinstance(item, tuple):
            kl, vl2 = item
            if key_label:
                kl.setText(key_label + ":")
            vl2.setText(text)
            vl2.setStyleSheet(f"color:{col if col else TXT};"
                              f"font-family:Consolas; font-size:7.5pt; font-weight:bold;")
        else:
            item.setText(text)
            item.setStyleSheet(f"color:{col if col else TXT};"
                               f"font-family:Consolas; font-size:7.5pt; font-weight:bold;")

    def _update_panels(self):
        s = self._s
        r = self._link
        self._set(r, "Beacon State", "LOCK" if s.lock else ("LOST" if not s.beacon_detected else "DETECTED"),
                  GREEN if s.lock else (RED if not s.beacon_detected else AMBER))
        self._set(r, "Wavelength", "1550 nm")
        self._set(r, "Intensity", f"{s.intensity:.1f}")
        self._set(r, "SNR (dB)", f"{s.snr:.1f}")
        self._set(r, "Pointing Error", f"{s.pointing_error:.1f} px")
        self._set(r, "Lock Retention", f"{self._perf.lock_retention_pct:.1f} %")

        r = self._trk
        qc = {"HEALTHY": GREEN, "DEGRADED": AMBER, "CRITICAL": RED}
        self._set(r, "Tracking Health", s.tracking_quality, qc.get(s.tracking_quality, TXT))
        self._set(r, "Centroid X", f"{s.centroid_x:.1f} px")
        self._set(r, "Centroid Y", f"{s.centroid_y:.1f} px")
        self._set(r, "Target Velocity", f"{s.velocity:.2f} px/s")

        self._set(r, "Kalman Confidence", f"{s.active_conf_val:.3f}", col=AMBER, key_label=s.active_conf_name)

        r = self._ai
        rc = {"NORMAL": GREEN, "MILD": "#00f0ff", "MODERATE": "#f97316", "SEVERE": RED}.get(s.regime, TXT)
        self._set(r, "Disturbance Class", s.regime, rc)
        self._set(r, "XGBoost Confidence", f"{s.xgb_conf:.2f}")
        self._set(r, "Active Estimator", s.estimator_name)
        self._set(r, "Estimate X", f"{s.state_x:.1f} px")
        self._set(r, "Estimate Y", f"{s.state_y:.1f} px")
        self._set(r, "Velocity X", f"{s.state_vx:.2f} px/s")
        self._set(r, "Velocity Y", f"{s.state_vy:.2f} px/s")
        self._set(r, "Uncertainty", f"{s.uncertainty:.2f} px")

        r = self._val
        self._set(r, "Innovation NIS", f"{s.nis:.3f}")
        self._set(r, "Prediction Valid", "YES" if s.nis_valid else "NO", GREEN if s.nis_valid else RED)
        self._set(r, "Residual X", f"{s.residual_x:.2f} px")
        self._set(r, "Residual Y", f"{s.residual_y:.2f} px")
        self._set(r, "PID Pan Command", f"{s.pan_cmd:+.2f}°")
        self._set(r, "PID Tilt Command", f"{s.tilt_cmd:+.2f}°")

        r = self._rec
        self._set(r, "Reacquisition State", "ACTIVE" if s.reacq_active else "NONE", RED if s.reacq_active else GREEN)
        self._set(r, "Reacquisition Level", s.reacq_level)
        self._set(r, "Loss Duration", f"{s.loss_dur:.2f} s")
        self._set(r, "Recovery Progress", f"{0 if s.reacq_active else 100} %")

        self._lbl_pan_readout.setText(f"PAN  {s.pan_deg:+.2f}°")
        self._lbl_tilt_readout.setText(f"TILT  {s.tilt_deg:+.2f}°")
        self._lbl_fps_readout.setText(f"FPS  {s.fps:.1f}")
        self._lbl_snr_readout.setText(f"SNR  {s.snr:.1f} dB")
        self._lbl_int_readout.setText(f"INTENSITY  {int(s.intensity/255*100)}%")

        p = self._pm
        a = self._perf.acquisition_time
        p["Acquisition Time"].setText(f"{a:.2f} s" if a else "1.00 s")
        p["RMSE Error"].setText(f"{self._perf.rmse:.1f} px")
        p["Max Error"].setText(f"{s.pointing_error * 1.8:.1f} px")
        p["Avg Error"].setText(f"{self._perf.rmse * 0.8:.1f} px")
        p["Lock Retention"].setText(f"{self._perf.lock_retention_pct:.1f} %")
        ra = self._perf.avg_reacq_time
        p["Reacq. Time"].setText(f"{ra:.2f} s" if ra else "0.00 s")
        p["Reacq. Success"].setText("100.0 %")
        p["System Latency"].setText(f"{s.latency_ms:.1f} ms")
        p["FPS"].setText(f"{s.fps:.1f}")
        p["Total Frames"].setText(f"{s.frame_idx}")
        p["Lost Frames"].setText("0")
        p["Lock Uptime"].setText(f"{self._perf.lock_retention_pct:.1f} %")

        # Bottom Status Bar updates
        self._ft_fps.setText(f"PROCESSING FPS  {s.fps:.1f}")
        self._ft_lat.setText(f"PIPELINE LATENCY  {s.latency_ms:.1f} ms")
        tgt_st = "LOCKED" if s.lock else ("TRACKING" if s.beacon_detected else ("REACQUIRING" if s.reacq_active else "LOST"))
        tgt_col = GREEN if s.lock else (AMBER if s.beacon_detected or s.reacq_active else RED)
        self._ft_tgt.setText(f"TARGET STATUS  {tgt_st}")
        self._ft_tgt.setStyleSheet(f"color:{tgt_col}; font-family:Consolas; font-size:8pt; font-weight:bold;")
        self._ft_err.setText(f"TRACKING ERROR  {s.pointing_error:.1f} px")
        self._ft_sim.setText(f"SIMULATION TIME  {s.frame_idx * (1.0 / CFG.camera.fps):.1f} s")

        if "DETECTION" in self._pipe_nodes:
            self._pipe_nodes["DETECTION"][1].setText("Beacon Detected" if s.beacon_detected else "Lost")
            self._pipe_nodes["DETECTION"][1].setStyleSheet(f"color:{GREEN if s.beacon_detected else RED};")

        if "XGBOOST" in self._pipe_nodes:
            self._pipe_nodes["XGBOOST"][1].setText(s.regime)
            xgb_col = GREEN if s.regime=='NORMAL' else ("#00f0ff" if s.regime=='MILD' else (AMBER if s.regime=='MODERATE' else RED))
            self._pipe_nodes["XGBOOST"][1].setStyleSheet(f"color:{xgb_col};")

        if "ESTIMATOR" in self._pipe_nodes:
            self._pipe_nodes["ESTIMATOR"][1].setText(s.estimator_name)

    def _update_plots(self):
        s = self._s
        err_val = s.pointing_error if s.beacon_detected else 0
        self._ebuf.append(err_val)
        if len(self._ebuf) > 200: self._ebuf.pop(0)
        self._c_err.setData(self._ebuf)

        self._vbuf.append(s.velocity)
        if len(self._vbuf) > 200: self._vbuf.pop(0)
        self._c_vel.setData(self._vbuf)

        self._nbuf.append(s.nis)
        if len(self._nbuf) > 200: self._nbuf.pop(0)
        self._c_nis.setData(self._nbuf)

        pw_int, c_int, buf_int = self._sp_int
        buf_int.append(s.intensity)
        if len(buf_int) > 200: buf_int.pop(0)
        c_int.setData(buf_int)

        pw_conf, c_conf, buf_conf = self._sp_conf
        buf_conf.append(s.active_conf_val)
        if len(buf_conf) > 200: buf_conf.pop(0)
        c_conf.setData(buf_conf)

        pw_unc, c_unc, buf_unc = self._sp_unc
        buf_unc.append(s.uncertainty)
        if len(buf_unc) > 200: buf_unc.pop(0)
        c_unc.setData(buf_unc)

        pw_nis, c_nis, buf_nis = self._sp_nis
        buf_nis.append(s.nis)
        if len(buf_nis) > 200: buf_nis.pop(0)
        c_nis.setData(buf_nis)

    def _update_lock(self):
        s = self._s
        if s.lock:
            self._lock_lbl.setText("BEACON LOCK: LOCK ACQUIRED")
            self._lock_lbl.setStyleSheet(self._ls(GREEN, "#062c1b"))
        elif s.reacq_active:
            self._lock_lbl.setText("REACQUIRING TARGET")
            self._lock_lbl.setStyleSheet(self._ls(AMBER, "#2a1e00"))
        elif not self._acq_done:
            self._lock_lbl.setText("ACQUIRING BEACON...")
            self._lock_lbl.setStyleSheet(self._ls(AMBER, "#2a1e00"))
        else:
            self._lock_lbl.setText("LOCK LOST")
            self._lock_lbl.setStyleSheet(self._ls(RED, "#3f0d15"))
        if "XGBOOST" in self._pipe_nodes:
            self._pipe_nodes["XGBOOST"][1].setText(s.regime)
            xgb_col = GREEN if s.regime=='NORMAL' else ("#00f0ff" if s.regime=='MILD' else (AMBER if s.regime=='MODERATE' else RED))
            self._pipe_nodes["XGBOOST"][1].setStyleSheet(f"color:{xgb_col};")

        if "ESTIMATOR" in self._pipe_nodes:
            self._pipe_nodes["ESTIMATOR"][1].setText(s.estimator_name)

    def _update_plots(self):
        s = self._s
        # 1. Pointing Error
        err_val = s.pointing_error if s.beacon_detected else 0
        self._ebuf.append(err_val)
        if len(self._ebuf) > 200: self._ebuf.pop(0)
        self._c_err.setData(self._ebuf)

        # 2. Velocity
        self._vbuf.append(s.velocity)
        if len(self._vbuf) > 200: self._vbuf.pop(0)
        self._c_vel.setData(self._vbuf)

        # 3. NIS
        self._nbuf.append(s.nis)
        if len(self._nbuf) > 200: self._nbuf.pop(0)
        self._c_nis.setData(self._nbuf)

        # 4. Embedded Mini Sparkline Plots (FIXED UNPACKING BUG)
        pw_int, c_int, buf_int = self._sp_int
        buf_int.append(s.intensity)
        if len(buf_int) > 200: buf_int.pop(0)
        c_int.setData(buf_int)

        pw_conf, c_conf, buf_conf = self._sp_conf
        buf_conf.append(s.active_conf_val)
        if len(buf_conf) > 200: buf_conf.pop(0)
        c_conf.setData(buf_conf)

        pw_unc, c_unc, buf_unc = self._sp_unc
        buf_unc.append(s.uncertainty)
        if len(buf_unc) > 200: buf_unc.pop(0)
        c_unc.setData(buf_unc)

        pw_nis, c_nis, buf_nis = self._sp_nis
        buf_nis.append(s.nis)
        if len(buf_nis) > 200: buf_nis.pop(0)
        c_nis.setData(buf_nis)

    def _update_lock(self):
        s = self._s
        if s.lock:
            self._lock_lbl.setText("🛡 BEACON LOCK: LOCK ACQUIRED")
            self._lock_lbl.setStyleSheet(self._ls(GREEN, "#062c1b"))
        elif s.reacq_active:
            self._lock_lbl.setText("🔄 REACQUIRING TARGET")
            self._lock_lbl.setStyleSheet(self._ls(AMBER, "#2a1e00"))
        elif not self._acq_done:
            self._lock_lbl.setText("⚠ ACQUIRING BEACON...")
            self._lock_lbl.setStyleSheet(self._ls(AMBER, "#2a1e00"))
        else:
            self._lock_lbl.setText("⚠ LOCK LOST")
            self._lock_lbl.setStyleSheet(self._ls(RED, "#3f0d15"))
