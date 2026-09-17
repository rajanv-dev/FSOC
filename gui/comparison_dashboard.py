"""gui/comparison_dashboard.py — Dedicated Adaptive Performance Comparison Dashboard.

Features:
  - Scientific comparison under 100% identical disturbance sequences
  - Synchronized side-by-side video playback of Existing vs Proposed methods
  - Performance Summary Table with auto-computed percentage improvements
  - 3 PyQtGraph comparative charts (Error vs Time, Lock Status vs Time, Metric Bar Chart)
  - AI Decision Timeline displaying dynamic regime transitions
  - CSV report exporter
"""
from __future__ import annotations
import os
import cv2
import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFrame, QGridLayout, QComboBox, QProgressBar,
    QSlider, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor, QImage, QPixmap

from comparison.comparison_runner import ComparisonWorker
from comparison.comparison_metrics import export_comparison_csv
from comparison.result_model import ComparisonResult, FrameRecord
from comparison.scenario_recorder import SCENARIO_DESCRIPTIONS


# ── Color Tokens ─────────────────────────────────────────────────────────────
BG      = "#04070d"
PANEL   = "#0a101d"
PANEL_B = "#142238"
ACCENT  = "#00f0ff"
GREEN   = "#10b981"
AMBER   = "#f59e0b"
RED     = "#ef4444"
TXT     = "#f8fafc"
TXT_MUT = "#94a3b8"


class ComparisonDashboard(QMainWindow):
    """Main window for the Adaptive Tracking Performance Comparison module."""

    def __init__(self, parent=None, initial_scenario="SEVERE", initial_duration=5.0):
        super().__init__(parent)
        self.setWindowTitle("FSOC AI Tracking — Adaptive Performance Comparison")
        self.setMinimumSize(1480, 920)
        self._style()

        self._scenario = initial_scenario
        self._duration = initial_duration
        self._worker: ComparisonWorker | None = None
        self._result: ComparisonResult | None = None

        # Replay playback state
        self._current_frame_idx = 0
        self._is_playing = False
        self._playback_speed = 1.0
        self._replay_timer = QTimer(self, timeout=self._on_replay_tick, interval=33)

        self._build_ui()
        # Automatically trigger initial comparison on opening
        QTimer.singleShot(300, self._start_comparison)



    def _style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {BG};
                color: {TXT};
                font-family: 'Segoe UI', -apple-system, sans-serif;
            }}
            QLabel {{ color: {TXT}; }}
            QComboBox {{
                background: {PANEL};
                color: {TXT};
                border: 1px solid {PANEL_B};
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 8.5pt;
                font-weight: bold;
            }}
            QComboBox QAbstractItemView {{
                background: {PANEL};
                color: {TXT};
                selection-background-color: #10b98144;
            }}
            QProgressBar {{
                background: {PANEL};
                border: 1px solid {PANEL_B};
                border-radius: 4px;
                text-align: center;
                color: {TXT};
                font-size: 8pt;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00f0ff, stop:1 #10b981);
                border-radius: 3px;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: {PANEL_B};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QTableWidget {{
                background: {PANEL};
                border: 1px solid {PANEL_B};
                gridline-color: {PANEL_B};
                color: {TXT};
                font-size: 8pt;
                border-radius: 4px;
            }}
            QHeaderView::section {{
                background: {BG};
                color: {TXT_MUT};
                padding: 4px 6px;
                font-weight: bold;
                border: 1px solid {PANEL_B};
            }}
        """)

    def _build_ui(self):
        cen = QWidget()
        self.setCentralWidget(cen)
        root = QVBoxLayout(cen)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # 1. Top Header & Control Setup Bar
        root.addWidget(self._mk_header_bar())

        # 1b. Scenario Description Label (hidden by default, shown when scenario has a description)
        self._lbl_scenario_desc = QLabel("")
        self._lbl_scenario_desc.setFont(QFont("Segoe UI", 7.5))
        self._lbl_scenario_desc.setStyleSheet(
            f"color:{ACCENT}; background:#001e2b; border:1px solid {ACCENT}44; "
            f"border-radius:4px; padding:4px 10px;"
        )
        self._lbl_scenario_desc.setWordWrap(True)
        self._lbl_scenario_desc.setVisible(False)
        root.addWidget(self._lbl_scenario_desc)
        self._update_scenario_desc(self._scenario)  # show on startup if needed

        # 2. Main Scrollable Workspace
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"background:{BG}; border:none;")

        scroll_content = QWidget()
        sv = QVBoxLayout(scroll_content)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(10)

        # Top Half: Side-by-Side Video & Status Viewers + Replay Controls
        sv.addWidget(self._mk_visual_comparison_section())

        # Middle: Performance Summary Table + AI Decision Timeline
        sv.addWidget(self._mk_summary_and_timeline_section())

        # Bottom: Three PyQtGraph Comparison Plots
        sv.addWidget(self._mk_charts_section())

        scroll.setWidget(scroll_content)
        root.addWidget(scroll, stretch=1)

    # ──────────────────────────────────────────────────────── 1. HEADER BAR ─
    def _mk_header_bar(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(56)
        bar.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 4, 14, 4)
        h.setSpacing(12)

        # Title
        icon = QLabel("⚖️")
        icon.setFont(QFont("Segoe UI Emoji", 16))
        h.addWidget(icon)

        tb = QVBoxLayout()
        tb.setSpacing(0)
        t1 = QLabel("ADAPTIVE PERFORMANCE COMPARISON")
        t1.setFont(QFont("Segoe UI", 11, QFont.Bold))
        t1.setStyleSheet(f"color:{ACCENT}; letter-spacing:1px;")
        t2 = QLabel("Conventional Fixed Tracking vs Proposed AI Adaptive Framework Under Equivalent Disturbance")
        t2.setFont(QFont("Segoe UI", 7.5))
        t2.setStyleSheet(f"color:{TXT_MUT};")
        tb.addWidget(t1); tb.addWidget(t2)
        h.addLayout(tb)

        h.addStretch(1)

        # Scenario Selector
        lbl_sc = QLabel("SCENARIO:")
        lbl_sc.setFont(QFont("Segoe UI", 8, QFont.Bold))
        lbl_sc.setStyleSheet(f"color:{TXT_MUT};")
        h.addWidget(lbl_sc)

        self._cb_scenario = QComboBox()
        self._cb_scenario.addItems([
            "HIGH_DYNAMIC_ADAPTIVE", "SEVERE", "MODERATE", "MILD", "NORMAL", "PROGRESSIVE"
        ])
        self._cb_scenario.setCurrentText(self._scenario)
        self._cb_scenario.currentTextChanged.connect(self._update_scenario_desc)
        h.addWidget(self._cb_scenario)

        # Duration Selector
        lbl_dur = QLabel("DURATION:")
        lbl_dur.setFont(QFont("Segoe UI", 8, QFont.Bold))
        lbl_dur.setStyleSheet(f"color:{TXT_MUT};")
        h.addWidget(lbl_dur)

        self._cb_duration = QComboBox()
        self._cb_duration.addItems(["5 sec", "10 sec", "15 sec", "30 sec"])
        self._cb_duration.setCurrentText(f"{int(self._duration)} sec")
        h.addWidget(self._cb_duration)


        # Run Button
        self._btn_run = QPushButton("▶ RUN COMPARISON")
        self._btn_run.setFixedHeight(34)
        self._btn_run.setFont(QFont("Segoe UI", 8.5, QFont.Bold))
        self._btn_run.setStyleSheet(
            f"QPushButton{{background:#065f46; color:{GREEN}; border:1px solid {GREEN}88; border-radius:4px; padding:0 16px;}}"
            f"QPushButton:hover{{background:#047857; color:white; border:1px solid {GREEN};}}"
            f"QPushButton:disabled{{background:#1e293b; color:{TXT_MUT}; border:1px solid {PANEL_B};}}"
        )
        self._btn_run.clicked.connect(self._start_comparison)
        h.addWidget(self._btn_run)

        # Export CSV Button
        self._btn_export = QPushButton("💾 EXPORT CSV")
        self._btn_export.setFixedHeight(34)
        self._btn_export.setFont(QFont("Segoe UI", 8.5, QFont.Bold))
        self._btn_export.setStyleSheet(
            f"QPushButton{{background:#1e293b; color:{ACCENT}; border:1px solid {ACCENT}66; border-radius:4px; padding:0 14px;}}"
            f"QPushButton:hover{{background:#00f0ff25; border:1px solid {ACCENT};}}"
        )
        self._btn_export.clicked.connect(self._on_export_csv)
        h.addWidget(self._btn_export)

        return bar

    # ──────────────────────────────────────────────── 2. DUAL VISUAL VIEWERS ─
    def _mk_visual_comparison_section(self) -> QWidget:
        box = QFrame()
        box.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(box)
        vl.setContentsMargins(12, 10, 12, 10)
        vl.setSpacing(8)

        # Top row: Progress Bar
        self._pbar_box = QHBoxLayout()
        self._status_lbl = QLabel("Ready to execute comparison run.")
        self._status_lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
        self._status_lbl.setStyleSheet(f"color:{TXT_MUT};")
        self._pbar_box.addWidget(self._status_lbl)

        self._pbar = QProgressBar()
        self._pbar.setFixedHeight(16)
        self._pbar.setValue(0)
        self._pbar.setVisible(False)
        self._pbar_box.addWidget(self._pbar, stretch=1)
        vl.addLayout(self._pbar_box)

        # Split Dual Camera Screens
        screens_layout = QHBoxLayout()
        screens_layout.setSpacing(12)

        # Left Screen: Existing Method
        w_left, self._lbl_cam_baseline, self._base_badges = self._mk_method_cam_card(
            title="EXISTING METHOD (CONVENTIONAL FIXED)",
            color="#f59e0b",
            is_proposed=False
        )
        screens_layout.addWidget(w_left, stretch=1)

        # Right Screen: Proposed Method
        w_right, self._lbl_cam_proposed, self._prop_badges = self._mk_method_cam_card(
            title="PROPOSED METHOD (AI ADAPTIVE TRACKING)",
            color="#00f0ff",
            is_proposed=True
        )
        screens_layout.addWidget(w_right, stretch=1)

        vl.addLayout(screens_layout)

        # Synchronized Replay Controller Bar
        rep_bar = QFrame()
        rep_bar.setFixedHeight(42)
        rep_bar.setStyleSheet(f"background:{BG}; border:1px solid {PANEL_B}; border-radius:4px;")
        rh = QHBoxLayout(rep_bar)
        rh.setContentsMargins(10, 2, 10, 2)
        rh.setSpacing(10)

        self._btn_play = QPushButton("⏸ PAUSE")
        self._btn_play.setFixedWidth(80)
        self._btn_play.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        self._btn_play.setStyleSheet(
            f"QPushButton{{background:#1e293b; color:{TXT}; border:1px solid {PANEL_B}; border-radius:3px;}}"
            f"QPushButton:hover{{background:#334155; color:{ACCENT};}}"
        )
        self._btn_play.clicked.connect(self._toggle_play)
        rh.addWidget(self._btn_play)

        self._lbl_time = QLabel("0.0s / 0.0s")
        self._lbl_time.setFont(QFont("Consolas", 8, QFont.Bold))
        self._lbl_time.setStyleSheet(f"color:{TXT_MUT};")
        rh.addWidget(self._lbl_time)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_slider_moved)
        rh.addWidget(self._slider, stretch=1)

        self._btn_speed = QPushButton("1.0x")
        self._btn_speed.setFixedWidth(50)
        self._btn_speed.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        self._btn_speed.setStyleSheet(
            f"QPushButton{{background:#1e293b; color:{TXT_MUT}; border:1px solid {PANEL_B}; border-radius:3px;}}"
            f"QPushButton:hover{{color:{TXT};}}"
        )
        self._btn_speed.clicked.connect(self._cycle_speed)
        rh.addWidget(self._btn_speed)

        vl.addWidget(rep_bar)
        return box

    def _mk_method_cam_card(self, title: str, color: str, is_proposed: bool) -> tuple[QWidget, QLabel, dict]:
        w = QFrame()
        w.setStyleSheet(f"background:{BG}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(6)

        # Header with Title
        th = QHBoxLayout()
        tl = QLabel(title)
        tl.setFont(QFont("Segoe UI", 8.5, QFont.Bold))
        tl.setStyleSheet(f"color:{color}; letter-spacing:0.5px;")
        th.addWidget(tl)
        th.addStretch()

        b_pipe = QLabel("PIPELINE: KCF + KF" if not is_proposed else "PIPELINE: XGBOOST + ADAPTIVE")
        b_pipe.setFont(QFont("Segoe UI", 7, QFont.Bold))
        b_pipe.setStyleSheet(f"color:{TXT_MUT}; background:#1e293b; border-radius:3px; padding:2px 6px;")
        th.addWidget(b_pipe)
        vl.addLayout(th)

        # Camera Display Viewport
        lbl_cam = QLabel()
        lbl_cam.setFixedSize(640, 360)
        lbl_cam.setAlignment(Qt.AlignCenter)
        lbl_cam.setStyleSheet(f"background:#020408; border:1px solid {PANEL_B}; border-radius:4px;")
        lbl_cam.setText("Awaiting Comparison Run...")
        lbl_cam.setFont(QFont("Segoe UI", 9))
        vl.addWidget(lbl_cam, alignment=Qt.AlignCenter)

        # Status Badges Row
        badges_layout = QHBoxLayout()
        badges_layout.setSpacing(6)

        badges = {}
        for key in ["LOCK", "ERROR", "ESTIMATOR", "REGIME" if is_proposed else "TYPE"]:
            item_box = QFrame()
            item_box.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:3px; padding:3px;")
            iv = QVBoxLayout(item_box)
            iv.setContentsMargins(4, 2, 4, 2)
            iv.setSpacing(0)

            k_lbl = QLabel(key)
            k_lbl.setFont(QFont("Segoe UI", 6.5, QFont.Bold))
            k_lbl.setStyleSheet(f"color:{TXT_MUT};")
            iv.addWidget(k_lbl)

            v_lbl = QLabel("--")
            v_lbl.setFont(QFont("Consolas", 7.5, QFont.Bold))
            v_lbl.setStyleSheet(f"color:{TXT};")
            iv.addWidget(v_lbl)

            badges[key] = v_lbl
            badges_layout.addWidget(item_box, stretch=1)

        vl.addLayout(badges_layout)
        return w, lbl_cam, badges

    # ──────────────────────── 3. SUMMARY TABLE & AI DECISION TIMELINE ───────
    def _mk_summary_and_timeline_section(self) -> QWidget:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {PANEL_B}; width: 4px; }}")

        # Left Side: Performance Summary Table
        w_summary = QFrame()
        w_summary.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl_sum = QVBoxLayout(w_summary)
        vl_sum.setContentsMargins(12, 10, 12, 10)
        vl_sum.setSpacing(8)

        t1 = QLabel("📊 QUANTITATIVE PERFORMANCE SUMMARY")
        t1.setFont(QFont("Segoe UI", 9, QFont.Bold))
        t1.setStyleSheet(f"color:{GREEN}; letter-spacing:0.5px;")
        vl_sum.addWidget(t1)

        self._table = QTableWidget(6, 4)
        self._table.setHorizontalHeaderLabels(["Performance Metric", "Existing (Fixed)", "Proposed (Adaptive)", "Improvement"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setFixedHeight(180)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        vl_sum.addWidget(self._table)

        splitter.addWidget(w_summary)

        # Right Side: AI Decision Timeline & Adaptive State
        w_ai = QFrame()
        w_ai.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl_ai = QVBoxLayout(w_ai)
        vl_ai.setContentsMargins(12, 10, 12, 10)
        vl_ai.setSpacing(8)

        t2 = QLabel("🤖 AI ADAPTIVE REGIME DECISION TIMELINE")
        t2.setFont(QFont("Segoe UI", 9, QFont.Bold))
        t2.setStyleSheet(f"color:{ACCENT}; letter-spacing:0.5px;")
        vl_ai.addWidget(t2)

        self._timeline_table = QTableWidget(0, 4)
        self._timeline_table.setHorizontalHeaderLabels(["Time", "Regime", "Selected Estimator", "XGBoost Conf."])
        self._timeline_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._timeline_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._timeline_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._timeline_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._timeline_table.verticalHeader().setVisible(False)
        self._timeline_table.setFixedHeight(180)
        self._timeline_table.setEditTriggers(QTableWidget.NoEditTriggers)
        vl_ai.addWidget(self._timeline_table)

        splitter.addWidget(w_ai)
        splitter.setSizes([600, 500])
        return splitter

    # ───────────────────────────────────────────── 4. COMPARATIVE PLOTS ─────
    def _mk_charts_section(self) -> QWidget:
        w = QFrame()
        w.setStyleSheet(f"background:{PANEL}; border:1px solid {PANEL_B}; border-radius:6px;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(12, 10, 12, 10)
        vl.setSpacing(8)

        tl = QLabel("📈 SYNCHRONIZED COMPARATIVE TELEMETRY CHARTS")
        tl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        tl.setStyleSheet(f"color:{TXT}; letter-spacing:0.5px;")
        vl.addWidget(tl)

        grid = QGridLayout()
        grid.setSpacing(8)

        # Plot 1: Tracking Error vs Time
        self._plot_err = pg.PlotWidget(title="GRAPH 1 — TRACKING ERROR (PIXELS) VS TIME")
        self._plot_err.setBackground(BG)
        self._plot_err.showGrid(x=True, y=True, alpha=0.2)
        self._plot_err.setLabel("bottom", "Time", units="s")
        self._plot_err.setLabel("left", "Pointing Error", units="px")
        self._plot_err.addLegend(offset=(-10, 10))
        self._curve_err_base = self._plot_err.plot(pen=pg.mkPen(AMBER, width=2, style=Qt.DashLine), name="Existing (Fixed)")
        self._curve_err_prop = self._plot_err.plot(pen=pg.mkPen(ACCENT, width=2.5), name="Proposed (AI Adaptive)")
        self._plot_err.setFixedHeight(220)
        grid.addWidget(self._plot_err, 0, 0)

        # Plot 2: Lock Status vs Time
        self._plot_lock = pg.PlotWidget(title="GRAPH 2 — LOCK STATUS (1=LOCKED, 0=LOST) VS TIME")
        self._plot_lock.setBackground(BG)
        self._plot_lock.showGrid(x=True, y=True, alpha=0.2)
        self._plot_lock.setLabel("bottom", "Time", units="s")
        self._plot_lock.setLabel("left", "Lock State")
        self._plot_lock.setYRange(-0.1, 1.15)
        self._plot_lock.addLegend(offset=(-10, 10))
        self._curve_lock_base = self._plot_lock.plot(pen=pg.mkPen(RED, width=1.8, style=Qt.DashLine), name="Existing Lock")
        self._curve_lock_prop = self._plot_lock.plot(pen=pg.mkPen(GREEN, width=2.5), name="Proposed Lock")
        self._plot_lock.setFixedHeight(220)
        grid.addWidget(self._plot_lock, 0, 1)

        # Plot 3: Metric Comparison Bar Chart
        self._plot_bar = pg.PlotWidget(title="GRAPH 3 — NORMALIZED RELATIVE METRIC COMPARISON")
        self._plot_bar.setBackground(BG)
        self._plot_bar.showGrid(x=False, y=True, alpha=0.2)
        self._plot_bar.setLabel("left", "Performance Score", units="%")
        self._plot_bar.setYRange(0, 120)
        self._plot_bar.setFixedHeight(220)
        grid.addWidget(self._plot_bar, 1, 0, 1, 2)

        vl.addLayout(grid)
        return w

    # ========================================================== EXECUTION ===
    def _update_scenario_desc(self, scenario: str):
        """Shows the scenario description banner when a scenario has a description."""
        # Normalise the internal key (strip emoji suffixes added to display names)
        key = scenario.replace(" ⭐", "").strip()
        desc = SCENARIO_DESCRIPTIONS.get(key, "")
        if desc:
            self._lbl_scenario_desc.setText(f"ℹ  {desc}")
            self._lbl_scenario_desc.setVisible(True)
        else:
            self._lbl_scenario_desc.setVisible(False)

    def _start_comparison(self):
        if self._worker and self._worker.isRunning():
            return

        raw = self._cb_scenario.currentText()
        self._scenario = raw.replace(" ⭐", "").strip()  # strip display-only emoji
        dur_str = self._cb_duration.currentText().split()[0]
        self._duration = float(dur_str)

        self._btn_run.setEnabled(False)
        self._pbar.setValue(0)
        self._pbar.setVisible(True)
        self._status_lbl.setText("Starting comparison execution...")

        self._worker = ComparisonWorker(scenario=self._scenario, duration_sec=self._duration, seed=42)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_comparison.connect(self._on_comparison_finished)
        self._worker.start()

    def _on_progress(self, pct: int, text: str):
        self._pbar.setValue(pct)
        self._status_lbl.setText(text)

    def _on_comparison_finished(self, result: ComparisonResult):
        self._result = result
        self._btn_run.setEnabled(True)
        self._pbar.setValue(100)
        self._status_lbl.setText(f"✓ Comparison Complete for scenario '{result.scenario_name}' ({result.duration_sec:.0f}s). Showing results:")

        # ── Development-only validation printout (console only, not shown in GUI) ───
        b = result.baseline.summary
        pr = result.proposed.summary
        n_events = len(result.decision_timeline)
        kf_to_ukf = sum(1 for e in result.decision_timeline if e.regime in ("MODERATE", "SEVERE"))
        severe_ev = sum(1 for e in result.decision_timeline if e.regime == "SEVERE")
        reacq_ev  = sum(1 for e in result.decision_timeline if "reacq" in e.estimator.lower())
        print("\n" + "═" * 60)
        print(f"  {result.scenario_name} VALIDATION  ({result.duration_sec:.0f}s, seed=42)")
        print("═" * 60)
        print(f"  {'Metric':<28} {'Baseline':>10}  {'Proposed':>10}")
        print(f"  {'-'*28} {'-'*10}  {'-'*10}")
        print(f"  {'RMSE (px)':<28} {b.rmse:>10.2f}  {pr.rmse:>10.2f}")
        print(f"  {'Mean Error (px)':<28} {b.mean_pointing_error:>10.2f}  {pr.mean_pointing_error:>10.2f}")
        print(f"  {'Lock Rate (%)':<28} {b.lock_percentage:>10.1f}  {pr.lock_percentage:>10.1f}")
        print(f"  {'Loss Rate (%)':<28} {b.target_loss_rate:>10.1f}  {pr.target_loss_rate:>10.1f}")
        reacq_b  = f"{b.avg_reacquisition_time:.2f}s"  if b.avg_reacquisition_time  else "N/A"
        reacq_pr = f"{pr.avg_reacquisition_time:.2f}s" if pr.avg_reacquisition_time else "N/A"
        print(f"  {'Avg Reacq Time':<28} {reacq_b:>10}  {reacq_pr:>10}")
        print(f"  {'Avg FPS':<28} {b.avg_fps:>10.1f}  {pr.avg_fps:>10.1f}")
        print(f"  Decision Events Total : {n_events}")
        print(f"  KF → UKF transitions  : {kf_to_ukf}")
        print(f"  SEVERE detections     : {severe_ev}")
        print(f"  Reacquisition events  : {reacq_ev}")
        if result.decision_timeline:
            print(f"  Timeline:")
            for ev in result.decision_timeline:
                print(f"    {ev.timestamp_sec:5.2f}s | {ev.regime:<10} | {ev.estimator:<30} | conf={ev.xgb_conf:.2f}")
        print("═" * 60 + "\n")
        # ── end validation printout ───────────────────────────────────────────

        # Update Summary Table
        self._populate_summary_table(result)

        # Update Decision Timeline
        self._populate_timeline_table(result)

        # Update Telemetry Plots
        self._populate_plots(result)

        # Setup Replay Scrubber
        n_frames = max(len(result.baseline.frames), len(result.proposed.frames))
        self._slider.setRange(0, max(0, n_frames - 1))
        self._current_frame_idx = 0
        self._slider.setValue(0)

        self._show_frame_at_index(0)
        self._is_playing = True
        self._btn_play.setText("⏸ PAUSE")
        self._replay_timer.start()

    def _populate_summary_table(self, result: ComparisonResult):
        self._table.setRowCount(0)
        for i, (name, data) in enumerate(result.improvements.items()):
            self._table.insertRow(i)

            # Metric Name
            it_name = QTableWidgetItem(name)
            it_name.setFont(QFont("Segoe UI", 8, QFont.Bold))
            self._table.setItem(i, 0, it_name)

            # Baseline Value
            b_val = data["baseline"]
            b_txt = f"{b_val:.2f} {data['unit']}" if b_val is not None else "N/A"
            it_b = QTableWidgetItem(b_txt)
            it_b.setTextAlignment(Qt.AlignCenter)
            it_b.setFont(QFont("Consolas", 8))
            self._table.setItem(i, 1, it_b)

            # Proposed Value
            p_val = data["proposed"]
            p_txt = f"{p_val:.2f} {data['unit']}" if p_val is not None else "N/A"
            it_p = QTableWidgetItem(p_txt)
            it_p.setTextAlignment(Qt.AlignCenter)
            it_p.setFont(QFont("Consolas", 8, QFont.Bold))
            it_p.setForeground(QColor(ACCENT))
            self._table.setItem(i, 2, it_p)

            # Improvement String (with directional arrow and color)
            it_imp = QTableWidgetItem(data["text"])
            it_imp.setTextAlignment(Qt.AlignCenter)
            it_imp.setFont(QFont("Segoe UI", 8.5, QFont.Bold))
            if data["is_beneficial"]:
                it_imp.setForeground(QColor(GREEN))
            else:
                it_imp.setForeground(QColor(RED))
            self._table.setItem(i, 3, it_imp)

    def _populate_timeline_table(self, result: ComparisonResult):
        self._timeline_table.setRowCount(0)
        if not result.decision_timeline:
            self._timeline_table.insertRow(0)
            it = QTableWidgetItem("0.00s")
            self._timeline_table.setItem(0, 0, it)
            self._timeline_table.setItem(0, 1, QTableWidgetItem(result.scenario_name))
            self._timeline_table.setItem(0, 2, QTableWidgetItem("Nominal State Tracking"))
            self._timeline_table.setItem(0, 3, QTableWidgetItem("0.95"))
            return

        for i, ev in enumerate(result.decision_timeline):
            self._timeline_table.insertRow(i)
            it_t = QTableWidgetItem(f"{ev.timestamp_sec:.2f}s")
            it_t.setFont(QFont("Consolas", 7.5))
            self._timeline_table.setItem(i, 0, it_t)

            it_r = QTableWidgetItem(ev.regime)
            it_r.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
            r_col = GREEN if ev.regime == "NORMAL" else (ACCENT if ev.regime == "MILD" else (AMBER if ev.regime == "MODERATE" else RED))
            it_r.setForeground(QColor(r_col))
            self._timeline_table.setItem(i, 1, it_r)

            it_e = QTableWidgetItem(ev.estimator)
            it_e.setFont(QFont("Segoe UI", 7.5))
            self._timeline_table.setItem(i, 2, it_e)

            it_c = QTableWidgetItem(f"{ev.xgb_conf:.2f}")
            it_c.setFont(QFont("Consolas", 7.5))
            it_c.setTextAlignment(Qt.AlignCenter)
            self._timeline_table.setItem(i, 3, it_c)

    def _populate_plots(self, result: ComparisonResult):
        # 1. Error Curve
        b_times = [f.t_sec for f in result.baseline.frames]
        b_errors = [f.pointing_error for f in result.baseline.frames]
        p_times = [f.t_sec for f in result.proposed.frames]
        p_errors = [f.pointing_error for f in result.proposed.frames]

        self._curve_err_base.setData(b_times, b_errors)
        self._curve_err_prop.setData(p_times, p_errors)

        # 2. Lock Curve
        b_locks = [1.0 if f.locked else 0.0 for f in result.baseline.frames]
        p_locks = [1.0 if f.locked else 0.0 for f in result.proposed.frames]

        self._curve_lock_base.setData(b_times, b_locks)
        self._curve_lock_prop.setData(p_times, p_locks)

        # 3. Bar Chart Comparison (Normalized)
        self._plot_bar.clear()
        categories = ["RMSE (px)", "Mean Err (px)", "Lock Rate (%)", "Loss Rate (%)", "FPS"]
        b_vals = [
            result.baseline.summary.rmse,
            result.baseline.summary.mean_pointing_error,
            result.baseline.summary.lock_percentage,
            result.baseline.summary.target_loss_rate,
            result.baseline.summary.avg_fps,
        ]
        p_vals = [
            result.proposed.summary.rmse,
            result.proposed.summary.mean_pointing_error,
            result.proposed.summary.lock_percentage,
            result.proposed.summary.target_loss_rate,
            result.proposed.summary.avg_fps,
        ]

        x = np.arange(len(categories))
        width = 0.35

        bar_base = pg.BarGraphItem(x=x - width/2, height=b_vals, width=width, brush=pg.mkBrush(AMBER))
        bar_prop = pg.BarGraphItem(x=x + width/2, height=p_vals, width=width, brush=pg.mkBrush(ACCENT))

        self._plot_bar.addItem(bar_base)
        self._plot_bar.addItem(bar_prop)

        # Set string category ticks on X Axis
        ax = self._plot_bar.getAxis("bottom")
        ticks = [(i, categories[i]) for i in range(len(categories))]
        ax.setTicks([ticks])

    # ========================================================== REPLAY LOOP ==
    def _toggle_play(self):
        self._is_playing = not self._is_playing
        self._btn_play.setText("⏸ PAUSE" if self._is_playing else "▶ PLAY")
        if self._is_playing:
            self._replay_timer.start()
        else:
            self._replay_timer.stop()

    def _cycle_speed(self):
        speeds = [0.5, 1.0, 2.0]
        idx = (speeds.index(self._playback_speed) + 1) % len(speeds)
        self._playback_speed = speeds[idx]
        self._btn_speed.setText(f"{self._playback_speed}x")
        self._replay_timer.setInterval(int(33 / self._playback_speed))

    def _on_slider_moved(self, val: int):
        self._current_frame_idx = val
        self._show_frame_at_index(val)

    def _on_replay_tick(self):
        if not self._result:
            return
        n_frames = max(len(self._result.baseline.frames), len(self._result.proposed.frames))
        if n_frames == 0:
            return

        self._current_frame_idx = (self._current_frame_idx + 1) % n_frames
        self._slider.blockSignals(True)
        self._slider.setValue(self._current_frame_idx)
        self._slider.blockSignals(False)

        self._show_frame_at_index(self._current_frame_idx)

    def _show_frame_at_index(self, idx: int):
        if not self._result:
            return

        bf: FrameRecord | None = self._result.baseline.frames[idx] if idx < len(self._result.baseline.frames) else None
        pf: FrameRecord | None = self._result.proposed.frames[idx] if idx < len(self._result.proposed.frames) else None

        t_sec = bf.t_sec if bf else (pf.t_sec if pf else 0.0)
        self._lbl_time.setText(f"{t_sec:.1f}s / {self._result.duration_sec:.0f}s")

        # Update Baseline Camera Canvas
        if bf and bf.preview_frame is not None:
            self._display_pixmap(self._lbl_cam_baseline, bf.preview_frame)
            self._base_badges["LOCK"].setText("LOCKED" if bf.locked else "LOST")
            self._base_badges["LOCK"].setStyleSheet(f"color:{GREEN if bf.locked else RED}; font-weight:bold;")
            self._base_badges["ERROR"].setText(f"{bf.pointing_error:.1f} px")
            self._base_badges["ESTIMATOR"].setText(bf.estimator_name)
            self._base_badges["TYPE"].setText("Fixed Baseline")

        # Update Proposed Camera Canvas
        if pf and pf.preview_frame is not None:
            self._display_pixmap(self._lbl_cam_proposed, pf.preview_frame)
            self._prop_badges["LOCK"].setText("LOCKED" if pf.locked else "LOST")
            self._prop_badges["LOCK"].setStyleSheet(f"color:{GREEN if pf.locked else RED}; font-weight:bold;")
            self._prop_badges["ERROR"].setText(f"{pf.pointing_error:.1f} px")
            self._prop_badges["ESTIMATOR"].setText(pf.estimator_name)
            self._prop_badges["REGIME"].setText(pf.regime)
            r_col = GREEN if pf.regime == "NORMAL" else (ACCENT if pf.regime == "MILD" else (AMBER if pf.regime == "MODERATE" else RED))
            self._prop_badges["REGIME"].setStyleSheet(f"color:{r_col}; font-weight:bold;")

    def _display_pixmap(self, label: QLabel, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qi).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_export_csv(self):
        if not self._result:
            QMessageBox.warning(self, "Export CSV", "No comparison results to export. Run a comparison first.")
            return

        default_path = os.path.join(os.path.expanduser("~"), "Downloads", f"FSOC_Comparison_{self._result.scenario_name}.csv")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Comparison Report", default_path, "CSV Files (*.csv)")
        if not file_path:
            return

        ok = export_comparison_csv(self._result, file_path)
        if ok:
            QMessageBox.information(self, "Export Complete", f"Comparison report successfully saved to:\n{file_path}")
        else:
            QMessageBox.critical(self, "Export Error", "Failed to write CSV file. Check permissions.")
