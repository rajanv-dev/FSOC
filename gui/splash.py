"""
gui/splash.py — Mode selection entry screen.
Animated starfield background with sleek glassmorphism mode cards.
"""
import math, random
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGraphicsDropShadowEffect, QFileDialog
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QPainter, QColor, QFont, QLinearGradient, QBrush, QPen
)


class _StarBG(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        rng = random.Random(7)
        self._stars = [(rng.random(), rng.random(), rng.random() * 2 + 0.5)
                       for _ in range(320)]
        self._t = 0.0
        QTimer(self, timeout=self._tick, interval=40).start()

    def _tick(self):
        self._t += 0.04
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        g = QLinearGradient(0, 0, 0, H)
        g.setColorAt(0, QColor(8, 12, 20))
        g.setColorAt(1, QColor(4, 6, 12))
        p.fillRect(0, 0, W, H, QBrush(g))

        for fx, fy, sz in self._stars:
            tw = 0.55 + 0.45 * math.sin(self._t * (sz + 1) + fx * 12)
            a  = int(tw * 220)
            r  = max(1, int(sz * 0.75))
            p.setBrush(QColor(210, 230, 255, a))
            p.setPen(Qt.NoPen)
            p.drawEllipse(int(fx * W) - r, int(fy * H) - r, r * 2, r * 2)

        # Earth arc decoration
        p.setBrush(QColor(16, 45, 110, 40))
        p.setPen(QPen(QColor(0, 240, 255, 30), 4))
        er = int(H * 0.95)
        p.drawEllipse(int(W * 0.65) - er, int(H * 1.15) - er, er * 2, er * 2)
        p.end()


class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_mode: str | None = None
        self.benchmark_file: str | None = None

        self.setWindowTitle("FSOC AI Virtual Camera Tracking System")
        self.setMinimumSize(1100, 680)

        self._bg = _StarBG(self)
        self._bg.setGeometry(0, 0, 9999, 9999)
        self._build()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._bg.setGeometry(0, 0, self.width(), self.height())

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addStretch(1)

        # Title
        t = QLabel("🛰 FSOC AI VIRTUAL CAMERA TRACKING SYSTEM")
        t.setAlignment(Qt.AlignCenter)
        t.setFont(QFont("Segoe UI", 22, QFont.Bold))
        t.setStyleSheet("color:#00f0ff; letter-spacing:2px; background:transparent;")
        g = QGraphicsDropShadowEffect()
        g.setBlurRadius(35)
        g.setColor(QColor("#00f0ff"))
        g.setOffset(0, 0)
        t.setGraphicsEffect(g)
        root.addWidget(t)

        sub = QLabel("Development of an AI-Based Virtual Camera Tracking System for Coarse Alignment of FSOC Terminals")
        sub.setAlignment(Qt.AlignCenter)
        sub.setFont(QFont("Segoe UI", 11))
        sub.setStyleSheet("color:#94a3b8; background:transparent; margin-top:6px;")
        root.addWidget(sub)
        root.addSpacing(42)

        lbl = QLabel("SELECT OPERATING MODE")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl.setStyleSheet("color:#64748b; letter-spacing:3px; background:transparent;")
        root.addWidget(lbl)
        root.addSpacing(20)

        row = QHBoxLayout()
        row.setSpacing(24)
        row.addStretch()

        row.addWidget(self._card("🖥", "SIMULATION MODE",
                                 "Synthetic 3D Virtual Space Environment\n\n"
                                 "• Photorealistic 1550nm IR Beacon\n"
                                 "• XGBoost Disturbance Classifier\n"
                                 "• Adaptive Kalman / Optical Flow / KCF\n"
                                 "• Closed-Loop PID Pan/Tilt Control",
                                 "#00f0ff", "simulation"))

        row.addWidget(self._card("🎬", "BENCHMARK MODE",
                                 "Video Input & Offline Performance Benchmark\n\n"
                                 "• Real MP4 Video Input Ingestion\n"
                                 "• Pipeline Tracking & Performance Evaluation\n"
                                 "• Ground Truth RMSE & Acquisition Time\n"
                                 "• Lock Retention & NIS Validation",
                                 "#a78bfa", "benchmark"))

        row.addWidget(self._card("⚖️", "ADAPTIVE COMPARISON ⭐",
                                 "Conventional Fixed vs Proposed AI Adaptive\n\n"
                                 "• 100% Identical Seed & Disturbance Sequence\n"
                                 "• Side-by-Side Synchronized Video & Telemetry\n"
                                 "• Real Measured RMSE & Pointing Error Delta\n"
                                 "• Dynamic AI Regime Timeline & CSV Export",
                                 "#10b981", "comparison"))

        row.addStretch()
        root.addLayout(row)
        root.addStretch(2)

        foot = QLabel("SIH 2024 Problem Statement  ·  Python · OpenCV · PySide6 · XGBoost · SciPy")
        foot.setAlignment(Qt.AlignCenter)
        foot.setFont(QFont("Consolas", 8))
        foot.setStyleSheet("color:#334155; background:transparent; padding-bottom:12px;")
        root.addWidget(foot)
        self.setStyleSheet("background:transparent;")

    def _card(self, icon, title, desc, col, mode):
        w = QWidget()
        w.setFixedSize(320, 290)
        w.setStyleSheet(f"QWidget{{background:rgba(15, 23, 42, 0.90);"
                        f"border:1px solid {col}44; border-radius:14px;}}"
                        f"QWidget:hover{{border:2px solid {col}; background:rgba(15, 23, 42, 0.98);}}")

        vl = QVBoxLayout(w)
        vl.setContentsMargins(24, 26, 24, 22)
        vl.setSpacing(10)

        il = QLabel(icon)
        il.setAlignment(Qt.AlignCenter)
        il.setFont(QFont("Segoe UI Emoji", 34))
        il.setStyleSheet("background:transparent; border:none;")
        vl.addWidget(il)

        tl = QLabel(title)
        tl.setAlignment(Qt.AlignCenter)
        tl.setFont(QFont("Segoe UI", 12, QFont.Bold))
        tl.setStyleSheet(f"color:{col}; background:transparent; border:none;")
        vl.addWidget(tl)

        dl = QLabel(desc)
        dl.setAlignment(Qt.AlignCenter)
        dl.setWordWrap(True)
        dl.setFont(QFont("Segoe UI", 9))
        dl.setStyleSheet("color:#94a3b8; background:transparent; border:none;")
        vl.addWidget(dl)

        vl.addStretch()

        btn = QPushButton("ENTER MODE →")
        btn.setFixedHeight(42)
        btn.setFont(QFont("Segoe UI", 9, QFont.Bold))
        btn.setStyleSheet(f"QPushButton{{background:{col}1f; color:{col};"
                          f"border:1px solid {col}66; border-radius:8px;}}"
                          f"QPushButton:hover{{background:{col}44; color:white; border:1px solid {col};}}")
        btn.clicked.connect(lambda *_, m=mode: self._select(m))
        vl.addWidget(btn)

        return w

    def _select(self, mode):
        if mode == "benchmark":
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
            if not path: return
            self.benchmark_file = path
        self.selected_mode = mode
        self.close()
