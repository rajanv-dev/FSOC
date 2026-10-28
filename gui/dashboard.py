"""
gui/dashboard.py — FSOC AI Virtual Camera Tracking System (Aerospace Command Center UI)

Re-exports the newly redesigned Aerospace Command Center Dashboard from main.py
to preserve 100% backward compatibility with all imports and benchmark modules.
"""
from __future__ import annotations
from main import (
    FSOCDashboard,
    ResponsiveVideoViewport,
    SearchRegionWidget,
    _TelemetryState
)

__all__ = ["FSOCDashboard", "ResponsiveVideoViewport", "SearchRegionWidget", "_TelemetryState"]
