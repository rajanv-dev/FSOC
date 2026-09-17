"""
simulation/satellite_renderer.py
Procedurally draws the receiver satellite with a 1550 nm IR beacon aperture.

Geometry (all sizes in pixels, canvas = sz×sz, satellite centred):
  ┌────────────────────────────────────────────┐
  │                  [dish]                     │
  │              [bus body]                     │
  │  [panel L]  [aperture]  [panel R]           │
  └────────────────────────────────────────────┘

The IR aperture is rendered as an intensely bright white-cyan disc with a
Gaussian halo so the acquisition pipeline's high-threshold pass always finds
it as the peak-brightness blob in the frame.
"""
import math
import sys
from pathlib import Path
import numpy as np
import cv2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ASSET_DIR = _PROJECT_ROOT / "assets"

def _load_satellite_img() -> np.ndarray | None:
    path = _ASSET_DIR / "satellite.png"
    if not path.exists():
        print(f"[ERROR] Satellite asset NOT FOUND at {path.resolve()}", file=sys.stderr)
        return None
    print(f"SATELLITE ASSET: FOUND {path.resolve()}", file=sys.stderr)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img

_SATELLITE_IMG = _load_satellite_img()


_SAT_CACHE: dict = {}

def draw_satellite(canvas: np.ndarray,
                   cx: int, cy: int,
                   size: int = 80,
                   angle_deg: float = 0.0,
                   ir_phase: float = 0.0) -> tuple[int, int]:
    """
    Draw receiver satellite onto a frame using real satellite asset assets/satellite.png.
    """
    s   = max(size, 40)
    is_bgr = (canvas.ndim == 3)
    tmp = np.zeros((s * 2, s * 2, 3), dtype=np.uint8) if is_bgr else np.zeros((s * 2, s * 2), dtype=np.uint8)

    lx  = s        # local centre x
    ly  = s        # local centre y
    bw = s // 3;   bh = s // 5

    if _SATELLITE_IMG is not None:

        cache_key = (s, is_bgr)
        if cache_key in _SAT_CACHE:
            bgr_sat, alpha = _SAT_CACHE[cache_key]
        else:
            sat_sub = cv2.resize(_SATELLITE_IMG, (int(s * 1.5), int(s * 1.5)), interpolation=cv2.INTER_AREA)
            if is_bgr:
                if sat_sub.ndim == 3 and sat_sub.shape[2] == 4:
                    bgr_sat = sat_sub[:, :, :3]
                    alpha = (sat_sub[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
                elif sat_sub.ndim == 3:
                    bgr_sat = sat_sub
                    alpha = ((cv2.cvtColor(bgr_sat, cv2.COLOR_BGR2GRAY) > 10).astype(np.float32))[:, :, None]
                else:
                    bgr_sat = cv2.cvtColor(sat_sub, cv2.COLOR_GRAY2BGR)
                    alpha = ((sat_sub > 10).astype(np.float32))[:, :, None]
            else:
                bgr_sat = cv2.cvtColor(sat_sub[:, :, :3], cv2.COLOR_BGR2GRAY) if sat_sub.ndim == 3 else sat_sub
                alpha = ((bgr_sat > 10).astype(np.float32))
            if len(_SAT_CACHE) > 10:
                _SAT_CACHE.clear()
            _SAT_CACHE[cache_key] = (bgr_sat, alpha)

        sh, sw = bgr_sat.shape[:2]
        ox = lx - sw // 2; oy = ly - sh // 2
        roi = tmp[oy:oy+sh, ox:ox+sw].astype(np.float32)
        tmp[oy:oy+sh, ox:ox+sw] = np.clip(roi * (1.0 - alpha) + bgr_sat * alpha, 0, 255).astype(np.uint8)

    else:
        # Fallback bus body
        col = (140, 140, 140) if is_bgr else 140
        cv2.rectangle(tmp, (lx - bw, ly - bh), (lx + bw, ly + bh), col, -1)

    # Solar panels
    pw = s // 2; ph = bh - 2
    p_col = (100, 40, 20) if is_bgr else 30
    g_col = (140, 70, 40) if is_bgr else 50
    for side in (-1, 1):
        px0 = lx + side * bw; px1 = lx + side * (bw + pw)
        x0, x1 = (min(px0, px1), max(px0, px1))
        cv2.rectangle(tmp, (x0, ly - ph), (x1, ly + ph), p_col, -1)
        cols = 5; rows = 3
        cw = (x1 - x0) // cols; ch = (2 * ph) // rows
        for ci in range(1, cols):
            cv2.line(tmp, (x0 + ci * cw, ly - ph), (x0 + ci * cw, ly + ph), g_col, 1)
        for ri in range(1, rows):
            cv2.line(tmp, (x0, ly - ph + ri * ch), (x1, ly - ph + ri * ch), g_col, 1)

    # ── 1550 nm Beacon Aperture (Vivid Red Light) ─────────────────────
    ap_x = lx + bw - 1
    ap_y = ly
    ap_r = max(3, s // 16)

    pulse = 0.85 + 0.15 * math.sin(ir_phase)
    peak  = int(255 * pulse)

    # Multi-layer Red Gaussian Glow & Halo
    for sigma_mult, alpha in [(4.0, 0.20), (2.5, 0.45), (1.2, 0.80)]:
        sigma = ap_r * sigma_mult
        spread = int(sigma * 3)
        x0_b = max(0, ap_x - spread); x1_b = min(tmp.shape[1], ap_x + spread + 1)
        y0_b = max(0, ap_y - spread); y1_b = min(tmp.shape[0], ap_y + spread + 1)
        if x1_b > x0_b and y1_b > y0_b:
            gx = np.arange(x0_b, x1_b, dtype=np.float32) - ap_x
            gy = np.arange(y0_b, y1_b, dtype=np.float32) - ap_y
            xx, yy = np.meshgrid(gx, gy)
            blob = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            
            if is_bgr:
                glow_r = np.clip(blob * peak * alpha, 0, 255).astype(np.uint8)
                glow_g = np.clip(blob * (peak * 0.2) * alpha, 0, 255).astype(np.uint8)
                glow_b = np.clip(blob * (peak * 0.1) * alpha, 0, 255).astype(np.uint8)
                roi = tmp[y0_b:y1_b, x0_b:x1_b].astype(np.int32)
                roi[:, :, 2] = np.clip(roi[:, :, 2] + glow_r, 0, 255)
                roi[:, :, 1] = np.clip(roi[:, :, 1] + glow_g, 0, 255)
                roi[:, :, 0] = np.clip(roi[:, :, 0] + glow_b, 0, 255)
                tmp[y0_b:y1_b, x0_b:x1_b] = roi.astype(np.uint8)
            else:
                glow = np.clip(blob * peak * alpha, 0, 255).astype(np.uint8)
                roi  = tmp[y0_b:y1_b, x0_b:x1_b]
                tmp[y0_b:y1_b, x0_b:x1_b] = np.clip(roi.astype(np.int32) + glow.astype(np.int32), 0, 255).astype(np.uint8)

    # Red Beacon aperture disc
    red_core = (30, 30, peak) if is_bgr else peak
    red_ring = (100, 100, 255) if is_bgr else min(255, peak + 30)
    cv2.circle(tmp, (ap_x, ap_y), ap_r, red_core, -1)
    cv2.circle(tmp, (ap_x, ap_y), ap_r + 1, red_ring, 1)

    # Rotate the whole tmp canvas
    M = cv2.getRotationMatrix2D((lx, ly), angle_deg, 1.0)
    tmp = cv2.warpAffine(tmp, M, (tmp.shape[1], tmp.shape[0]))

    theta = math.radians(-angle_deg)
    dx, dy = ap_x - lx, ap_y - ly
    ap_rx = int(lx + dx * math.cos(theta) - dy * math.sin(theta))
    ap_ry = int(ly + dx * math.sin(theta) + dy * math.cos(theta))

    # Composite tmp onto canvas
    H, W = canvas.shape[:2]
    th, tw = tmp.shape[:2]
    ox = cx - lx; oy = cy - ly

    sx0 = max(0, -ox);    sy0 = max(0, -oy)
    sx1 = min(tw, W - ox); sy1 = min(th, H - oy)
    dx0 = max(0, ox);     dy0 = max(0, oy)
    dx1 = dx0 + (sx1 - sx0); dy1 = dy0 + (sy1 - sy0)

    if dx1 > dx0 and dy1 > dy0 and sx1 > sx0 and sy1 > sy0:
        src = tmp[sy0:sy1, sx0:sx1]
        dst = canvas[dy0:dy1, dx0:dx1]
        canvas[dy0:dy1, dx0:dx1] = np.maximum(dst, src)

    beacon_x = int(np.clip(ap_rx + ox, 0, W - 1))
    beacon_y = int(np.clip(ap_ry + oy, 0, H - 1))
    return beacon_x, beacon_y
