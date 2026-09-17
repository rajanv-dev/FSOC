"""
vision/acquisition.py — 1550 nm IR Beacon Acquisition

On a real InGaAs/SWIR camera, a 1550 nm laser beacon appears as the
SINGLE BRIGHTEST spot in the image — far outshining background illumination,
stars, or Earth albedo.

Strategy:
  1. High-threshold pass: isolate pixels > IR_THRESHOLD (the beacon)
  2. If found → connected components, score by brightness × SNR
  3. If not found → fall back to OTSU (for noisy/disturbed frames)
"""
import numpy as np
import cv2

IR_THRESHOLD = 180   # pixels above this are "definitely IR beacon"
OTSU_FALLBACK = True


def acquire_beacon(frame: np.ndarray) -> list[tuple[int, int, float]]:
    """
    Returns list of (cx, cy, score) sorted descending by score.

    Primary path:  high-threshold → blob detection (incorporating red light prominence)
    Fallback path: OTSU adaptive threshold
    """
    candidates = _high_threshold_detect(frame)
    if not candidates and OTSU_FALLBACK:
        candidates = _otsu_detect(frame)
    return candidates


def _preprocess_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Helper to derive processing map (combining brightness and red prominence) and BGR frame."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b, g, r = frame[:, :, 0].astype(float), frame[:, :, 1].astype(float), frame[:, :, 2].astype(float)
        red_prom = np.clip(1.8 * r - g - b, 0, 255).astype(np.uint8)
        proc = cv2.addWeighted(gray, 0.4, red_prom, 0.6, 0)
        return proc, frame
    return frame, None


def _high_threshold_detect(frame: np.ndarray) -> list[tuple[int, int, float]]:
    """Find the brightest red beacon blob hot-spot."""
    proc, bgr = _preprocess_frame(frame)
    blurred = cv2.GaussianBlur(proc, (3, 3), 0)

    # Binary mask: bright & red-boosted beacon pixels
    thresh_val = min(IR_THRESHOLD, int(blurred.max() * 0.7)) if blurred.max() > 100 else IR_THRESHOLD
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    candidates = []

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 2 or area > 3000:
            continue

        cx_f, cy_f = float(centroids[i][0]), float(centroids[i][1])
        cx, cy = int(cx_f), int(cy_f)

        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 6:
            continue

        # Compute SNR against surrounding annulus background
        H, W = proc.shape[:2]
        r_sig = max(int(np.sqrt(area) * 1.5), 8)
        r_ann = r_sig + 16
        x0 = max(0, cx - r_ann); y0 = max(0, cy - r_ann)
        x1 = min(W, cx + r_ann); y1 = min(H, cy + r_ann)
        roi = proc[y0:y1, x0:x1].astype(np.float32)
        ys, xs = np.ogrid[y0:y1, x0:x1]
        d2 = (xs - cx)**2 + (ys - cy)**2
        sig_mask = d2 <= r_sig**2
        ann_mask = (d2 > r_sig**2) & (d2 <= r_ann**2)

        sig_roi = roi[sig_mask[:roi.shape[0], :roi.shape[1]]] if np.any(sig_mask) else np.array([0])
        ann_roi = roi[ann_mask[:roi.shape[0], :roi.shape[1]]] if np.any(ann_mask) else np.array([1])

        peak    = float(sig_roi.max()) if sig_roi.size else 0.0
        bg_mean = float(ann_roi.mean()) if ann_roi.size else 0.0
        bg_std  = float(ann_roi.std())  + 1e-6

        snr   = (peak - bg_mean) / bg_std
        red_factor = 1.0
        if bgr is not None and 0 <= cy < H and 0 <= cx < W:
            pix_b, pix_g, pix_r = bgr[cy, cx].astype(float)
            if pix_r > pix_g and pix_r > pix_b:
                red_factor = 1.5 + (pix_r - max(pix_g, pix_b)) / 255.0

        score = snr * min(area, 300) / 300.0 * red_factor

        if score > 0.8:
            candidates.append((cx, cy, float(score)))

    candidates.sort(key=lambda x: -x[2])
    return candidates[:3]


def _otsu_detect(frame: np.ndarray) -> list[tuple[int, int, float]]:
    """Adaptive OTSU fallback for heavily disturbed / low-SNR frames."""
    proc, bgr = _preprocess_frame(frame)
    blurred = cv2.GaussianBlur(proc, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    candidates = []

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4 or area > 2000:
            continue
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        w  = stats[i, cv2.CC_STAT_WIDTH]
        h  = stats[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) / max(min(w, h), 1) > 5:
            continue
        H_f, W_f = proc.shape[:2]
        roi  = proc[max(0, cy-12):cy+13, max(0, cx-12):cx+13]
        snr  = (roi.max() - roi.mean()) / (roi.std() + 1e-6) if roi.size else 0
        score = float(snr) * min(area, 200) / 200.0
        if score > 0.8:
            candidates.append((cx, cy, score))

    candidates.sort(key=lambda x: -x[2])
    return candidates[:3]

