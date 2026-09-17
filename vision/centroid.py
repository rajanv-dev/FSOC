"""vision/centroid.py — Intensity-weighted centroid computation."""
import numpy as np
from config import CFG

def compute_centroid(frame: np.ndarray, cx: int, cy: int,
                     radius: int = 18) -> tuple[float, float, float]:
    H, W = frame.shape[:2]
    x0 = max(0, cx - radius); y0 = max(0, cy - radius)
    x1 = min(W, cx + radius); y1 = min(H, cy + radius)
    roi_raw = frame[y0:y1, x0:x1]
    if roi_raw.ndim == 3:
        # Weight by red channel prominence for red beacon tracking
        b, g, r = roi_raw[:, :, 0].astype(np.float64), roi_raw[:, :, 1].astype(np.float64), roi_raw[:, :, 2].astype(np.float64)
        roi = np.clip(1.8 * r - g - b, 1.0, 255.0)
    else:
        roi = roi_raw.astype(np.float64)

    total = roi.sum()
    if total < 1e-6:
        return float(cx), float(cy), 0.0
    ys, xs = np.mgrid[y0:y1, x0:x1]
    xc = float((xs * roi).sum() / total)
    yc = float((ys * roi).sum() / total)
    return xc, yc, float(total)

def compute_snr(frame: np.ndarray, cx: int, cy: int) -> float:
    H, W = frame.shape[:2]
    r_sig = 10; r_ann = 22
    x0 = max(0, cx - r_ann); y0 = max(0, cy - r_ann)
    x1 = min(W, cx + r_ann); y1 = min(H, cy + r_ann)
    roi_raw = frame[y0:y1, x0:x1]
    if roi_raw.ndim == 3:
        b, g, r = roi_raw[:, :, 0].astype(np.float32), roi_raw[:, :, 1].astype(np.float32), roi_raw[:, :, 2].astype(np.float32)
        roi = np.clip(1.8 * r - g - b, 0.0, 255.0)
    else:
        roi = roi_raw.astype(np.float32)
    ys, xs = np.ogrid[y0:y1, x0:x1]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    sig_mask = d2 <= r_sig ** 2
    ann_mask = (d2 > r_sig ** 2) & (d2 <= r_ann ** 2)
    if not np.any(sig_mask) or not np.any(ann_mask):
        return 0.0
    peak  = float(roi[sig_mask].mean())
    noise = float(roi[ann_mask].std()) + 1e-6
    return (peak - float(roi[ann_mask].mean())) / noise

