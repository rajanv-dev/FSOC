"""
comparison_diagnostic.py — Deep per-frame pipeline diagnostic.

Proves WHY the two pipelines produce identical or different estimates.
Logs: ground truth, baseline estimate, proposed estimate, regime, estimator,
      KF state, UKF state, OF, SNR, disturbance fields every frame.

Run:  python comparison_diagnostic.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import numpy as np
from config import CFG
from simulation.environment import SimulationEnvironment
from comparison.scenario_recorder import ScenarioConfig
from comparison.baseline_tracker import BaselineFixedTracker
from comparison.comparison_runner import ProposedAdaptiveTracker

SCENARIO   = "HIGH_DYNAMIC_ADAPTIVE"
DURATION   = 10.0
SEED       = 42
LOG_EVERY  = 1           # Log every N frames (1 = all)
LOG_FRAMES = 300         # Max frames to log (keep output manageable)
SHOW_DRIFT = True        # Print frames where BASE_EST != AI_EST by > threshold
DRIFT_THR  = 0.5         # px threshold to consider "diverged"


def run_both(scenario, duration_sec, seed):
    """Run both trackers sequentially over the SAME disturbance sequence.
    Returns per-frame records for baseline and proposed."""
    total_frames = int(duration_sec * CFG.camera.fps)
    dt = 1.0 / CFG.camera.fps

    # ─── BASELINE ───────────────────────────────────────────────────────────
    env = SimulationEnvironment()
    ScenarioConfig.seed_environment(env, seed)
    bt = BaselineFixedTracker()
    b_data = []

    for f in range(total_frames):
        t = f * dt
        ScenarioConfig.apply_scenario(env, scenario, t, duration_sec)
        env.step(dt)
        frame = env.render()
        gt_x, gt_y = env.beacon_px, env.beacon_py
        d = CFG.disturbance

        rec, (pc, tc) = bt.process_frame(frame, dt, gt_x, gt_y, generate_overlay=False)
        b_data.append({
            "f": f, "t": t,
            "gt_x": gt_x, "gt_y": gt_y,
            "meas_x": bt.state_x,        # last KCF → KF measurement proxy
            "meas_y": bt.state_y,
            "est_x":  rec.estimated_x,
            "est_y":  rec.estimated_y,
            "err":    rec.pointing_error,
            "lock":   rec.locked,
            "det":    rec.detected,
            "kf_unc": rec.uncertainty,
            "extra_jit": d.extra_jitter_amplitude_px,
            "noise":  d.noise_type,
            "sigma":  d.gaussian_sigma,
            "sp_rat": d.salt_pepper_ratio,
            "force_loss": env.beacon.is_forced_loss,
            "fade":   d.beacon_fade_factor,
        })
        if bt.acq_done:
            env.apply_pan_tilt_rate(pc, tc, dt)

    # ─── PROPOSED ───────────────────────────────────────────────────────────
    env.reset()
    ScenarioConfig.seed_environment(env, seed)
    pt = ProposedAdaptiveTracker()
    p_data = []

    for f in range(total_frames):
        t = f * dt
        ScenarioConfig.apply_scenario(env, scenario, t, duration_sec)
        env.step(dt)
        frame = env.render()
        gt_x, gt_y = env.beacon_px, env.beacon_py

        rec, (pc, tc), ev = pt.process_frame(frame, dt, gt_x, gt_y, generate_overlay=False)
        p_data.append({
            "f": f, "t": t,
            "gt_x": gt_x, "gt_y": gt_y,
            "meas_x": pt.state_x,
            "meas_y": pt.state_y,
            "est_x":  rec.estimated_x,
            "est_y":  rec.estimated_y,
            "err":    rec.pointing_error,
            "lock":   rec.locked,
            "det":    rec.detected,
            "regime": rec.regime,
            "estimator": rec.estimator_name,
            "xgb":    rec.xgb_conf,
            "nis":    rec.nis,
            "unc":    rec.uncertainty,
        })
        if pt.acq_done:
            env.apply_pan_tilt_rate(pc, tc, dt)

    return b_data, p_data


def print_table(b_data, p_data):
    n = min(len(b_data), len(p_data))
    logged = 0

    print("\n" + "=" * 130)
    print(f"  PER-FRAME DIAGNOSTIC  ({SCENARIO}, {DURATION}s, seed={SEED})")
    print("=" * 130)
    hdr = (f"{'FRM':>4} {'T':>5} | {'GT_X':>6} {'GT_Y':>6} | "
           f"{'B_EST_X':>8} {'B_EST_Y':>8} {'B_ERR':>7} {'B_LK':>4} | "
           f"{'P_EST_X':>8} {'P_EST_Y':>8} {'P_ERR':>7} {'P_LK':>4} | "
           f"{'DRIFT_X':>8} {'DRIFT_Y':>8} | "
           f"{'REGIME':<10} {'ESTIMATOR':<25} {'XGB':>5} | "
           f"{'JITTER':>7} {'NOISE':<10} {'LOSS':>5}")
    print(hdr)
    print("-" * 130)

    drift_frames = 0
    max_drift = 0.0
    max_drift_frame = -1

    for i in range(n):
        b, p = b_data[i], p_data[i]
        dx = p["est_x"] - b["est_x"]
        dy = p["est_y"] - b["est_y"]
        drift = float(np.hypot(dx, dy))

        if drift > max_drift:
            max_drift = drift
            max_drift_frame = i

        if drift > DRIFT_THR:
            drift_frames += 1

        # Log all or only diverged frames
        should_log = (i % LOG_EVERY == 0) and logged < LOG_FRAMES
        if SHOW_DRIFT:
            should_log = should_log or (drift > DRIFT_THR)

        if should_log:
            logged += 1
            force = "YES" if b.get("force_loss", False) else "no"
            jitter = f"{b.get('extra_jit', 0):.0f}px"
            noise_label = f"{b.get('noise','?')[:8]}"
            regime = p.get("regime", "?")
            est    = p.get("estimator", "?")[:24]
            xgb    = p.get("xgb", 0)
            b_lk   = "Y" if b["lock"] else "N"
            p_lk   = "Y" if p["lock"] else "N"

            print(
                f"{b['f']:>4} {b['t']:>5.2f} | "
                f"{b['gt_x']:>6.1f} {b['gt_y']:>6.1f} | "
                f"{b['est_x']:>8.2f} {b['est_y']:>8.2f} {b['err']:>7.2f} {b_lk:>4} | "
                f"{p['est_x']:>8.2f} {p['est_y']:>8.2f} {p['err']:>7.2f} {p_lk:>4} | "
                f"{dx:>+8.2f} {dy:>+8.2f} | "
                f"{regime:<10} {est:<25} {xgb:>5.2f} | "
                f"{jitter:>7} {noise_label:<10} {force:>5}"
            )

    print("-" * 130)
    print(f"\n  Frames with divergence > {DRIFT_THR}px : {drift_frames}/{n} "
          f"({100*drift_frames/n:.1f}%)")
    print(f"  Max drift = {max_drift:.2f}px at frame {max_drift_frame} "
          f"(t={max_drift_frame / CFG.camera.fps:.2f}s)")
    print()


def analyze_divergence_causes(b_data, p_data):
    """Categorise the cause of each divergence frame."""
    n = min(len(b_data), len(p_data))
    causes = {"both_locked_same_meas": 0, "lock_threshold_diff": 0,
              "loss_baseline_only": 0, "loss_proposed_only": 0,
              "both_lost": 0, "regime_induced": 0}

    for i in range(n):
        b, p = b_data[i], p_data[i]
        dx = p["est_x"] - b["est_x"]
        drift = float(np.hypot(dx, p["est_y"] - b["est_y"]))

        if drift < DRIFT_THR:
            causes["both_locked_same_meas"] += 1
        elif not b["det"] and p["det"]:
            causes["loss_baseline_only"] += 1
        elif b["det"] and not p["det"]:
            causes["loss_proposed_only"] += 1
        elif not b["det"] and not p["det"]:
            causes["both_lost"] += 1
        elif b["lock"] != p["lock"]:
            causes["lock_threshold_diff"] += 1
        else:
            causes["regime_induced"] += 1

    print("  Divergence cause analysis:")
    for k, v in causes.items():
        print(f"    {k:<30}: {v:>5} frames ({100*v/n:.1f}%)")


def print_summary(b_data, p_data):
    n = min(len(b_data), len(p_data))
    b_errs = [d["err"] for d in b_data if d["det"]]
    p_errs = [d["err"] for d in p_data if d["det"]]
    b_lock = sum(1 for d in b_data if d["lock"])
    p_lock = sum(1 for d in p_data if d["lock"])

    print(f"\n  SUMMARY:")
    print(f"  Baseline  RMSE={np.sqrt(np.mean(np.square(b_errs))):.2f}px  "
          f"Mean={np.mean(b_errs):.2f}px  Lock={100*b_lock/n:.1f}%  "
          f"Det={100*sum(1 for d in b_data if d['det'])/n:.1f}%")
    print(f"  Proposed  RMSE={np.sqrt(np.mean(np.square(p_errs))):.2f}px  "
          f"Mean={np.mean(p_errs):.2f}px  Lock={100*p_lock/n:.1f}%  "
          f"Det={100*sum(1 for d in p_data if d['det'])/n:.1f}%")

    # Regime breakdown
    regimes = {}
    for d in p_data:
        r = d.get("regime", "?")
        regimes[r] = regimes.get(r, 0) + 1
    print(f"\n  Regime breakdown (proposed):")
    for r, cnt in sorted(regimes.items()):
        print(f"    {r:<12}: {cnt:>5} frames ({100*cnt/n:.1f}%)")

    # KF vs UKF proof
    print(f"\n  ROOT CAUSE ANALYSIS:")
    both_det = [(b,p) for b,p in zip(b_data, p_data) if b["det"] and p["det"]]
    if both_det:
        drifts = [float(np.hypot(p["est_x"]-b["est_x"], p["est_y"]-b["est_y"]))
                  for b,p in both_det]
        print(f"    Frames where BOTH detected  : {len(both_det)}")
        print(f"    Mean estimate drift         : {np.mean(drifts):.4f} px")
        print(f"    Max  estimate drift         : {np.max(drifts):.4f} px")
        print(f"    → If max drift ≈ 0: KF ≡ UKF for linear dynamics (confirmed)")
        print(f"    → If max drift > 1: estimators genuinely diverge")


if __name__ == "__main__":
    import cv2
    print(f"\nDiagnostic: {SCENARIO}, {DURATION}s, seed={SEED}")
    print("Running baseline...")
    b_data, p_data = run_both(SCENARIO, DURATION, SEED)

    print_table(b_data, p_data)
    analyze_divergence_causes(b_data, p_data)
    print_summary(b_data, p_data)

    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)
    both_det = [(b,p) for b,p in zip(b_data, p_data) if b["det"] and p["det"]]
    if both_det:
        drifts = [float(np.hypot(p["est_x"]-b["est_x"], p["est_y"]-b["est_y"]))
                  for b,p in both_det]
        if max(drifts) < 0.01:
            print("  CONFIRMED: KF == UKF (mathematically identical for linear dynamics).")
            print("  Improvement comes ONLY from lock threshold (90px vs 110px) and reacq speed.")
            print("  FIX NEEDED: Add adaptive Q/R to AdaptiveEstimator.set_regime() so")
            print("  the proposed system genuinely diverges from baseline during MODERATE/SEVERE.")
        else:
            print(f"  OK: Estimators diverge by up to {max(drifts):.2f}px during tracking.")
            print("  Pipeline is working correctly.")
    print()
