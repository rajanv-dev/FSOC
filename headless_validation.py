"""headless_validation.py — Runs HIGH_DYNAMIC_ADAPTIVE, NORMAL, SEVERE without GUI to measure actual metrics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from config import CFG
from simulation.environment import SimulationEnvironment
from comparison.scenario_recorder import ScenarioConfig
from comparison.baseline_tracker import BaselineFixedTracker
from comparison.comparison_runner import ProposedAdaptiveTracker
from comparison.comparison_metrics import build_comparison_result


def run_scenario(name, duration_sec=10.0, seed=42, verbose=True):
    total_frames = int(duration_sec * CFG.camera.fps)
    dt = 1.0 / CFG.camera.fps

    # --- BASELINE ---
    env = SimulationEnvironment()
    ScenarioConfig.seed_environment(env, seed)
    bt = BaselineFixedTracker()
    b_frames = []
    for f in range(total_frames):
        t = f * dt
        ScenarioConfig.apply_scenario(env, name, t, duration_sec)
        env.step(dt)
        frame = env.render()
        gt_x, gt_y = env.beacon_px, env.beacon_py
        rec, (pc, tc) = bt.process_frame(frame, dt, gt_x, gt_y, generate_overlay=False)
        b_frames.append(rec)
        if bt.acq_done:
            env.apply_pan_tilt_rate(pc, tc, dt)

    # --- PROPOSED ---
    env.reset()
    ScenarioConfig.seed_environment(env, seed)
    pt = ProposedAdaptiveTracker()
    p_frames = []
    p_events = []
    for f in range(total_frames):
        t = f * dt
        ScenarioConfig.apply_scenario(env, name, t, duration_sec)
        env.step(dt)
        frame = env.render()
        gt_x, gt_y = env.beacon_px, env.beacon_py
        rec, (pc, tc), ev = pt.process_frame(frame, dt, gt_x, gt_y, generate_overlay=False)
        p_frames.append(rec)
        if ev:
            p_events.append(ev)
        if pt.acq_done:
            env.apply_pan_tilt_rate(pc, tc, dt)

    result = build_comparison_result(name, duration_sec, b_frames, p_frames, p_events)
    return result


def print_result(result, label=""):
    b = result.baseline.summary
    p = result.proposed.summary
    imp = result.improvements
    def safe(s):
        # Replace Unicode arrows with ASCII for Windows cp1252 console
        return s.replace('\u2191', 'UP').replace('\u2193', 'DN').replace('\u2192', '->').replace('\u2192', '->')

    print(f"\n{'='*65}")
    print(f"  {label or result.scenario_name}  ({result.duration_sec:.0f}s, seed=42)")
    print(f"{'='*65}")
    print(f"  {'Metric':<22} {'Baseline':>10}  {'Proposed':>10}  {'Improvement':>14}")
    print(f"  {'-'*22} {'-'*10}  {'-'*10}  {'-'*14}")
    print(f"  {'RMSE (px)':<22} {b.rmse:>10.2f}  {p.rmse:>10.2f}  {safe(imp['Tracking RMSE']['text']):>14}")
    print(f"  {'Mean Error (px)':<22} {b.mean_pointing_error:>10.2f}  {p.mean_pointing_error:>10.2f}  {safe(imp['Mean Pointing Error']['text']):>14}")
    print(f"  {'Lock Rate (%)':<22} {b.lock_percentage:>10.1f}  {p.lock_percentage:>10.1f}  {safe(imp['Lock Retention']['text']):>14}")
    print(f"  {'Loss Rate (%)':<22} {b.target_loss_rate:>10.1f}  {p.target_loss_rate:>10.1f}  {safe(imp['Target Loss Rate']['text']):>14}")
    reacq_b  = f"{b.avg_reacquisition_time:.2f}s" if b.avg_reacquisition_time  else "N/A"
    reacq_p  = f"{p.avg_reacquisition_time:.2f}s" if p.avg_reacquisition_time  else "N/A"
    print(f"  {'Avg Reacq Time':<22} {reacq_b:>10}  {reacq_p:>10}  {safe(imp['Avg Reacquisition Time']['text']):>14}")
    print(f"  {'Avg FPS':<22} {b.avg_fps:>10.1f}  {p.avg_fps:>10.1f}  {safe(imp['Processing FPS']['text']):>14}")
    print(f"\n  Decision Events: {len(result.decision_timeline)}")
    for ev in result.decision_timeline:
        print(f"    {ev.timestamp_sec:5.2f}s | {ev.regime:<10} | {ev.estimator:<28} | conf={ev.xgb_conf:.2f}")


if __name__ == "__main__":
    print("\nFSOC Headless Validation — HIGH_DYNAMIC_ADAPTIVE Scenario\n")

    print("Running HIGH_DYNAMIC_ADAPTIVE (10s) ...")
    r_hda = run_scenario("HIGH_DYNAMIC_ADAPTIVE", duration_sec=10.0)
    print_result(r_hda, "HIGH_DYNAMIC_ADAPTIVE RESULTS")

    print("\n\nRunning NORMAL (5s) — both should be similar ...")
    r_norm = run_scenario("NORMAL", duration_sec=5.0)
    print_result(r_norm, "NORMAL (backward compat check)")

    print("\n\nRunning SEVERE (5s) — existing scenario must still work ...")
    r_sev = run_scenario("SEVERE", duration_sec=5.0)
    print_result(r_sev, "SEVERE (backward compat check)")

    # Determinism: run HIGH_DYNAMIC_ADAPTIVE twice, compare RMSE
    print("\n\nDeterminism check: running HIGH_DYNAMIC_ADAPTIVE twice ...")
    r_hda2 = run_scenario("HIGH_DYNAMIC_ADAPTIVE", duration_sec=5.0)
    r_hda3 = run_scenario("HIGH_DYNAMIC_ADAPTIVE", duration_sec=5.0)
    same = abs(r_hda2.baseline.summary.rmse - r_hda3.baseline.summary.rmse) < 0.001
    print(f"  Run 1 Baseline RMSE: {r_hda2.baseline.summary.rmse:.4f}")
    print(f"  Run 2 Baseline RMSE: {r_hda3.baseline.summary.rmse:.4f}")
    print(f"  Deterministic: {'YES' if same else 'NO - FAIL'}")

    print("\n\nALL VALIDATIONS COMPLETE")
