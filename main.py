"""
main.py — FSOC AI Virtual Camera Tracking System — Entry point

Flow:
  1. Splash screen → user picks Simulation or Benchmark mode
  2. Dashboard opens in chosen mode

CLI:
  python main.py --train-xgb   # train XGBoost then exit
"""
import sys, argparse

def main():
    p = argparse.ArgumentParser(description="FSOC AI Virtual Camera Tracking System")
    p.add_argument("--train-xgb", action="store_true",
                   help="Train XGBoost classifier on synthetic data and exit.")
    p.add_argument("--compare", action="store_true",
                   help="Launch directly into Adaptive Performance Comparison mode.")
    p.add_argument("--scenario", type=str, default="SEVERE",
                   help="Initial scenario for comparison mode (NORMAL, MILD, MODERATE, SEVERE, PROGRESSIVE).")
    p.add_argument("--video", "--mp4", type=str, default=None, dest="video_file",
                   help="Path to .mp4 video file to bypass virtual camera and run Benchmark Performance-2.")
    args = p.parse_args()

    if args.train_xgb:
        print("Training XGBoost disturbance classifier …")
        from ai.xgboost_classifier import DisturbanceClassifier
        ok = DisturbanceClassifier.train_and_save(8000)
        print("Saved." if ok else "Failed — check xgboost installation.")
        sys.exit(0)

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("FSOC AI Tracking System")

    # Direct Video Ingestion (Benchmark Performance-2 Bypass)
    if args.video_file:
        from gui.dashboard import FSOCDashboard
        win = FSOCDashboard(mode="benchmark", mp4_file=args.video_file)
        win.showMaximized()
        sys.exit(app.exec())

    if args.compare:
        from gui.comparison_dashboard import ComparisonDashboard
        win = ComparisonDashboard(initial_scenario=args.scenario)
        win.showMaximized()
        sys.exit(app.exec())

    from gui.splash import SplashScreen
    splash = SplashScreen()
    splash.showMaximized()
    app.exec()

    mode = splash.selected_mode or "simulation"

    if mode == "comparison":
        from gui.comparison_dashboard import ComparisonDashboard
        win = ComparisonDashboard(initial_scenario="HIGH_DYNAMIC_ADAPTIVE")
        win.showMaximized()
        sys.exit(app.exec())
    else:
        from gui.dashboard import FSOCDashboard
        win = FSOCDashboard(mode=mode,
                            mp4_file=splash.benchmark_file)
        win.showMaximized()
        sys.exit(app.exec())

if __name__ == "__main__":
    main()

