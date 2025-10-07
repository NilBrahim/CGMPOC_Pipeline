# scripts/smoke_test.py
# Runs the POC pipeline end-to-end and verifies a few basic invariants.
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# ensure project root is importable (parent of "scripts/")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    DATA_RAW, DATA_INTERIM, DATA_DERIVED, REPORTS,
    ID_COL, TS_COL, GLU_COL, TIR_LOW, TIR_HIGH
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

def run_script(name: str):
    path = SCRIPTS / name
    print(f"\n[run] python {path}")
    subprocess.run([sys.executable, str(path)], check=True)

def assert_file_exists(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Expected file not found: {p}")
    print(f"[ok] Found: {p}")

def check_daily_metrics(csv_path: Path):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    if df.empty:
        raise AssertionError("daily_metrics.csv is empty.")
    required = {"date", ID_COL, "mean", "sd", "cv", "gmi", "tir_pct"}
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"daily_metrics.csv missing columns: {missing}")
    if not ((df["tir_pct"] >= 0).all() and (df["tir_pct"] <= 100).all()):
        raise AssertionError("TIR percentages out of [0,100].")
    if not np.isfinite(df["cv"]).any():
        raise AssertionError("CV has no finite values.")
    print("[ok] daily_metrics.csv basic checks passed.")

def check_participant_summary(csv_path: Path):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise AssertionError("participant_summary.csv is empty.")
    required = {ID_COL, "mean", "sd", "cv", "gmi", "tir_pct"}
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"participant_summary.csv missing columns: {missing}")
    print("[ok] participant_summary.csv basic checks passed.")

def main():
    # 0) raw inputs present
    assert_file_exists(DATA_RAW / "combined_subset.xlsx")
    assert_file_exists(DATA_RAW / "CGMacros_dictionary.xlsx")

    # 1→4) run pipeline scripts
    run_script("01_load_and_peek.py")
    run_script("02_subset_and_clean.py")
    run_script("03_features_and_plots.py")
    run_script("04_poc_report.py")

    # 5) check outputs
    assert_file_exists(DATA_INTERIM / "poc_clean.csv")
    assert_file_exists(DATA_DERIVED / "daily_metrics.csv")
    assert_file_exists(DATA_DERIVED / "participant_summary.csv")
    assert_file_exists(REPORTS / "CGM_POC_report.html")

    check_daily_metrics(DATA_DERIVED / "daily_metrics.csv")
    check_participant_summary(DATA_DERIVED / "participant_summary.csv")

    print("\n🎉 SMOKE TEST PASSED — pipeline is healthy.")

if __name__ == "__main__":
    main()