# scripts/11_run_all.py
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

SCRIPTS = [
    "scripts/01_load_and_peek.py",
    "scripts/02_subset_and_clean.py",
    "scripts/03_features_and_plots.py",
    "scripts/04_poc_report.py",
    "scripts/05_ml_baseline.py",
    "scripts/06_meal_outcomes.py",
    "scripts/07_pgr_model.py",
    "scripts/08_pgr_plots.py",
    "scripts/09_pgr_tree.py",
    "scripts/10_sensitivity.py",
]

def run(path):
    print(f"\n[run] {path}")
    subprocess.run([PY, str(ROOT / path)], check=True)

def main():
    print("CGM POC — end-to-end runner")
    for s in SCRIPTS:
        run(s)
    print("\nAll done. Artifacts are in data/derived/ and reports/.")

if __name__ == "__main__":
    main()