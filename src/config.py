# src/config.py
from pathlib import Path

# ---- Project roots ----
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_DERIVED = ROOT / "data" / "derived"
REPORTS = ROOT / "reports"

# ---- Column names ----
ID_COL = "Person_ID"
TS_COL = "Timestamp"
GLU_COL = "glucose_mgdl"

# ---- CGM params ----
RESAMPLE_RULE = "5min"
TIR_LOW, TIR_HIGH = 70, 180
HYPO_L1, HYPER_L1 = 54, 250