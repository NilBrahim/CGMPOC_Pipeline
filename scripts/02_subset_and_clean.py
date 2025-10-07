# scripts/02_subset_and_clean.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

# use central config paths
from src.config import DATA_RAW, DATA_INTERIM

# ---------- paths ----------
RAW_COMBINED = DATA_RAW / "combined_subset.xlsx"
RAW_DICT     = DATA_RAW / "CGMacros_dictionary.xlsx"   # not used here but kept for symmetry
OUTDIR       = DATA_INTERIM
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- config ----------
# UPDATED: use your 6-participant POC set
SUBJECTS = {46, 35, 22, 44, 34, 17}

COMBINED_SHEET = "Sheet1"
BIO_SHEET      = "bio"

ID_COL   = "Person_ID"
BIO_ID_COL = "subject"
TS_COL   = "Timestamp"
DEXCOM_COL = "Dexcom GL"
LIBRE_COL  = "Libre GL"

# nutrition / activity columns we’ll keep if present
KEEP_CANDIDATES = [
    "Meal Type", "Calories", "Carbs", "Protein", "Fat", "Fiber", "Sugar",
    "Amount Consumed", "Intensity", "HR", "Steps", "RecordIndex", "Image path",
]

DROP_CANDIDATES = [
    "Unnamed: 0",           # common artifact
    "Amount Consumed ",     # duplicate name with trailing space
]

# ---------- load ----------
print("Reading:", RAW_COMBINED)
df = pd.read_excel(RAW_COMBINED, sheet_name=COMBINED_SHEET)

# Sanity check columns
missing = [c for c in [TS_COL, ID_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ---------- subset to our POC participants ----------
df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype("Int64")
df = df[df[ID_COL].isin(SUBJECTS)].copy()

present_ids = sorted([int(x) for x in df[ID_COL].dropna().unique()])
print("POC target IDs:", sorted(SUBJECTS))
print("Present after subsetting:", present_ids)

# ---------- timestamp parsing ----------
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce", utc=False)
bad_ts = df[TS_COL].isna().sum()
if bad_ts > 0:
    print(f"[warn] {bad_ts} rows had invalid timestamps and will be dropped.")
df = df.dropna(subset=[TS_COL]).sort_values([ID_COL, TS_COL])

# ---------- drop obvious junk columns if present ----------
for c in DROP_CANDIDATES:
    if c in df.columns:
        df = df.drop(columns=c)

# ---------- create unified glucose column (mg/dL) ----------
for c in [DEXCOM_COL, LIBRE_COL]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if DEXCOM_COL not in df.columns and LIBRE_COL not in df.columns:
    raise ValueError("No CGM column found (expected 'Dexcom GL' and/or 'Libre GL').")

# prefer Dexcom reading when available, else fall back to Libre
df["glucose_mgdl"] = np.where(
    df.get(DEXCOM_COL, pd.Series(index=df.index)).notna()
        if DEXCOM_COL in df.columns else False,
    df[DEXCOM_COL] if DEXCOM_COL in df.columns else np.nan,
    df[LIBRE_COL] if LIBRE_COL in df.columns else np.nan
)

# if units look like mmol/L (median < 25), convert to mg/dL
med = np.nanmedian(df["glucose_mgdl"])
if np.isfinite(med) and med < 25:
    print("[info] glucose appears to be in mmol/L — converting to mg/dL.")
    df["glucose_mgdl"] = df["glucose_mgdl"] * 18.0

# ---------- select a tidy set of columns ----------
keep_cols = [TS_COL, ID_COL, "glucose_mgdl"]
for c in KEEP_CANDIDATES:
    if c in df.columns and c not in keep_cols:
        keep_cols.append(c)

tidy = df[keep_cols].copy()

# ---------- de-duplicate same-timestamp rows per person ----------
tidy = tidy.drop_duplicates(subset=[ID_COL, TS_COL])

# ---------- small QC summaries ----------
per_id_counts = (
    tidy.groupby(ID_COL)
        .agg(n_rows=("glucose_mgdl", "size"),
             pct_missing_glucose=("glucose_mgdl", lambda s: s.isna().mean()*100))
        .reset_index()
)
print("\nPer-participant QC:")
print(per_id_counts.to_string(index=False))

# timestamp coverage
coverage = tidy.groupby(ID_COL).agg(
    start_time=(TS_COL, "min"),
    end_time=(TS_COL, "max")
).reset_index()
print("\nTime coverage by participant:")
print(coverage.to_string(index=False))

# ---------- save outputs ----------
csv_path     = OUTDIR / "poc_clean.csv"
parquet_path = OUTDIR / "poc_clean.parquet"
tidy.to_csv(csv_path, index=False)
try:
    tidy.to_parquet(parquet_path, index=False)  # needs pyarrow or fastparquet
    print(f"\nSaved: {csv_path} and {parquet_path}")
except Exception as e:
    print(f"\nSaved: {csv_path}")
    print("[note] Could not save Parquet (install 'pyarrow' to enable). Error:", e)