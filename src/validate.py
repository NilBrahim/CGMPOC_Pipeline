# src/validate.py
import pandas as pd

def assert_columns(df: pd.DataFrame, required):
    """Raise an error if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def basic_cgm_sanity(df: pd.DataFrame, id_col: str, ts_col: str, glu_col: str):
    """Run simple quality checks on CGM data."""
    if df[glu_col].isna().all():
        raise ValueError("All glucose values are NaN — check data merge.")
    if (df[glu_col] < 20).any() or (df[glu_col] > 600).any():
        print("[warn] Some glucose values are outside plausible range (20–600 mg/dL).")
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        raise TypeError(f"Column {ts_col} must be datetime type.")
    if df[id_col].isna().any():
        print("[warn] Some rows missing participant ID.")
    print("[ok] Basic CGM sanity check passed.")