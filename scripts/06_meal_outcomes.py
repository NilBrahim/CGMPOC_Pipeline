# scripts/06_meal_outcomes.py
# Goal:
# - From tidy CGM + meal rows, build per-meal outcomes for postprandial modeling.
# - Outcomes computed on a 2h window after meal; baseline from 30m pre-meal.
# - Saves: data/derived/meal_outcomes.csv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import (
    DATA_INTERIM, DATA_DERIVED,
    ID_COL, TS_COL, GLU_COL, RESAMPLE_RULE
)

IN_CSV = DATA_INTERIM / "poc_clean.csv"
OUT_DERIVED = DATA_DERIVED
OUT_DERIVED.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DERIVED / "meal_outcomes.csv"

# --------- parameters (feel free to tweak later) ----------
BASELINE_MIN   = 30   # minutes before meal for baseline stats
WINDOW_MIN     = 120  # minutes post-meal for response
MIN_COVERAGE   = 0.60 # require >= 60% CGM points in the 2h window (>= 0.6* (120/5)=14 points)
MIN_CGM_POINTS = int((WINDOW_MIN/5) * MIN_COVERAGE)  # 14 for 2h, 5min bins
MEAL_MIN_CALS  = 50   # basic meal detection threshold
MEAL_MIN_CARBS = 10   # grams
MMOL_SUSPECT   = 25   # if median < 25, assume mmol/L and convert*18

# Candidate columns for nutrition and context
NUTR_COLS = ["Meal Type", "Calories", "Carbs", "Protein", "Fat", "Fiber", "Sugar", "Amount Consumed"]
ACTV_COLS = ["Steps", "HR", "Intensity"]

def trapezoid_minutes(y, dt_min=5):
    if len(y) < 2:
        return np.nan
    return float(np.trapezoid(y, dx=dt_min))

def build_meal_table(df):
    """Detect meal rows and return tidy meal table with key nutrition fields."""
    cols_have = [c for c in NUTR_COLS if c in df.columns]
    if not cols_have:
        raise ValueError("No nutrition columns found. Expected one of: " + ", ".join(NUTR_COLS))

    # Coerce numeric nutrition
    for c in ["Calories","Carbs","Protein","Fat","Fiber","Sugar","Amount Consumed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic meal flag: non-null Meal Type OR calories/carbs above thresholds
    has_mealtype = df["Meal Type"].notna() if "Meal Type" in df.columns else False
    enough_cals  = (df["Calories"] >= MEAL_MIN_CALS) if "Calories" in df.columns else False
    enough_carb  = (df["Carbs"]   >= MEAL_MIN_CARBS) if "Carbs"   in df.columns else False

    is_meal = has_mealtype | enough_cals | enough_carb
    meals = df.loc[is_meal, [ID_COL, TS_COL] + cols_have].copy()
    meals = meals.rename(columns={TS_COL: "meal_time"})
    meals = meals.sort_values([ID_COL, "meal_time"]).reset_index(drop=True)

    # Gap since previous meal (in hours) per participant
    meals["prev_meal_time"] = meals.groupby(ID_COL)["meal_time"].shift(1)
    meals["gap_prev_meal_hr"] = (meals["meal_time"] - meals["prev_meal_time"]).dt.total_seconds() / 3600.0
    return meals

def resample_cgm(df):
    frames = []
    for pid, g in df[[ID_COL, TS_COL, GLU_COL]].groupby(ID_COL):
        r = g.set_index(TS_COL).sort_index()[[GLU_COL]].resample(RESAMPLE_RULE).median()
        r[ID_COL] = pid
        frames.append(r.reset_index())
    r = pd.concat(frames, ignore_index=True)
    r["date"] = r[TS_COL].dt.date
    return r

def meal_outcomes_for_row(meal_row, r5, df_raw):
    pid = meal_row[ID_COL]
    t0  = meal_row["meal_time"]

    # windows
    base_start = t0 - pd.Timedelta(minutes=BASELINE_MIN)
    win_end    = t0 + pd.Timedelta(minutes=WINDOW_MIN)

    # slice resampled CGM around meal
    g_pre = r5[(r5[ID_COL]==pid) & (r5[TS_COL] >= base_start) & (r5[TS_COL] < t0)][[TS_COL, GLU_COL]].copy()
    g_win = r5[(r5[ID_COL]==pid) & (r5[TS_COL] >= t0) & (r5[TS_COL] <= win_end)][[TS_COL, GLU_COL]].copy()

    # coverage check
    if len(g_win) < MIN_CGM_POINTS or g_pre.empty:
        return None

    # baseline
    g0 = g_pre[GLU_COL].median(skipna=True)
    sd0 = g_pre[GLU_COL].std(skipna=True)

    # response stats
    g_rel = (g_win[GLU_COL] - g0).dropna()
    if g_rel.empty:
        return None

    # Δpeak & time-to-peak
    idx_max = g_rel.idxmax()
    delta_peak = float(g_rel.max())
    t_peak_min = (g_win.loc[idx_max, TS_COL] - t0).total_seconds()/60.0 if pd.notna(idx_max) else np.nan

    # incremental AUC above baseline (clip negatives to 0)
    g_pos = np.clip(g_rel.values, 0, None)
    iauc = trapezoid_minutes(g_pos, dt_min=5)

    # end at 2h & recovery
    g_end = float(g_win[GLU_COL].iloc[-1]) if not g_win.empty else np.nan
    recovery_2h = g_end - g0

    # mean/sd during window
    mean_win = float(g_win[GLU_COL].mean())
    sd_win   = float(g_win[GLU_COL].std())

    # percent time > 180 mg/dL in window (or 10 mmol/L proxy)
    pct_above_180 = float((g_win[GLU_COL] > 180).mean() * 100.0)

    # context: steps/hr in pre-window if present in raw
    steps_sum_2h = np.nan
    hr_mean_1h   = np.nan
    if "Steps" in df_raw.columns:
        s = df_raw[(df_raw[ID_COL]==pid) & (df_raw[TS_COL] >= t0 - pd.Timedelta(minutes=120)) & (df_raw[TS_COL] < t0)]["Steps"]
        steps_sum_2h = float(pd.to_numeric(s, errors="coerce").sum())
    if "HR" in df_raw.columns:
        h = df_raw[(df_raw[ID_COL]==pid) & (df_raw[TS_COL] >= t0 - pd.Timedelta(minutes=60)) & (df_raw[TS_COL] < t0)]["HR"]
        hr_mean_1h = float(pd.to_numeric(h, errors="coerce").mean())

    # time of day / weekday
    hour = t0.hour
    dow  = t0.weekday()  # 0=Mon

    # nutrition pull-through (safe with missing)
    row = {
        ID_COL: int(pid),
        "meal_time": t0,
        "gap_prev_meal_hr": meal_row.get("gap_prev_meal_hr", np.nan),
        "meal_type": meal_row.get("Meal Type", np.nan),
        "calories": meal_row.get("Calories", np.nan),
        "carbs_g":  meal_row.get("Carbs", np.nan),
        "protein_g":meal_row.get("Protein", np.nan),
        "fat_g":    meal_row.get("Fat", np.nan),
        "fiber_g":  meal_row.get("Fiber", np.nan),
        # baseline context
        "baseline_mgdl": float(g0) if pd.notna(g0) else np.nan,
        "baseline_sd":   float(sd0) if pd.notna(sd0) else np.nan,
        # postprandial outcomes
        "delta_peak":    delta_peak,
        "t_peak_min":    t_peak_min,
        "iauc_0_120":    iauc,
        "mean_0_120":    mean_win,
        "sd_0_120":      sd_win,
        "pct_above_180_0_120": pct_above_180,
        "recovery_120":  recovery_2h,
        # extra context
        "hour": hour,
        "dow":  dow,
        "steps_prev_120m": steps_sum_2h,
        "hr_mean_prev_60m": hr_mean_1h,
    }
    return row

def main():
    print("Reading:", IN_CSV)
    df = pd.read_csv(IN_CSV)

    # types
    df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")
    df = df.dropna(subset=[TS_COL, ID_COL]).sort_values([ID_COL, TS_COL])

    # unify glucose units (in case)
    med = np.nanmedian(pd.to_numeric(df[GLU_COL], errors="coerce"))
    if np.isfinite(med) and med < MMOL_SUSPECT:
        print("[info] glucose appears to be in mmol/L — converting to mg/dL.")
        df[GLU_COL] = pd.to_numeric(df[GLU_COL], errors="coerce") * 18.0
    else:
        df[GLU_COL] = pd.to_numeric(df[GLU_COL], errors="coerce")

    # build meals table
    meals = build_meal_table(df)
    print(f"Detected {len(meals)} meal rows before CGM coverage checks.")

    # resample CGM to 5-min
    r5 = resample_cgm(df)

    # compute outcomes per meal
    rows = []
    for _, mrow in meals.iterrows():
        out = meal_outcomes_for_row(mrow, r5, df)
        if out is not None:
            rows.append(out)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("No meals passed CGM coverage rules. Try lowering MIN_COVERAGE or thresholds.")
    else:
        # simple QC prints
        per_id = out_df.groupby(ID_COL).size().rename("n_meals").reset_index()
        print("\nMeals kept per participant:")
        print(per_id.to_string(index=False))
        print(f"\nTotal meals kept: {len(out_df)}")

        # save
        out_df = out_df.sort_values([ID_COL, "meal_time"]).reset_index(drop=True)
        out_df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved meal outcomes -> {OUT_CSV}")

if __name__ == "__main__":
    main()