# scripts/03_features_and_plots.py
# Purpose:
# - Load data/interim/poc_clean.csv (from step 2)
# - Resample CGM to 5-min per participant (median), robustly (no .apply pitfalls)
# - Compute daily CGM metrics (mean, SD, CV, GMI, TIR bands, hypo/hyper %)
# - Apply a daily completeness rule (>=75% of 5-min samples; i.e., >=216/288)
# - Save:
#     * daily_completeness.csv
#     * daily_days_summary.csv (new: per-ID day counts kept vs all)
#     * daily_metrics_all.csv   (before filtering)
#     * daily_metrics.csv       (after filtering; main table)
#     * participant_summary.csv (rollup over filtered days)
# - Export a simple per-participant day trace plot (prefers filtered days)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# project paths & params
from src.config import (
    DATA_INTERIM, DATA_DERIVED,
    ID_COL, TS_COL, GLU_COL,
    RESAMPLE_RULE, TIR_LOW, TIR_HIGH, HYPO_L1, HYPER_L1
)

# ---------- paths ----------
IN_CSV = DATA_INTERIM / "poc_clean.csv"
OUTDIR = DATA_DERIVED
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- completeness parameters ----------
MIN_DAY_COMPLETENESS = 0.75   # 75%
EXPECTED_SAMPLES_PER_DAY = int((24 * 60) / 5)  # 288 for 5-min bins
MIN_COUNT_PER_DAY = int(EXPECTED_SAMPLES_PER_DAY * MIN_DAY_COMPLETENESS)  # 216

# ---------- helpers ----------
def daily_metrics(g_day: pd.DataFrame) -> dict:
    """Compute daily CGM metrics from a (already resampled) day's worth of rows."""
    g = g_day[GLU_COL].dropna()
    n = len(g)
    if n == 0:
        return dict(
            n=0, mean=np.nan, sd=np.nan, cv=np.nan, gmi=np.nan,
            tir_pct=np.nan, below54_pct=np.nan, below70_pct=np.nan,
            above180_pct=np.nan, above250_pct=np.nan
        )

    mean = g.mean()
    sd = g.std(ddof=1) if n > 1 else 0.0
    cv = (sd / mean * 100.0) if mean > 0 else np.nan
    # GMI formula (Agarwal et al. 2018): GMI(%) = 3.31 + 0.02392 * mean_mgdl
    gmi = 3.31 + 0.02392 * mean

    below54 = (g < HYPO_L1).mean() * 100
    below70 = (g < TIR_LOW).mean() * 100
    in_range = ((g >= TIR_LOW) & (g <= TIR_HIGH)).mean() * 100
    above180 = (g > TIR_HIGH).mean() * 100
    above250 = (g >= HYPER_L1).mean() * 100

    return dict(
        n=n, mean=mean, sd=sd, cv=cv, gmi=gmi,
        tir_pct=in_range,
        below54_pct=below54,
        below70_pct=below70,
        above180_pct=above180,
        above250_pct=above250
    )

# ---------- load ----------
print("Reading:", IN_CSV)
df = pd.read_csv(IN_CSV, parse_dates=[TS_COL]).sort_values([ID_COL, TS_COL])

# ---------- resample per participant (robust loop) ----------
resamp_frames = []
for pid, g in df[[ID_COL, TS_COL, GLU_COL]].groupby(ID_COL):
    gg = g.set_index(TS_COL).sort_index()
    if gg.empty:
        continue
    r = gg[[GLU_COL]].resample(RESAMPLE_RULE).median()
    r[ID_COL] = pid
    resamp_frames.append(r.reset_index())

if not resamp_frames:
    raise ValueError("No data available after resampling. Check input CSV and column names.")

resamp = pd.concat(resamp_frames, ignore_index=True)

# ---------- per-day completeness ----------
resamp["date"] = resamp[TS_COL].dt.date

day_counts = (
    resamp.groupby([ID_COL, "date"], as_index=False)[GLU_COL]
          .apply(lambda s: s.notna().sum())
          .rename(columns={GLU_COL: "n_nonnull"})
)
day_counts["completeness_pct"] = (day_counts["n_nonnull"] / EXPECTED_SAMPLES_PER_DAY) * 100

# keep only sufficiently complete days
keep_days = day_counts[day_counts["n_nonnull"] >= MIN_COUNT_PER_DAY][[ID_COL, "date"]]
resamp_kept = resamp.merge(keep_days, on=[ID_COL, "date"], how="inner")

# small summary: all vs kept day counts per participant
all_day_counts = day_counts.groupby(ID_COL, as_index=False).size().rename(columns={"size": "n_days_all"})
kept_day_counts = keep_days.groupby(ID_COL, as_index=False).size().rename(columns={"size": "n_days_kept"})
days_summary = all_day_counts.merge(kept_day_counts, on=ID_COL, how="left").fillna({"n_days_kept": 0}).astype({ "n_days_kept": int })
print("\nPer-ID day counts (all vs kept):")
print(days_summary.sort_values(ID_COL).to_string(index=False))

# ---------- daily metrics (ALL days, unfiltered) ----------
daily_frames_all = []
for (pid, d), g in resamp.groupby([ID_COL, "date"]):
    m = daily_metrics(g)
    m[ID_COL], m["date"] = pid, d
    daily_frames_all.append(m)

daily_all = pd.DataFrame(daily_frames_all, columns=[
    ID_COL, "date", "n", "mean", "sd", "cv", "gmi",
    "tir_pct", "below54_pct", "below70_pct", "above180_pct", "above250_pct"
]).sort_values([ID_COL, "date"])

# ---------- daily metrics (FILTERED, kept days only) ----------
daily_frames = []
for (pid, d), g in resamp_kept.groupby([ID_COL, "date"]):
    m = daily_metrics(g)
    m[ID_COL], m["date"] = pid, d
    daily_frames.append(m)

daily = pd.DataFrame(daily_frames, columns=[
    ID_COL, "date", "n", "mean", "sd", "cv", "gmi",
    "tir_pct", "below54_pct", "below70_pct", "above180_pct", "above250_pct"
]).sort_values([ID_COL, "date"])

# ---------- per-participant rollup (mean across filtered days) ----------
agg_cols = ["mean","sd","cv","gmi","tir_pct","below70_pct","above180_pct","below54_pct","above250_pct"]
rollup = (
    daily.groupby(ID_COL)[agg_cols]
         .mean(numeric_only=True)
         .reset_index()
         .sort_values(ID_COL)
)

# ---------- save outputs ----------
completeness_path = OUTDIR / "daily_completeness.csv"
days_summary_path = OUTDIR / "daily_days_summary.csv"   # new
daily_all_path    = OUTDIR / "daily_metrics_all.csv"
daily_filt_path   = OUTDIR / "daily_metrics.csv"        # filtered becomes the main
rollup_path       = OUTDIR / "participant_summary.csv"

day_counts.to_csv(completeness_path, index=False)
days_summary.to_csv(days_summary_path, index=False)
daily_all.to_csv(daily_all_path, index=False)
daily.to_csv(daily_filt_path, index=False)
rollup.to_csv(rollup_path, index=False)

print(f"Saved completeness -> {completeness_path}")
print(f"Saved day summary  -> {days_summary_path}")
print(f"Saved all days     -> {daily_all_path}")
print(f"Saved filtered     -> {daily_filt_path}")
print(f"Saved summary      -> {rollup_path}")

# ---------- simple per-participant plot (prefer kept days; fallback to any) ----------
# pick first kept day per participant; if none kept, fall back to first available
sample_days_kept = (
    resamp_kept.groupby(ID_COL)["date"].apply(lambda s: s.iloc[0]).to_dict()
)
sample_days_any = (
    resamp.groupby(ID_COL)["date"].apply(lambda s: s.iloc[0]).to_dict()
)

sample_days = {}
for pid in set(list(sample_days_any.keys()) + list(sample_days_kept.keys())):
    sample_days[pid] = sample_days_kept.get(pid, sample_days_any.get(pid))

for pid, day in sample_days.items():
    g = resamp[(resamp[ID_COL] == pid) & (resamp["date"] == day)].copy()
    if g.empty:
        continue
    plt.figure(figsize=(10, 4))
    plt.plot(g[TS_COL], g[GLU_COL], lw=1.5)
    plt.title(f"Participant {pid} — {day} — 5-min CGM")
    plt.xlabel("Time")
    plt.ylabel("Glucose (mg/dL)")
    plt.tight_layout()
    fig_path = OUTDIR / f"p{pid}_{day}_trace.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")