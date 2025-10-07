# scripts/04_poc_report.py
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

# project config
from src.config import DATA_DERIVED, REPORTS, ID_COL, TIR_LOW, TIR_HIGH, DATA_RAW

# ---------- inputs/outputs ----------
rollup_csv = DATA_DERIVED / "participant_summary.csv"   # from step 03 (filtered days)
daily_csv  = DATA_DERIVED / "daily_metrics.csv"         # filtered daily metrics
bio_xlsx   = DATA_RAW / "CGMacros_dictionary.xlsx"
plots_dir  = DATA_DERIVED
out_report = REPORTS / "CGM_POC_report.html"

BIO_SHEET   = "bio"
HBA1C_COL   = "A1c PDL (Lab)"
BIO_ID_COL  = "subject"

print("Reading:", rollup_csv)
print("Reading:", daily_csv)
print("Reading:", bio_xlsx)
print("Saving report to:", out_report)

# ---- load ----
rollup = pd.read_csv(rollup_csv)
daily  = pd.read_csv(daily_csv, parse_dates=["date"])
bio    = pd.read_excel(bio_xlsx, sheet_name=BIO_SHEET)

# keep a few attributes from bio
bio_cols_keep = [BIO_ID_COL, HBA1C_COL, "Age", "Gender", "BMI"]
bio_small = bio[[c for c in bio_cols_keep if c in bio.columns]].copy()

# normalize IDs
bio_small[BIO_ID_COL] = pd.to_numeric(bio_small[BIO_ID_COL], errors="coerce").astype("Int64")
rollup[ID_COL]        = pd.to_numeric(rollup[ID_COL], errors="coerce").astype("Int64")

# merge HbA1c/demographics
prof = rollup.merge(bio_small, left_on=ID_COL, right_on=BIO_ID_COL, how="left").drop(columns=[BIO_ID_COL])

# metabolic-state labels by HbA1c
bins   = [-np.inf, 5.7, 6.4, np.inf]
labels = ["Healthy", "Pre-diabetes", "T2D"]
prof["MetabolicState"] = pd.cut(pd.to_numeric(prof[HBA1C_COL], errors="coerce"), bins=bins, labels=labels)

# sort/round for display
order = {"Healthy":0, "Pre-diabetes":1, "T2D":2}
prof["state_order"] = prof["MetabolicState"].map(order)
prof = prof.sort_values(["state_order", ID_COL]).drop(columns=["state_order"])

for c in ["mean","sd","cv","gmi","tir_pct","below70_pct","above180_pct","below54_pct","above250_pct"]:
    if c in prof.columns:
        prof[c] = prof[c].round(2)

# save participant profile table as CSV (nice to keep)
profile_csv = DATA_DERIVED / "participant_profile.csv"
prof.to_csv(profile_csv, index=False)
print(f"Saved {profile_csv}")

# choose first filtered day per participant to show plot
first_day = (
    daily.sort_values([ID_COL, "date"])
         .groupby(ID_COL, as_index=False)
         .first()[[ID_COL, "date"]]
)
first_day["plot_file"] = first_day.apply(
    lambda r: plots_dir / f"p{int(r[ID_COL])}_{r['date'].date()}_trace.png", axis=1
)

# ----- dynamic title/intro -----
n_participants = prof[ID_COL].nunique()
state_counts = (
    prof["MetabolicState"]
    .value_counts(dropna=False)
    .rename_axis("state")
    .to_frame("n")
    .reindex(labels, fill_value=0)
)

title = f"CGM POC — {n_participants}-Participant Prototype"
ts_now = datetime.now().strftime("%Y-%m-%d %H:%M")

intro = (
    f"<p>Proof-of-concept with <b>{n_participants}</b> participants to validate the pipeline: "
    "I/O, cleaning, 5-min resampling, CGM metrics (TIR / Mean / CV / GMI), and reporting.</p>"
    f"<p>Class counts — "
    f"Healthy: <b>{int(state_counts.loc['Healthy', 'n'])}</b>, "
    f"Pre-diabetes: <b>{int(state_counts.loc['Pre-diabetes', 'n'])}</b>, "
    f"T2D: <b>{int(state_counts.loc['T2D', 'n'])}</b>.</p>"
)

# format table for report
display_cols = [ID_COL, HBA1C_COL, "MetabolicState", "mean", "sd", "cv", "gmi", "tir_pct", "below70_pct", "above180_pct"]
display_cols = [c for c in display_cols if c in prof.columns]
table_html = prof[display_cols].rename(columns={
    ID_COL: "Participant",
    HBA1C_COL: "HbA1c (%)",
    "mean":"Mean (mg/dL)",
    "sd":"SD (mg/dL)",
    "cv":"CV (%)",
    "gmi":"GMI (%)",
    "tir_pct":"TIR 70–180 (%)",
    "below70_pct":"% <70",
    "above180_pct":"% >180"
}).to_html(index=False, border=0, classes="data-table")

# small gallery of plots
cards = []
for _, r in first_day.iterrows():
    pid = int(r[ID_COL])
    d   = r["date"].date()
    img = r["plot_file"]
    if img.exists():
        # path from reports/ to data/derived/ images
        rel = Path("..") / "data" / "derived" / img.name
        cards.append(
            f"""
            <div class="card">
              <div class="card-title">Participant {pid} — {d}</div>
              <img src="{rel.as_posix()}" alt="trace p{pid}">
            </div>
            """
        )
gallery_html = "<div class='cards'>" + "\n".join(cards) + "</div>"

html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
h1 {{ margin-bottom: 4px; }}
.subtitle {{ color: #666; margin-bottom: 24px; }}
.data-table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
.data-table th, .data-table td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: right; }}
.data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 24px; }}
.card {{ border: 1px solid #eee; border-radius: 10px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
.card-title {{ font-weight: 600; margin-bottom: 8px; }}
.card img {{ width: 100%; height: auto; border-radius: 8px; }}
.footer {{ color:#777; margin-top: 24px; font-size: 0.9em; }}
.small {{ color:#777; font-size: 0.9em; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="subtitle small">Generated: {ts_now}</div>
  {intro}
  <h2>Participant Summary</h2>
  {table_html}
  <h2>Example Day — 5-min CGM Traces</h2>
  {gallery_html}
  <div class="footer">
    <p class="small">Notes: TIR={TIR_LOW}–{TIR_HIGH} mg/dL. CV = 100×SD/Mean. GMI = 3.31 + 0.02392×Mean(mg/dL).</p>
  </div>
</body>
</html>
"""

# write file
out_report.write_text(html, encoding="utf-8")
print(f"Saved HTML report -> {out_report}")