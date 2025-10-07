# scripts/01_load_and_peek.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

# central paths
from src.config import DATA_RAW

COMBINED = DATA_RAW / "combined_subset.xlsx"
DICT     = DATA_RAW / "CGMacros_dictionary.xlsx"

# ---- POC participant set (6 total; 2 per class target) ----
TARGET_IDS = {46, 35, 22, 44, 34, 17}  # update if you change the POC set

print("Looking for:", COMBINED)
print("Looking for:", DICT)
if not COMBINED.exists() or not DICT.exists():
    raise FileNotFoundError("Raw files not found. Check data/raw/ or config paths.")

# ---- load ----
df  = pd.read_excel(COMBINED, sheet_name="Sheet1")
bio = pd.read_excel(DICT, sheet_name="bio")  # participant-level labs (HbA1c)

id_col     = "Person_ID"          # in combined_subset.xlsx
bio_id_col = "subject"            # in bio sheet
hba1c_col  = "A1c PDL (Lab)"      # HbA1c column in bio

# ---- filter to the 6 target IDs (and keep only rows we need) ----
df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
df_poc = df[df[id_col].isin(TARGET_IDS)].copy()

# show which IDs are present vs missing
present_ids = set(pd.Series(df_poc[id_col].dropna().unique()).astype(int).tolist())
missing_ids = sorted(list(TARGET_IDS - present_ids))
extra_ids   = sorted(list(present_ids - TARGET_IDS))
print("\nPOC target IDs:", sorted(TARGET_IDS))
print("Present in combined_subset:", sorted(present_ids))
if missing_ids:
    print("WARNING: These target IDs are missing from combined_subset:", missing_ids)
if extra_ids:
    print("Note: These extra IDs are present (not in TARGET_IDS):", extra_ids)

# ---- attach HbA1c ----
bio[bio_id_col] = pd.to_numeric(bio[bio_id_col], errors="coerce").astype("Int64")
ids_df = (
    df_poc[[id_col]].dropna().drop_duplicates()
        .rename(columns={id_col: bio_id_col})
)
id_with_hba1c = ids_df.merge(
    bio[[bio_id_col, hba1c_col]],
    on=bio_id_col,
    how="left"
).rename(columns={bio_id_col: id_col})

# ---- derive metabolic state from HbA1c ----
bins   = [-np.inf, 5.7, 6.4, np.inf]
labels = ["Healthy", "Pre-diabetes", "T2D"]
id_with_hba1c["HbA1c_num"] = pd.to_numeric(id_with_hba1c[hba1c_col], errors="coerce")
id_with_hba1c["MetabolicState"] = pd.cut(id_with_hba1c["HbA1c_num"], bins=bins, labels=labels)

# ---- pretty print ----
print("\nParticipants in your POC with HbA1c & state:")
to_show = id_with_hba1c[[id_col, hba1c_col, "MetabolicState"]].sort_values(id_col)
print(to_show.to_string(index=False))

# ---- sanity counters per class ----
counts = (
    id_with_hba1c["MetabolicState"]
    .value_counts(dropna=False)
    .rename_axis("MetabolicState")
    .to_frame("n_participants")
    .reset_index()
)
print("\nCounts by metabolic state:")
print(counts.to_string(index=False))

# ---- extra sanity checks ----
n_missing_hba1c = id_with_hba1c[hba1c_col].isna().sum()
if n_missing_hba1c > 0:
    print(f"\nWARNING: {n_missing_hba1c} of {len(id_with_hba1c)} participants have missing HbA1c.")
    print("         Ensure all 6 POC IDs have HbA1c in bio sheet for proper labeling.")

if len(present_ids) < len(TARGET_IDS):
    print("\nNOTE: Not all target participants are present in the combined data.")
    print("      The next steps will proceed with those available, but LOPO will be less informative.")