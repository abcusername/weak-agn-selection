# QC_06_center_check_table.py
# ---------------------------------------------------
# Build a "center consistency" table from cleaning logs
# - Input: r-band cleaning log + g-band cleaning log
# - Output: a seed table to manually fill SPEC_center_ok
#
# Key output columns:
#   TARGETID
#   r_status, g_status
#   sep_r_arcsec, sep_g_arcsec
#   VAR_center_ok_r, VAR_center_ok_g, VAR_center_ok_both
#   chosen_oid_r, chosen_oid_g
#   n_clean_r, n_clean_g
#   SPEC_center_ok (blank, to be filled manually)
#   note (blank)
#
# You can later merge this with your DESI spectra checks.

import os
import re
import numpy as np
import pandas as pd

# -------------------------
# USER SETTINGS
# -------------------------
R_LOG = r"C:\Users\30126\Desktop\AGN\ZTF_lightcurves_CLEAN_R_MINFIX\cleaning_log_SEYFERT.csv"
G_LOG = r"C:\Users\30126\Desktop\AGN\ZTF_lightcurves_CLEAN_GRi_MINFIX\cleaning_log_SEYFERT_g.csv"

BASE = r"C:\Users\30126\Desktop\AGN"
OUTDIR = os.path.join(BASE, r"TO_ADVISOR_CHECK_MINFIX\06_center_check")
os.makedirs(OUTDIR, exist_ok=True)
OUTCSV = os.path.join(OUTDIR, "seyfert_center_check_seed.csv")

# Threshold for "centered" (arcsec)
CENTER_TH_ARCSEC = 1.0  # try 1.0 first; can relax to 1.5 later

# -------------------------
# Helpers
# -------------------------
def normalize_targetid(x):
    """Make sure TARGETID is a pure digit string."""
    s = str(x).strip()
    # remove possible trailing band markers like 'g' or non-digits
    s2 = re.sub(r"\D", "", s)
    return s2 if s2 != "" else s

def load_log(path, band_label):
    df = pd.read_csv(path)

    if "TARGETID" not in df.columns:
        raise RuntimeError(f"TARGETID not found in {path}. cols={list(df.columns)}")

    df["TARGETID_norm"] = df["TARGETID"].apply(normalize_targetid)

    # expected columns
    for c in ["status", "reason", "chosen_oid", "chosen_sep_arcsec", "n_clean", "n_oid"]:
        if c not in df.columns:
            # create empty if missing (robust)
            df[c] = np.nan

    # cast numeric
    df["chosen_sep_arcsec"] = pd.to_numeric(df["chosen_sep_arcsec"], errors="coerce")
    df["n_clean"] = pd.to_numeric(df["n_clean"], errors="coerce")
    df["n_oid"] = pd.to_numeric(df["n_oid"], errors="coerce")

    # rename columns with band prefix
    keep = df[[
        "TARGETID_norm", "TARGETID", "status", "reason",
        "chosen_oid", "chosen_sep_arcsec", "n_clean", "n_oid"
    ]].copy()

    keep = keep.rename(columns={
        "TARGETID": f"TARGETID_{band_label}",
        "status": f"{band_label}_status",
        "reason": f"{band_label}_reason",
        "chosen_oid": f"chosen_oid_{band_label}",
        "chosen_sep_arcsec": f"sep_{band_label}_arcsec",
        "n_clean": f"n_clean_{band_label}",
        "n_oid": f"n_oid_{band_label}",
    })

    return keep

def center_ok(sep_arcsec, status):
    """Return Y/N based on sep and status OK."""
    if str(status).strip().upper() != "OK":
        return "N"
    if not np.isfinite(sep_arcsec):
        return "N"
    return "Y" if sep_arcsec <= CENTER_TH_ARCSEC else "N"

# -------------------------
# Main
# -------------------------
def main():
    r = load_log(R_LOG, "r")
    g = load_log(G_LOG, "g")

    # Outer merge: keep all IDs appearing in either log
    m = pd.merge(r, g, on="TARGETID_norm", how="outer")

    # Choose a single TARGETID column for display
    m["TARGETID"] = m["TARGETID_r"].fillna(m["TARGETID_g"])
    m["TARGETID"] = m["TARGETID"].apply(normalize_targetid)

    # Compute center flags
    m["VAR_center_ok_r"] = m.apply(lambda row: center_ok(row.get("sep_r_arcsec"), row.get("r_status")), axis=1)
    m["VAR_center_ok_g"] = m.apply(lambda row: center_ok(row.get("sep_g_arcsec"), row.get("g_status")), axis=1)
    m["VAR_center_ok_both"] = np.where((m["VAR_center_ok_r"] == "Y") & (m["VAR_center_ok_g"] == "Y"), "Y", "N")

    # Prepare columns for manual spectra check
    m["SPEC_center_ok"] = ""  # fill with Y/N/unclear manually
    m["note"] = ""            # optional notes (e.g., off-center spectra, multiple sources, etc.)

    # Order columns
    out = m[[
        "TARGETID",
        "r_status", "sep_r_arcsec", "n_clean_r", "chosen_oid_r",
        "g_status", "sep_g_arcsec", "n_clean_g", "chosen_oid_g",
        "VAR_center_ok_r", "VAR_center_ok_g", "VAR_center_ok_both",
        "SPEC_center_ok", "note",
        "r_reason", "g_reason"
    ]].copy()

    # Sort: put best (both centered) on top, then by smallest sep
    out["sep_r_arcsec"] = pd.to_numeric(out["sep_r_arcsec"], errors="coerce")
    out["sep_g_arcsec"] = pd.to_numeric(out["sep_g_arcsec"], errors="coerce")

    out["_rank"] = (out["VAR_center_ok_both"] != "Y").astype(int)
    out = out.sort_values(by=["_rank", "sep_r_arcsec", "sep_g_arcsec"], ascending=[True, True, True]).drop(columns=["_rank"])

    out.to_csv(OUTCSV, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("Center-check seed table generated ✅")
    print(f"Threshold: chosen_sep_arcsec <= {CENTER_TH_ARCSEC} arcsec")
    print(f"Saved to: {OUTCSV}")
    print("=" * 60)
    print("Next: manually fill SPEC_center_ok using Legacy Survey viewer + DESI spectra location.")
    print("Suggested labels for SPEC_center_ok: Y / N / unclear")

if __name__ == "__main__":
    main()
