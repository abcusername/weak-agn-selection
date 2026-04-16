# 08_export_examples_for_advisor_with_residual.py
# ------------------------------------------------------------
# Export 20 bright Seyfert examples for advisor check:
# - Pick: brightest 10 + faintest 10 from stage1_seyfert_bright.csv
# - Save original LC CSV (all columns) as *_advisor.csv
# - Plot mag vs MJD with error bars as *_advisor.png
# - Plot normalized residuals in flux space as *_resid.png
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"C:\Users\30126\Desktop\AGN"
STAGE1 = os.path.join(BASE, "TO_ADVISOR_CHECK_MINFIX", "stage1_seyfert_bright.csv")
LC_DIR = os.path.join(BASE, "ZTF_lightcurves_CLEAN_R_MINFIX", "SEYFERT")

OUTDIR = os.path.join(BASE, "TO_ADVISOR_CHECK_MINFIX",
                      "lightcurves_cleanR_MINFIX", "SEYFERT_examples")
os.makedirs(OUTDIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def mag_to_flux(mag):
    return 10 ** (-0.4 * mag)

def magerr_to_fluxerr(mag, magerr):
    f = mag_to_flux(mag)
    return f * (np.log(10) * 0.4) * magerr

# -------------------------
# Load stage1 list
# -------------------------
df = pd.read_csv(STAGE1).sort_values("Mean_mag")

# Pick 20 examples: brightest 10 + faintest 10
pick = pd.concat([df.head(10), df.tail(10)], ignore_index=True)

miss = 0
for _, row in pick.iterrows():
    tid = int(row["TARGETID"])
    f = os.path.join(LC_DIR, f"{tid}_rband_clean.csv")
    if not os.path.exists(f):
        print("MISS:", tid, "expected", os.path.basename(f))
        miss += 1
        continue

    lc = pd.read_csv(f)

    # Require mjd/mag/magerr
    if not all(c in lc.columns for c in ["mjd", "mag", "magerr"]):
        print("BAD COLS:", tid, "missing one of mjd/mag/magerr")
        miss += 1
        continue

    # Use the columns directly
    MJD = lc["mjd"].values
    MAG = lc["mag"].values
    ERR = lc["magerr"].values

    # Export CSV (keep all columns for advisor)
    out_csv = os.path.join(OUTDIR, f"{tid}_advisor.csv")
    lc.to_csv(out_csv, index=False)

    # -------------------------
    # Plot 1: mag vs time (with error bars)
    # -------------------------
    plt.figure(figsize=(7, 3))
    plt.errorbar(MJD, MAG, yerr=ERR, fmt='.', markersize=2, linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("MJD")
    plt.ylabel("r-band mag")
    plt.title(f"SEYFERT {tid}  Mean={row['Mean_mag']:.2f}  "
              f"N={int(row['N'])}  Fvar={row['Fvar_percent']:.2f}%")
    out_png = os.path.join(OUTDIR, f"{tid}_advisor.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

    # -------------------------
    # Plot 2: normalized residuals in flux space
    # r = (f - fbar) / sigma_f
    # -------------------------
    m = np.isfinite(MJD) & np.isfinite(MAG) & np.isfinite(ERR) & (ERR > 0)
    mjd = MJD[m].astype(float)
    mag = MAG[m].astype(float)
    me  = ERR[m].astype(float)

    if len(mag) >= 5:
        fflux = mag_to_flux(mag)
        ferr  = magerr_to_fluxerr(mag, me)

        w = 1.0 / (ferr**2)
        fbar = np.sum(w * fflux) / np.sum(w)
        r = (fflux - fbar) / ferr

        plt.figure(figsize=(7, 3))
        plt.axhline(0, linewidth=1)
        plt.plot(mjd, r, '.', markersize=2)
        plt.xlabel("MJD")
        plt.ylabel("(f - f̄)/σ_f")
        plt.title(f"SEYFERT {tid} normalized residuals (flux)")
        out_png2 = os.path.join(OUTDIR, f"{tid}_resid.png")
        plt.tight_layout()
        plt.savefig(out_png2, dpi=220)
        plt.close()
    else:
        print("WARN:", tid, "too few good points for residual plot")

    print("OK:", tid)

print("Done. OUTDIR:", OUTDIR, "MISS:", miss)
