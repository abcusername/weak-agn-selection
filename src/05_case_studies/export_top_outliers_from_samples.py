import os
import re
import glob
import numpy as np
import pandas as pd

# =========================
# Paths
# =========================
BASE = r"C:\Users\30126\Desktop\AGN"
SAMPLE_DIR = os.path.join(BASE, "TO_ADVISOR_CHECK_MINFIX",
                          "lightcurves_cleanR_MINFIX", "SEYFERT_examples")

OUTDIR = os.path.join(BASE, "TO_ADVISOR_CHECK_MINFIX",
                      "auto_classify_from_samples", "top_outliers_tables")
os.makedirs(OUTDIR, exist_ok=True)

TOPK = 30

# =========================
# Helpers
# =========================
def mag_to_flux(mag):
    return 10 ** (-0.4 * mag)

def magerr_to_fluxerr(mag, magerr):
    f = mag_to_flux(mag)
    return f * (np.log(10) * 0.4) * magerr

def parse_tid(fp):
    m = re.match(r"(\d+)_advisor\.csv$", os.path.basename(fp))
    return int(m.group(1)) if m else None

def compute_z(lc):
    # require mjd/mag/magerr
    t = lc["mjd"].to_numpy(dtype=float)
    mag = lc["mag"].to_numpy(dtype=float)
    merr = lc["magerr"].to_numpy(dtype=float)

    good = np.isfinite(t) & np.isfinite(mag) & np.isfinite(merr)
    lc = lc.loc[good].copy()

    mag = lc["mag"].to_numpy(dtype=float)
    merr = lc["magerr"].to_numpy(dtype=float)

    f = mag_to_flux(mag)
    fe = magerr_to_fluxerr(mag, merr)

    w = 1.0 / np.maximum(fe, 1e-12) ** 2
    fhat = np.sum(w * f) / np.sum(w)

    z = (f - fhat) / np.maximum(fe, 1e-12)
    lc["flux"] = f
    lc["fluxerr"] = fe
    lc["z_flux"] = z
    lc["abs_z"] = np.abs(z)
    return lc

# =========================
# Main
# =========================
csvs = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*_advisor.csv")))
if len(csvs) == 0:
    raise FileNotFoundError(f"No *_advisor.csv found in: {SAMPLE_DIR}")

summary_rows = []

for fp in csvs:
    tid = parse_tid(fp)
    if tid is None:
        print("Skip (bad name):", fp)
        continue

    lc = pd.read_csv(fp)
    need_cols = {"mjd", "mag", "magerr"}
    if not need_cols.issubset(lc.columns):
        print("Skip (missing cols):", tid, "cols=", lc.columns.tolist())
        continue

    lc2 = compute_z(lc)

    # TOPK by |z|
    top = lc2.sort_values("abs_z", ascending=False).head(TOPK).copy()

    out_fp = os.path.join(OUTDIR, f"{tid}_TOP{TOPK}_outliers.csv")
    top.to_csv(out_fp, index=False, encoding="utf-8-sig")

    summary_rows.append({
        "TARGETID": tid,
        "N": len(lc2),
        "top1_absz": float(top["abs_z"].iloc[0]),
        "top5_median_absz": float(np.median(top["abs_z"].head(5))),
        "frac_gt8": float(np.mean(lc2["abs_z"] > 8.0)),
        "outfile": out_fp
    })

    print("OK:", tid, "->", os.path.basename(out_fp))

summary = pd.DataFrame(summary_rows).sort_values("top1_absz", ascending=False)
summary_fp = os.path.join(OUTDIR, f"SUMMARY_TOP{TOPK}_outliers.csv")
summary.to_csv(summary_fp, index=False, encoding="utf-8-sig")

print("\nDone.")
print("OUTDIR:", OUTDIR)
print("Summary:", summary_fp)
