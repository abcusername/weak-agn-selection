# QC_05_multiband_booklet_and_corr.py
# ---------------------------------------------------
# Multi-band (g+r) correlation QC for Seyfert GOLD sample
# - Reads Seyfert GOLD list from your stage1 file (or stats if needed)
# - Loads g from: ZTF_lightcurves_CLEAN_GRi_MINFIX
# - Loads r from: ZTF_lightcurves_CLEAN_R_MINFIX  (your existing r-only clean)
# - Produces:
#   1) PDF booklet: each target one page with g/r lightcurves (same MJD axis)
#   2) CSV summary: N_g, N_r, N_matched, corr_gr, slope, etc.
#
# Notes:
# - Matching is done by nearest neighbor in time within MAX_DT days
# - Uses robust median-subtracted delta-mags for correlation
# - Does NOT require i-band

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------
# Paths (EDIT if needed)
# -------------------------
BASE = r"C:\Users\30126\Desktop\AGN"

# Input: your Seyfert GOLD list (Mean<18, N>500) already prepared
GOLD_CSV = os.path.join(BASE, r"TO_ADVISOR_CHECK_MINFIX\stage1_seyfert_bright.csv")
# If your file is actually under TO_ADVISOR_CHECK_MINFIX (as screenshot shows), keep.
# If it's under TO_ADVISOR_CHECK_MINFIX\stage1_seyfert_bright.csv (already), ok.

# Lightcurve roots
G_DIR_ROOT = os.path.join(BASE, "ZTF_lightcurves_CLEAN_GRi_MINFIX")   # g-band clean outputs
R_DIR_ROOT = os.path.join(BASE, "ZTF_lightcurves_CLEAN_R_MINFIX")     # r-band clean outputs

# Output
OUTDIR = os.path.join(BASE, r"TO_ADVISOR_CHECK_MINFIX\05_multiband_corr")
os.makedirs(OUTDIR, exist_ok=True)
OUTPDF = os.path.join(OUTDIR, "Seyfert_gold_gr_multiband_QC_booklet.pdf")
OUTCSV = os.path.join(OUTDIR, "Seyfert_gold_gr_multiband_corr_summary.csv")

# -------------------------
# Parameters
# -------------------------
MAX_TARGETS = 30          # enough for advisor; change to 50 if you want
MAX_DT_DAYS = 1.0         # time matching window (days)
MIN_MATCHED = 30          # minimum matched pairs to compute corr robustly

# Plot settings
plt.rcParams["axes.unicode_minus"] = False

# -------------------------
# Helpers
# -------------------------
def read_lc(path):
    df = pd.read_csv(path)
    # required
    for c in ["mjd", "mag", "magerr"]:
        if c not in df.columns:
            return None
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mjd", "mag", "magerr"])
    df = df.sort_values("mjd").reset_index(drop=True)
    return df

def match_by_time(g_df, r_df, max_dt=1.0):
    """
    Nearest-neighbor match g->r by time within max_dt (days).
    Returns matched arrays: mjd_g, mag_g, magerr_g, mjd_r, mag_r, magerr_r
    """
    tg = g_df["mjd"].values
    tr = r_df["mjd"].values
    if len(tg) == 0 or len(tr) == 0:
        return None

    # for each g time, find closest r time using searchsorted
    idx = np.searchsorted(tr, tg)
    idx0 = np.clip(idx - 1, 0, len(tr) - 1)
    idx1 = np.clip(idx,     0, len(tr) - 1)

    dt0 = np.abs(tr[idx0] - tg)
    dt1 = np.abs(tr[idx1] - tg)
    use1 = dt1 < dt0
    best = np.where(use1, idx1, idx0)
    dt = np.where(use1, dt1, dt0)

    keep = dt <= max_dt
    if keep.sum() == 0:
        return None

    gg = g_df.iloc[np.where(keep)[0]].copy()
    rr = r_df.iloc[best[keep]].copy()

    return (
        gg["mjd"].values, gg["mag"].values, gg["magerr"].values,
        rr["mjd"].values, rr["mag"].values, rr["magerr"].values
    )

def robust_corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 5:
        return np.nan
    # subtract median
    x0 = x - np.nanmedian(x)
    y0 = y - np.nanmedian(y)
    # Pearson on median-subtracted
    if np.nanstd(x0) == 0 or np.nanstd(y0) == 0:
        return np.nan
    return float(np.corrcoef(x0, y0)[0, 1])

def robust_slope(x, y):
    """
    Simple robust slope: slope of y vs x using median absolute approach.
    Here we use np.polyfit after median-subtraction as a practical proxy.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 5:
        return np.nan
    x0 = x - np.nanmedian(x)
    y0 = y - np.nanmedian(y)
    try:
        b = np.polyfit(x0, y0, 1)[0]
        return float(b)
    except Exception:
        return np.nan

def find_lc_file(root, class_name, tid, band):
    # band: 'g' or 'r'
    pattern = os.path.join(root, class_name.upper(), f"{tid}_{band}band_clean.csv")
    m = glob.glob(pattern)
    return m[0] if len(m) > 0 else None

# -------------------------
# Main
# -------------------------
def main():
    gold = pd.read_csv(GOLD_CSV)

    # normalize columns
    # Expected: TARGETID and Class columns exist
    if "TARGETID" not in gold.columns:
        raise RuntimeError(f"TARGETID not found in {GOLD_CSV}. Columns={list(gold.columns)}")
    if "Class" not in gold.columns:
        # if absent, assume Seyfert
        gold["Class"] = "Seyfert"

    # keep Seyfert only (gold sample should already be Seyfert)
    gold = gold[gold["Class"].astype(str).str.lower().str.contains("seyfert")].copy()

    # take top N (as-is order)
    gold = gold.head(MAX_TARGETS).copy()
    print(f"Loaded GOLD list: {len(gold)} targets from {GOLD_CSV}")

    rows = []
    with PdfPages(OUTPDF) as pdf:
        for k, row in gold.iterrows():
            tid = str(row["TARGETID"])
            cls = "SEYFERT"  # directory name
            g_path = find_lc_file(G_DIR_ROOT, cls, tid, "g")
            r_path = find_lc_file(R_DIR_ROOT, cls, tid, "r")

            status = "OK"
            reason = ""
            if g_path is None:
                status = "NO_G"
                reason = "missing gband_clean.csv"
            if r_path is None:
                status = "NO_R"
                reason = "missing rband_clean.csv"

            corr = np.nan
            slope = np.nan
            n_g = n_r = n_m = 0

            fig = plt.figure(figsize=(10, 7))
            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)

            if status == "OK":
                g_df = read_lc(g_path)
                r_df = read_lc(r_path)
                if g_df is None or r_df is None:
                    status = "BAD_FILE"
                    reason = "cannot read mjd/mag/magerr"
                else:
                    n_g = len(g_df)
                    n_r = len(r_df)

                    # plot raw (not matched)
                    ax1.errorbar(g_df["mjd"], g_df["mag"], yerr=g_df["magerr"],
                                 fmt=".", ms=2, alpha=0.7)
                    ax1.set_ylabel("g-band mag")
                    ax1.invert_yaxis()
                    ax1.grid(True, alpha=0.2)

                    ax2.errorbar(r_df["mjd"], r_df["mag"], yerr=r_df["magerr"],
                                 fmt=".", ms=2, alpha=0.7)
                    ax2.set_ylabel("r-band mag")
                    ax2.set_xlabel("MJD")
                    ax2.invert_yaxis()
                    ax2.grid(True, alpha=0.2)

                    m = match_by_time(g_df, r_df, max_dt=MAX_DT_DAYS)
                    if m is None:
                        status = "NO_MATCH"
                        reason = f"no pairs within {MAX_DT_DAYS} d"
                    else:
                        mjdg, mg, eg, mjdr, mr, er = m
                        n_m = len(mg)
                        if n_m >= MIN_MATCHED:
                            corr = robust_corr(mg, mr)
                            slope = robust_slope(mg, mr)
                        else:
                            reason = f"matched<{MIN_MATCHED} (n={n_m})"

            title = f"Seyfert GOLD {tid} | status={status}"
            if status == "OK":
                title += f" | N_g={n_g} N_r={n_r} matched={n_m} corr(gr)={corr:.2f} slope={slope:.2f}"
            else:
                title += f" | {reason}"
            fig.suptitle(title, fontsize=12)

            # footer
            ax2.text(0.01, 0.02,
                     f"g: {g_path or 'None'}\n"
                     f"r: {r_path or 'None'}\n"
                     f"match window: {MAX_DT_DAYS} d",
                     transform=ax2.transAxes, fontsize=8, va="bottom")

            pdf.savefig(fig)
            plt.close(fig)

            rows.append(dict(
                TARGETID=tid,
                status=status,
                reason=reason,
                g_file=g_path or "",
                r_file=r_path or "",
                N_g=n_g,
                N_r=n_r,
                N_matched=n_m,
                corr_gr=corr,
                slope_gr=slope,
                match_dt_days=MAX_DT_DAYS
            ))

    pd.DataFrame(rows).to_csv(OUTCSV, index=False)
    print("\nDone.")
    print(f"Saved PDF: {OUTPDF}")
    print(f"Saved CSV: {OUTCSV}")
    print(f"Output dir: {OUTDIR}")

if __name__ == "__main__":
    main()
