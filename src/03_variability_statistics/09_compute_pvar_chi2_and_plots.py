# 09_compute_pvar_chi2_and_plots_sysfloor_FIXED.py
# ------------------------------------------------------------
# Compute Pvar from chi-square test (constant-flux model) with
# optional systematic error floor (in mag), plus diagnostic columns,
# and produce plots (Pvar vs MeanMag, hist, variable fraction vs mag).
#
# FIXED for your v5 table schema and real Class spellings:
# - v5 columns: TARGETID, Class, N, Time_span, Mean_mag, Fvar_percent
# - Class in file: Seyfert / LINER / Composite / Star-forming (mixed case)
# We normalize Class once into Class_norm: SEYFERT/LINER/COMPOSITE/STARFORMING
# and use it everywhere (folders + plots + masks).
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

# =========================
# Paths
# =========================
BASE = r"C:\Users\30126\Desktop\AGN"
STATS_IN = os.path.join(BASE, "AGN_variability_statistics_v5_cleanR_MINFIX.csv")
LC_BASE  = os.path.join(BASE, "ZTF_lightcurves_CLEAN_R_MINFIX")

OUT_STATS  = os.path.join(BASE, "AGN_variability_statistics_v7_cleanR_MINFIX_withPvar_FvarSys.csv")
OUT_FIGDIR = os.path.join(BASE, "TO_ADVISOR_CHECK_MINFIX", "figs")
os.makedirs(OUT_FIGDIR, exist_ok=True)

# =========================
# User settings
# =========================
SYS_FLOORS_MAG = [0.00, 0.01, 0.02, 0.03, 0.05]   # mag systematic floors to test
PVAR_THR = 0.99                                   # for variable-fraction plot (Pvar > thr)
BINS = np.arange(14, 22.5, 0.5)                   # magnitude bins for fraction plot
MIN_PER_BIN = 20                                  # min objects per bin to compute fraction

# Outlier handling
DO_SIGMA_CLIP = True
CLIP_K = 5.0
CLIP_MAX_ITER = 3

# Choose one sys floor for hist/fraction/diagnostic plots
HIST_SYS = 0.02

# =========================
# Utils
# =========================
def mag_to_flux(mag):
    return 10 ** (-0.4 * mag)

def magerr_to_fluxerr(mag, magerr):
    f = mag_to_flux(mag)
    return f * (np.log(10) * 0.4) * magerr

def normalize_class_label(x):
    """
    Normalize your mixed spellings into canonical labels:
    SEYFERT / LINER / COMPOSITE / STARFORMING
    """
    s = str(x).strip().upper()
    # remove separators
    s2 = s.replace("_", "").replace("-", "").replace(" ", "")
    if s2 in ["SEYFERT", "SEYFERTS"]:
        return "SEYFERT"
    if s2 in ["LINER", "LINERS"]:
        return "LINER"
    if s2 in ["COMPOSITE", "COMP"]:
        return "COMPOSITE"
    if s2 in ["STARFORMING", "STARFORMATION", "STARFORMINGGALAXY", "STARFORMINGGALAXIES"]:
        return "STARFORMING"
    # any string containing both STAR and FORM -> starforming
    if ("STAR" in s) and ("FORM" in s):
        return "STARFORMING"
    return "UNKNOWN"

def class_to_dir(cls_norm):
    """
    Folder names under LC_BASE are:
    SEYFERT / LINER / COMPOSITE / STARFORMING
    """
    if cls_norm in ["SEYFERT", "LINER", "COMPOSITE", "STARFORMING"]:
        return cls_norm
    return None

def lc_path(cls_dir, targetid):
    return os.path.join(LC_BASE, cls_dir, f"{int(targetid)}_rband_clean.csv")

def sigma_clip_flux(f, fe, k=5.0, max_iter=3):
    """
    Iterative sigma clipping in flux space based on normalized residuals:
    clip |(f - fbar)/fe| > k
    """
    m = np.isfinite(f) & np.isfinite(fe) & (fe > 0)
    f = f[m]; fe = fe[m]
    if len(f) < 5:
        return f, fe

    for _ in range(max_iter):
        w = 1.0 / (fe**2)
        fbar = np.sum(w * f) / np.sum(w)
        r = (f - fbar) / fe
        keep = np.isfinite(r) & (np.abs(r) <= k)
        if keep.sum() == len(f):
            break
        f, fe = f[keep], fe[keep]
        if len(f) < 5:
            break
    return f, fe

def compute_pvar_from_mag(mag, magerr, sys_floor_mag=0.02, do_clip=True):
    """
    Compute chi2 against constant-flux model using inverse-variance weights.
    Add systematic floor in mag: magerr_eff = sqrt(magerr^2 + sys^2).

    Return:
      Pvar=1-p, p, chi2, dof, N_used, chi2_red, minuslog10p
    """
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    magerr_eff = np.sqrt(magerr**2 + sys_floor_mag**2)

    m = np.isfinite(mag) & np.isfinite(magerr_eff) & (magerr_eff > 0)
    mag = mag[m]; magerr_eff = magerr_eff[m]
    if len(mag) < 5:
        return (np.nan, np.nan, np.nan, np.nan, len(mag), np.nan, np.nan)

    f  = mag_to_flux(mag)
    fe = magerr_to_fluxerr(mag, magerr_eff)

    if do_clip:
        f, fe = sigma_clip_flux(f, fe, k=CLIP_K, max_iter=CLIP_MAX_ITER)

    Nuse = len(f)
    if Nuse < 5:
        return (np.nan, np.nan, np.nan, np.nan, Nuse, np.nan, np.nan)

    w = 1.0 / (fe**2)
    fbar = np.sum(w * f) / np.sum(w)

    chi2_val = np.sum(((f - fbar)**2) * w)
    dof = Nuse - 1

    p = chi2.sf(chi2_val, dof)
    Pvar = 1.0 - p

    # safety
    if np.isfinite(Pvar):
        Pvar = float(np.clip(Pvar, 0.0, 1.0))
    else:
        Pvar = np.nan

    chi2_red = chi2_val / dof if dof > 0 else np.nan
    minuslog10p = -np.log10(p) if (p is not None and p > 0) else np.inf

    return (Pvar, p, chi2_val, dof, Nuse, chi2_red, minuslog10p)

# =========================
# Main
# =========================
df = pd.read_csv(STATS_IN)

# Check required columns in your v5 file
required = ["TARGETID", "Class"]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in STATS_IN: {missing_cols}")

# Normalize class once
df["Class_norm"] = df["Class"].apply(normalize_class_label)

# Prepare output columns for each sys floor
for sys in SYS_FLOORS_MAG:
    tag = f"{int(round(sys*1000)):03d}"  # 0.02 -> "020"
    df[f"Pvar_sys{tag}"] = np.nan
    df[f"p_chi2_sys{tag}"] = np.nan
    df[f"chi2_sys{tag}"] = np.nan
    df[f"dof_sys{tag}"] = np.nan
    df[f"N_used_pvar_sys{tag}"] = 0
    df[f"chi2red_sys{tag}"] = np.nan
    df[f"mlog10p_sys{tag}"] = np.nan

miss = 0
miss_by_class = {}

for idx, row in df.iterrows():
    tid = int(row["TARGETID"])
    cls_norm = row["Class_norm"]
    cls_dir = class_to_dir(cls_norm)

    if cls_dir is None:
        miss += 1
        miss_by_class[cls_norm] = miss_by_class.get(cls_norm, 0) + 1
        continue

    fpath = lc_path(cls_dir, tid)
    if not os.path.exists(fpath):
        miss += 1
        miss_by_class[cls_dir] = miss_by_class.get(cls_dir, 0) + 1
        continue

    lc = pd.read_csv(fpath)
    if not all(c in lc.columns for c in ["mjd", "mag", "magerr"]):
        miss += 1
        miss_by_class[cls_dir] = miss_by_class.get(cls_dir, 0) + 1
        continue

    mag = lc["mag"].values
    magerr = lc["magerr"].values

    for sys in SYS_FLOORS_MAG:
        tag = f"{int(round(sys*1000)):03d}"
        Pvar, pval, chi2v, dof, Nuse, chi2red, mlog10p = compute_pvar_from_mag(
            mag, magerr, sys_floor_mag=sys, do_clip=DO_SIGMA_CLIP
        )
        df.at[idx, f"Pvar_sys{tag}"] = Pvar
        df.at[idx, f"p_chi2_sys{tag}"] = pval
        df.at[idx, f"chi2_sys{tag}"] = chi2v
        df.at[idx, f"dof_sys{tag}"] = dof
        df.at[idx, f"N_used_pvar_sys{tag}"] = int(Nuse)
        df.at[idx, f"chi2red_sys{tag}"] = chi2red
        df.at[idx, f"mlog10p_sys{tag}"] = mlog10p

# Save table
df.to_csv(OUT_STATS, index=False)
print("Saved:", OUT_STATS)
print("Missing LC files:", miss)
print("Missing by class:", miss_by_class)

# =========================
# Plots
# =========================
if "Mean_mag" not in df.columns:
    print("[WARN] Mean_mag not found in stats table. Skip magnitude-based plots.")
else:
    # 1) Pvar vs MeanMag for each sys floor
    for sys in SYS_FLOORS_MAG:
        tag = f"{int(round(sys*1000)):03d}"
        ycol = f"Pvar_sys{tag}"

        plt.figure(figsize=(7, 5))
        for cls in ["SEYFERT", "LINER", "COMPOSITE", "STARFORMING"]:
            d = df[df["Class_norm"] == cls]
            if len(d) == 0:
                continue
            plt.scatter(d["Mean_mag"], d[ycol], s=6, alpha=0.4, label=cls)

        plt.xlabel("Mean r-band mag")
        plt.ylabel(f"Pvar = 1 - p(chi2)   [sys={sys:.2f} mag]")
        plt.ylim(-0.05, 1.05)
        plt.title(f"Pvar vs Mean magnitude (sys floor = {sys:.2f} mag)")
        plt.legend(markerscale=3, frameon=True)
        out1 = os.path.join(OUT_FIGDIR, f"Fig_Pvar_vs_MeanMag_sys{tag}.png")
        plt.tight_layout()
        plt.savefig(out1, dpi=250)
        plt.close()
        print("Saved:", out1)

    # 2) Hist: Seyfert vs Star-forming at HIST_SYS
    tagH = f"{int(round(HIST_SYS*1000)):03d}"
    ycolH = f"Pvar_sys{tagH}"

    plt.figure(figsize=(7, 4))
    d1 = df[df["Class_norm"] == "SEYFERT"][ycolH].dropna()
    d2 = df[df["Class_norm"] == "STARFORMING"][ycolH].dropna()

    if len(d1) > 0:
        plt.hist(d1, bins=40, alpha=0.5, density=True, label=f"SEYFERT (N={len(d1)})")
    else:
        print("[WARN] No SEYFERT data for histogram.")
    if len(d2) > 0:
        plt.hist(d2, bins=40, alpha=0.5, density=True, label=f"STARFORMING (N={len(d2)})")
    else:
        print("[WARN] No STARFORMING data for histogram.")

    plt.xlabel("Pvar")
    plt.ylabel("Density")
    plt.title(f"Pvar distribution (sys floor = {HIST_SYS:.2f} mag)")
    plt.legend(frameon=True)
    out2 = os.path.join(OUT_FIGDIR, f"Fig_Pvar_hist_Seyfert_vs_Starforming_sys{tagH}.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=250)
    plt.close()
    print("Saved:", out2)

    # 3) Variable fraction vs magnitude at HIST_SYS
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    plt.figure(figsize=(7, 5))

    for cls in ["SEYFERT", "LINER", "COMPOSITE", "STARFORMING"]:
        d = df[df["Class_norm"] == cls].copy()
        if len(d) == 0:
            continue

        fracs = []
        for b0, b1 in zip(BINS[:-1], BINS[1:]):
            x = d[(d["Mean_mag"] >= b0) & (d["Mean_mag"] < b1)].dropna(subset=[ycolH])
            if len(x) < MIN_PER_BIN:
                fracs.append(np.nan)
            else:
                fracs.append(np.mean(x[ycolH] > PVAR_THR))

        plt.plot(centers, fracs, marker="o", linewidth=1, label=cls)

    plt.xlabel("Mean r-band mag (bin centers)")
    plt.ylabel(f"Variable fraction (Pvar > {PVAR_THR})")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Variable fraction vs magnitude (sys floor = {HIST_SYS:.2f} mag)")
    plt.legend(frameon=True)
    out3 = os.path.join(OUT_FIGDIR, f"Fig_variable_fraction_vs_mag_Pvar_sys{tagH}.png")
    plt.tight_layout()
    plt.savefig(out3, dpi=250)
    plt.close()
    print("Saved:", out3)

    # 4) Diagnostic: reduced chi2 vs MeanMag (HIST_SYS)
    chi2col = f"chi2red_sys{tagH}"
    plt.figure(figsize=(7, 5))
    for cls in ["SEYFERT", "LINER", "COMPOSITE", "STARFORMING"]:
        d = df[df["Class_norm"] == cls]
        if len(d) == 0:
            continue
        plt.scatter(d["Mean_mag"], d[chi2col], s=6, alpha=0.4, label=cls)

    plt.xlabel("Mean r-band mag")
    plt.ylabel(f"Reduced chi2   [sys={HIST_SYS:.2f} mag]")
    plt.yscale("log")
    plt.title("Reduced chi2 vs Mean magnitude (diagnostic)")
    plt.legend(markerscale=3, frameon=True)
    out4 = os.path.join(OUT_FIGDIR, f"Fig_chi2red_vs_MeanMag_sys{tagH}.png")
    plt.tight_layout()
    plt.savefig(out4, dpi=250)
    plt.close()
    print("Saved:", out4)

print("Done.")
