import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"C:\Users\30126\Desktop\AGN"
STATS_IN = os.path.join(BASE, "AGN_variability_statistics_v6_cleanR_MINFIX_withPvar.csv")
LC_BASE  = os.path.join(BASE, "ZTF_lightcurves_CLEAN_R_MINFIX")

OUT_STATS = os.path.join(BASE, "AGN_variability_statistics_v7_cleanR_MINFIX_withPvar_FvarSys.csv")
OUT_FIGDIR = os.path.join(BASE, "TO_ADVISOR_CHECK_MINFIX", "figs")
os.makedirs(OUT_FIGDIR, exist_ok=True)

# ---------- helpers ----------
def mag_to_flux(mag):
    return 10 ** (-0.4 * mag)

def magerr_to_fluxerr(mag, magerr):
    f = mag_to_flux(mag)
    return f * (np.log(10) * 0.4) * magerr

def class_to_dir(cls):
    s = str(cls).strip().upper()
    s = s.replace("-", "").replace("_", "").replace(" ", "")
    if s.startswith("STAR"):
        return "STARFORMING"
    if s.startswith("SEYF"):
        return "SEYFERT"
    if s.startswith("LINER"):
        return "LINER"
    if s.startswith("COMP"):
        return "COMPOSITE"
    return None

def lc_path(cls_dir, targetid):
    return os.path.join(LC_BASE, cls_dir, f"{int(targetid)}_rband_clean.csv")

def compute_fvar_flux(mag, magerr, sigma_sys_mag=0.0):
    """
    Fvar in flux space:
    Fvar = sqrt( S^2 - mean(sigma_f^2) ) / mean(f)
    where sigma_f includes random magerr + sys floor in mag.
    """
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    m = np.isfinite(mag) & np.isfinite(magerr) & (magerr > 0)
    mag = mag[m]; magerr = magerr[m]
    N = len(mag)
    if N < 5:
        return np.nan, N, np.nan, np.nan

    # combine random + sys floor in mag space
    magerr_eff = np.sqrt(magerr**2 + sigma_sys_mag**2)

    f = mag_to_flux(mag)
    fe = magerr_to_fluxerr(mag, magerr_eff)

    fbar = np.mean(f)
    if not np.isfinite(fbar) or fbar <= 0:
        return np.nan, N, np.nan, np.nan

    S2 = np.var(f, ddof=1)           # sample variance
    noise = np.mean(fe**2)           # mean squared flux error
    excess = S2 - noise

    if excess <= 0:
        return 0.0, N, S2, noise     # treat as non-variable under this noise model

    Fvar = np.sqrt(excess) / fbar
    return Fvar, N, S2, noise

# ---------- main ----------
df = pd.read_csv(STATS_IN)
if "TARGETID" not in df.columns or "Class" not in df.columns:
    raise RuntimeError("stats missing TARGETID/Class")

# sys floors to scan (mag)
sys_list = [0.00, 0.01, 0.02, 0.03, 0.05]

# containers: for each sys, store per-object Fvar
for s in sys_list:
    df[f"Fvar_sys{int(s*1000):03d}"] = np.nan

# compute
for i, row in df.iterrows():
    tid = int(row["TARGETID"])
    cls_dir = class_to_dir(row["Class"])
    if cls_dir is None:
        continue

    f = lc_path(cls_dir, tid)
    if not os.path.exists(f):
        continue

    lc = pd.read_csv(f)
    if not all(c in lc.columns for c in ["mag", "magerr"]):
        continue

    mag = lc["mag"].values
    magerr = lc["magerr"].values

    for s in sys_list:
        Fvar, Nuse, S2, noise = compute_fvar_flux(mag, magerr, sigma_sys_mag=s)
        df.at[i, f"Fvar_sys{int(s*1000):03d}"] = 100.0 * Fvar  # percent

df.to_csv(OUT_STATS, index=False)
print("Saved:", OUT_STATS)

# ---------- plots ----------
if "Mean_mag" in df.columns:
    # A: original (sys=0)
    plt.figure(figsize=(7,5))
    for cls in ["SEYFERT","LINER","COMPOSITE","STARFORMING","STAR-FORMING","Star-forming"]:
        d = df[df["Class"].astype(str).str.upper() == cls]
        if len(d)==0: 
            continue
        plt.scatter(d["Mean_mag"], d["Fvar_sys000"], s=6, alpha=0.4, label=cls)
    plt.xlabel("Mean r-band mag")
    plt.ylabel("Fvar (%) [sys=0]")
    plt.title("Fvar vs Mean mag (no sys floor)")
    plt.legend(markerscale=3, frameon=True)
    out = os.path.join(OUT_FIGDIR, "Fig_Fvar_vs_MeanMag_sys000.png")
    plt.tight_layout(); plt.savefig(out, dpi=250); plt.close()
    print("Saved:", out)

    # B: with sys floors
    for s in sys_list[1:]:
        col = f"Fvar_sys{int(s*1000):03d}"
        plt.figure(figsize=(7,5))
        for cls in ["SEYFERT","LINER","COMPOSITE","STARFORMING","STAR-FORMING","Star-forming"]:
            d = df[df["Class"].astype(str).str.upper() == cls]
            if len(d)==0:
                continue
            plt.scatter(d["Mean_mag"], d[col], s=6, alpha=0.4, label=cls)
        plt.xlabel("Mean r-band mag")
        plt.ylabel(f"Fvar (%) [sys={s:.2f} mag]")
        plt.title(f"Fvar vs Mean mag (sys floor = {s:.2f} mag)")
        plt.legend(markerscale=3, frameon=True)
        out = os.path.join(OUT_FIGDIR, f"Fig_Fvar_vs_MeanMag_sys{int(s*1000):03d}.png")
        plt.tight_layout(); plt.savefig(out, dpi=250); plt.close()
        print("Saved:", out)

    # variable fraction vs mag for different sys floors
    bins = np.arange(14, 22.5, 0.5)
    centers = 0.5*(bins[:-1]+bins[1:])
    thr = 5.0  # start with your previous cut; we can tune later

    plt.figure(figsize=(7,5))
    # focus on STARFORMING and SEYFERT as the key contrast
    def mask_starforming(s):
        u = s.astype(str).str.upper()
        return (u=="STARFORMING") | (u=="STAR-FORMING") | (u=="STAR FORMING") | (u=="STAR-FORMING")

    groups = {
        "SEYFERT": df["Class"].astype(str).str.upper()=="SEYFERT",
        "STARFORMING": mask_starforming(df["Class"]),
    }

    for gname, gmask in groups.items():
        d0 = df[gmask].copy()
        for s in [0.00, 0.02, 0.05]:
            col = f"Fvar_sys{int(s*1000):03d}"
            fracs=[]
            for b0,b1 in zip(bins[:-1],bins[1:]):
                x = d0[(d0["Mean_mag"]>=b0)&(d0["Mean_mag"]<b1)][col].dropna()
                if len(x)<20:
                    fracs.append(np.nan)
                else:
                    fracs.append(np.mean(x>thr))
            plt.plot(centers, fracs, marker='o', linewidth=1,
                     label=f"{gname}  sys={s:.2f}")

    plt.xlabel("Mean r-band mag (bin centers)")
    plt.ylabel(f"Variable fraction (Fvar > {thr:.1f}%)")
    plt.ylim(-0.05, 1.05)
    plt.title("Variable fraction vs magnitude (Fvar threshold)")
    plt.legend(frameon=True, ncol=2)
    out = os.path.join(OUT_FIGDIR, "Fig_variable_fraction_vs_mag_FvarSys.png")
    plt.tight_layout(); plt.savefig(out, dpi=250); plt.close()
    print("Saved:", out)

print("Done.")
