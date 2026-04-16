# 06_make_Figure4_Figure5_cleanR_MINFIX_paper.py
# ---------------------------------------------------
# Paper-grade Figure 4/5 maker for cleanR MINFIX outputs
# - Reads stats: AGN_variability_statistics_v5_cleanR_MINFIX.csv
# - Reads lightcurves: ZTF_lightcurves_CLEAN_R_MINFIX/<CLASS>/*_rband_clean.csv
# - Produces:
#   Figure4: LINER high-Fvar examples
#   Figure5: LINER vs Seyfert high-Fvar morphology
#
# Improvements vs old:
# - Optional brightness cap (max_mean_mag) to reduce faint-end systematics
# - Use errorbar (magerr) for paper-style visualization
# - Robust numeric cleaning

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Paths (MINFIX)
# -------------------------
BASE = r"C:\Users\30126\Desktop\AGN"
STATS = os.path.join(BASE, "AGN_variability_statistics_v5_cleanR_MINFIX.csv")

CLEAN_BASE = os.path.join(BASE, "ZTF_lightcurves_CLEAN_R_MINFIX")
CLEAN_DIRS = {
    "Seyfert":      os.path.join(CLEAN_BASE, "SEYFERT"),
    "Composite":    os.path.join(CLEAN_BASE, "COMPOSITE"),
    "LINER":        os.path.join(CLEAN_BASE, "LINER"),
    "Star-forming": os.path.join(CLEAN_BASE, "STARFORMING"),
}

FIGDIR = os.path.join(BASE, "FIGS_CLEAN_R_MINFIX")
os.makedirs(FIGDIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def norm_class(s: str) -> str:
    s = str(s).strip().lower()
    if s == "liner":
        return "LINER"
    if s == "seyfert":
        return "Seyfert"
    if s == "composite":
        return "Composite"
    if s in ["star-forming", "starforming", "star_forming", "sf"]:
        return "Star-forming"
    return s.title()

def load_clean_lc(class_name: str, targetid: str):
    """Load one clean lightcurve CSV, require mjd/mag/magerr."""
    p = os.path.join(CLEAN_DIRS[class_name], f"{targetid}_rband_clean.csv")
    if not os.path.exists(p):
        return None

    try:
        df = pd.read_csv(p)
    except Exception:
        return None

    for c in ["mjd", "mag", "magerr"]:
        if c not in df.columns:
            return None

    df["mjd"] = pd.to_numeric(df["mjd"], errors="coerce")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["magerr"] = pd.to_numeric(df["magerr"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mjd", "mag", "magerr"])

    if len(df) == 0:
        return None

    return df.sort_values("mjd")

def pick_top(df_stats: pd.DataFrame, class_name: str, *,
             k: int = 6, min_fvar: float = 10.0, min_n: int = 30,
             max_mean_mag: float | None = None):
    """
    Pick top-k highest Fvar targets in one class under QC.

    max_mean_mag: optional brightness cap (e.g. 18.5) to reduce faint-driven systematics.
                  None disables.
    """
    sub = df_stats[
        (df_stats["Class"] == class_name) &
        (df_stats["Fvar_percent"] >= min_fvar) &
        (df_stats["N"] >= min_n)
    ].copy()

    if max_mean_mag is not None:
        sub = sub[sub["Mean_mag"] <= max_mean_mag]

    sub = sub.sort_values("Fvar_percent", ascending=False).head(k)
    return sub

def plot_panel(ax, lc: pd.DataFrame, title: str):
    # Paper-friendly: show error bars
    ax.errorbar(
        lc["mjd"].values, lc["mag"].values,
        yerr=lc["magerr"].values,
        fmt="o", ms=3, lw=0.6, capsize=0,
        alpha=0.85
    )
    ax.set_xlabel("MJD")
    ax.set_ylabel("r-band mag")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2)

# -------------------------
# Figure 4
# -------------------------
def make_figure4(df_stats: pd.DataFrame, *,
                 k: int = 6, min_fvar: float = 10.0, min_n: int = 30,
                 max_mean_mag: float | None = None):
    top = pick_top(df_stats, "LINER", k=k, min_fvar=min_fvar, min_n=min_n, max_mean_mag=max_mean_mag)
    n = len(top)
    if n == 0:
        print("No LINER satisfies selection for Figure 4.")
        return

    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows), dpi=250)
    axes = np.array(axes).reshape(-1)

    for i, (_, r) in enumerate(top.iterrows()):
        tid = str(r["TARGETID"])
        lc = load_clean_lc("LINER", tid)
        if lc is None:
            axes[i].set_title(f"LINER {tid}\n(MISSING LC)")
            axes[i].axis("off")
            continue

        oid = lc["oid"].iloc[0] if "oid" in lc.columns else "NA"
        title = (
            f"LINER {tid}\n"
            f"oid={oid}, Fvar={r['Fvar_percent']:.1f}%, N={int(r['N'])}, "
            f"ΔT={r['Time_span']:.0f} d, MeanMag={r['Mean_mag']:.2f}"
        )
        plot_panel(axes[i], lc, title)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    crit = f"Selection: Fvar≥{min_fvar:.1f}%, N≥{min_n}"
    if max_mean_mag is not None:
        crit += f", Mean_mag≤{max_mean_mag:.2f}"

    fig.suptitle(f"Figure 4. High-Fvar LINER r-band lightcurves (cleanR MINFIX)\n{crit}",
                 fontsize=14, y=0.995)
    fig.tight_layout()

    out = os.path.join(FIGDIR, "Figure4_HighFvar_LINER_lightcurves_cleanR_MINFIX.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

# -------------------------
# Figure 5
# -------------------------
def make_figure5(df_stats: pd.DataFrame, *,
                 k: int = 3, min_fvar: float = 10.0, min_n: int = 30,
                 max_mean_mag: float | None = None):
    top_l = pick_top(df_stats, "LINER",   k=k, min_fvar=min_fvar, min_n=min_n, max_mean_mag=max_mean_mag)
    top_s = pick_top(df_stats, "Seyfert", k=k, min_fvar=min_fvar, min_n=min_n, max_mean_mag=max_mean_mag)

    n = min(len(top_l), len(top_s), k)
    if n == 0:
        print("No objects satisfy selection for Figure 5.")
        return

    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 7.2), dpi=250, sharey=False)

    for j in range(n):
        # LINER row
        rL = top_l.iloc[j]
        tidL = str(rL["TARGETID"])
        lcL = load_clean_lc("LINER", tidL)
        if lcL is None:
            axes[0, j].set_title(f"LINER {tidL}\n(MISSING)")
            axes[0, j].axis("off")
        else:
            oidL = lcL["oid"].iloc[0] if "oid" in lcL.columns else "NA"
            titleL = f"LINER {tidL}\noid={oidL}, Fvar={rL['Fvar_percent']:.1f}%, MeanMag={rL['Mean_mag']:.2f}"
            plot_panel(axes[0, j], lcL, titleL)

        # Seyfert row
        rS = top_s.iloc[j]
        tidS = str(rS["TARGETID"])
        lcS = load_clean_lc("Seyfert", tidS)
        if lcS is None:
            axes[1, j].set_title(f"Seyfert {tidS}\n(MISSING)")
            axes[1, j].axis("off")
        else:
            oidS = lcS["oid"].iloc[0] if "oid" in lcS.columns else "NA"
            titleS = f"Seyfert {tidS}\noid={oidS}, Fvar={rS['Fvar_percent']:.1f}%, MeanMag={rS['Mean_mag']:.2f}"
            plot_panel(axes[1, j], lcS, titleS)

    crit = f"Selection: Fvar≥{min_fvar:.1f}%, N≥{min_n}"
    if max_mean_mag is not None:
        crit += f", Mean_mag≤{max_mean_mag:.2f}"

    fig.suptitle(f"Figure 5. High-Fvar LINER vs Seyfert morphology (cleanR MINFIX)\n{crit}",
                 fontsize=14, y=0.995)
    fig.tight_layout()

    out = os.path.join(FIGDIR, "Figure5_LINER_vs_Seyfert_highFvar_lightcurves_cleanR_MINFIX.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

# -------------------------
# Main
# -------------------------
def main():
    df = pd.read_csv(STATS, dtype={"TARGETID": str})
    df["TARGETID"] = df["TARGETID"].astype(str)
    df["Class"] = df["Class"].apply(norm_class)

    for c in ["Fvar_percent", "N", "Mean_mag", "Time_span"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Class", "Fvar_percent", "N", "Mean_mag", "Time_span"])

    # ====== 论文建议：如果你要“更稳健”的示例，请打开亮度上限 ======
    # max_mean_mag = 18.5   # 推荐尝试 18.0 / 18.5 / None 对比
    max_mean_mag = None

    make_figure4(df, k=6, min_fvar=10.0, min_n=30, max_mean_mag=max_mean_mag)
    make_figure5(df, k=3, min_fvar=10.0, min_n=30, max_mean_mag=max_mean_mag)

if __name__ == "__main__":
    main()


