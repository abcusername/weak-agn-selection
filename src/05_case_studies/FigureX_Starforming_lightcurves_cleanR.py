# FigureX_Starforming_lightcurves_cleanR.py
# Purpose:
#   Figure X. Star-forming galaxy r-band lightcurves (cleanR)
#   Show morphology-level comparison: SF should look noise-dominated under the same cleaning.
#
# Input:
#   1) AGN_variability_statistics_v5_cleanR.csv  (contains Class, TARGETID, Fvar_percent, N, Time_span, Mean_mag)
#   2) Clean lightcurves: ZTF_lightcurves_CLEAN_R/STARFORMING/*_rband_clean.csv
#
# Output:
#   FIGS_CLEAN_R/FigureX_Starforming_lightcurves_cleanR.png

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# 忽略特定警告
warnings.filterwarnings('ignore', category=FutureWarning)

# ======================
# USER SETTINGS
# ======================
BASE = r"C:\Users\30126\Desktop\AGN"

STATS_CSV = os.path.join(BASE, "AGN_variability_statistics_v5_cleanR_MINFIX.csv")
CLEAN_ROOT = os.path.join(BASE,"ZTF_lightcurves_CLEAN_R_MINFIX")
OUTDIR = os.path.join(BASE, "FIGS_CLEAN_R_MINFIX")
os.makedirs(OUTDIR, exist_ok=True)

SF_DIR = os.path.join(CLEAN_ROOT, "STARFORMING")

# Figure layout
N_SHOW = 6                 # how many SF examples to show
NROWS, NCOLS = 2, 3
FIGSIZE = (14, 8)

# What is "noise-dominated" here?
# -> Prefer low Fvar (e.g., <= 5%), but keep decent sampling.
FVAR_MAX = 5.0             # pick examples with Fvar <= this
MIN_N = 50                 # avoid too few points
MIN_DT = 500               # days; avoid super short span that looks weird
MAX_DT = 3500              # sanity cap

# Plot style
USE_ERRORBAR = False       # if True: errorbar; if False: scatter (faster & cleaner)
POINT_SIZE = 10
ALPHA = 0.75

# ======================
# Helpers
# ======================
def safe_read_lc(fp):
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        print(f"Warning: Failed to read {fp} - {e}")
        return None
    need = {"mjd", "mag", "magerr"}
    if not need.issubset(df.columns):
        print(f"Warning: Missing required columns in {fp}")
        return None
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mjd", "mag", "magerr"])
    if len(df) < 5:
        print(f"Warning: Too few points in {fp} (n={len(df)})")
        return None
    return df

def load_stats():
    try:
        df = pd.read_csv(STATS_CSV, dtype={"TARGETID": str})
    except Exception as e:
        raise FileNotFoundError(f"Cannot read stats CSV: {e}")
    
    # normalize Class naming
    def norm_class(s):
        s = str(s).strip().lower()
        if s in ["liner", "liners"]: return "LINER"
        if s in ["seyfert", "seyferts"]: return "Seyfert"
        if s in ["composite", "composites"]: return "Composite"
        if s in ["star-forming", "starforming", "star forming", "sf", "star forming galaxy"]: 
            return "Star-forming"
        return s.title()
    
    df["Class"] = df["Class"].apply(norm_class)

    for c in ["Fvar_percent", "N", "Time_span", "Mean_mag"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Check data availability
    initial_count = len(df)
    df = df.dropna(subset=["TARGETID", "Class", "Fvar_percent", "N", "Time_span", "Mean_mag"])
    final_count = len(df)
    
    print(f"Loaded stats: {initial_count} rows, after cleaning: {final_count} rows")
    print("Class distribution:")
    print(df["Class"].value_counts())
    
    return df

def pick_sf_examples(stats_sf, n_show=N_SHOW):
    """
    Prefer:
      - low Fvar (<= FVAR_MAX)
      - N >= MIN_N
      - Time_span within [MIN_DT, MAX_DT]
    If not enough, relax gradually.
    """
    s = stats_sf.copy()
    print(f"Available Star-forming targets: {len(s)}")
    
    if len(s) == 0:
        print("Error: No Star-forming targets available")
        return pd.DataFrame()

    # primary filter
    cand = s[(s["Fvar_percent"] <= FVAR_MAX) &
             (s["N"] >= MIN_N) &
             (s["Time_span"] >= MIN_DT) &
             (s["Time_span"] <= MAX_DT)].copy()
    
    print(f"After primary filter (Fvar<={FVAR_MAX}, N>={MIN_N}, {MIN_DT}<=dt<={MAX_DT}): {len(cand)} candidates")

    if len(cand) < n_show:
        print(f"Not enough candidates ({len(cand)} < {n_show}), relaxing criteria...")
        # relax Fvar a bit
        cand = s[(s["Fvar_percent"] <= max(FVAR_MAX, 8.0)) &
                 (s["N"] >= max(20, MIN_N//2)) &
                 (s["Time_span"] >= 200)].copy()
        print(f"After relaxed filter: {len(cand)} candidates")

    if len(cand) == 0:
        print("No candidates after filtering, using all available targets")
        # last resort: random from all SF
        cand = s.copy()
        print(f"Using all {len(cand)} available targets")

    # pick "most noise-like": lowest Fvar first; then higher N is better
    cand = cand.sort_values(["Fvar_percent", "N"], ascending=[True, False]).head(max(n_show*3, n_show))

    print(f"Selected {len(cand)} candidates for stratification")

    # take diverse Mean_mag to avoid all identical-looking panels
    # (simple stratified by quantiles)
    if len(cand) <= n_show:
        print(f"Only {len(cand)} candidates, using all")
        return cand.head(n_show)

    # stratify by Mean_mag quantiles
    cand = cand.copy()
    # Use observed=True to fix the FutureWarning
    cand["q"] = pd.qcut(cand["Mean_mag"], q=min(6, len(cand)), duplicates="drop", labels=False)
    
    picked = []
    # Fix the groupby warning by adding observed=True
    for _, grp in cand.groupby("q", observed=True):
        if len(grp) > 0:
            picked.append(grp.head(1))
        if len(picked) >= n_show:
            break
    
    if not picked:
        print("Warning: No groups formed during stratification, using first n_show candidates")
        out = cand.head(n_show)
    else:
        out = pd.concat(picked).head(n_show)
    
    print(f"Selected {len(out)} examples for plotting")
    return out

def find_clean_file_sf(targetid):
    fp = os.path.join(SF_DIR, f"{targetid}_rband_clean.csv")
    if os.path.exists(fp):
        return fp
    else:
        # Try alternative naming patterns
        alt_patterns = [
            os.path.join(SF_DIR, f"{targetid}_clean.csv"),
            os.path.join(SF_DIR, f"{targetid}.csv"),
            os.path.join(SF_DIR, f"TARGET_{targetid}_rband_clean.csv"),
        ]
        for alt_fp in alt_patterns:
            if os.path.exists(alt_fp):
                print(f"Found alternative file: {alt_fp}")
                return alt_fp
        print(f"Warning: No clean file found for target {targetid}")
        return None

# ======================
# Main
# ======================
def main():
    print("=" * 70)
    print("Star-forming Galaxy Lightcurves Plotter")
    print("=" * 70)
    
    if not os.path.exists(STATS_CSV):
        raise FileNotFoundError(f"Stats CSV not found: {STATS_CSV}")

    if not os.path.isdir(SF_DIR):
        raise FileNotFoundError(f"STARFORMING cleanR folder not found: {SF_DIR}")
    
    print(f"Stats CSV: {STATS_CSV}")
    print(f"Star-forming directory: {SF_DIR}")
    
    # Check if there are any clean files
    clean_files = glob.glob(os.path.join(SF_DIR, "*_rband_clean.csv"))
    print(f"Found {len(clean_files)} clean lightcurve files")
    
    if len(clean_files) == 0:
        print("Warning: No clean lightcurve files found!")
        # Try alternative pattern
        clean_files = glob.glob(os.path.join(SF_DIR, "*.csv"))
        print(f"Found {len(clean_files)} CSV files (any pattern)")
    
    stats = load_stats()
    stats_sf = stats[stats["Class"] == "Star-forming"].copy()
    
    if len(stats_sf) == 0:
        print("Available classes in stats:", stats["Class"].unique())
        raise RuntimeError("No Star-forming rows found in stats CSV. Check Class naming.")

    print(f"\nStar-forming statistics: {len(stats_sf)} targets")
    
    picked = pick_sf_examples(stats_sf, n_show=N_SHOW)
    
    if len(picked) == 0:
        raise RuntimeError("Could not select any Star-forming examples to plot.")
    
    print(f"\nSelected {len(picked)} Star-forming examples:")
    for idx, row in picked.iterrows():
        print(f"  {row['TARGETID']}: Fvar={row['Fvar_percent']:.1f}%, N={row['N']}, dt={row['Time_span']:.0f}d")

    # prepare figure
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    shown = 0
    used = []
    failed = []

    for _, r in picked.iterrows():
        tid = str(r["TARGETID"])
        fp = find_clean_file_sf(tid)
        
        if fp is None:
            print(f"Warning: No clean file for target {tid}")
            failed.append(tid)
            continue

        lc = safe_read_lc(fp)
        if lc is None:
            print(f"Warning: Could not read clean file for target {tid}")
            failed.append(tid)
            continue

        ax = axes[shown]
        mjd = lc["mjd"].to_numpy()
        mag = lc["mag"].to_numpy()
        magerr = lc["magerr"].to_numpy()

        if USE_ERRORBAR:
            ax.errorbar(mjd, mag, yerr=magerr, fmt="o", ms=2.5, capsize=1.5, alpha=ALPHA)
        else:
            ax.scatter(mjd, mag, s=POINT_SIZE, alpha=ALPHA)

        # astronomical convention: brighter up
        ax.invert_yaxis()
        ax.set_xlabel("MJD")
        ax.set_ylabel("r-band mag")

        fvar = float(r["Fvar_percent"])
        n = int(r["N"])
        dt = float(r["Time_span"])

        ax.set_title(f"Star-forming {tid}\nFvar={fvar:.1f}%, N={n}, ΔT={dt:.0f} d", fontsize=11)
        ax.grid(True, alpha=0.25)

        used.append((tid, fvar, n, dt))
        shown += 1
        if shown >= len(axes):
            break

    # if not enough panels filled, blank the rest
    for k in range(shown, len(axes)):
        axes[k].axis("off")
        axes[k].text(0.5, 0.5, "No data", 
                    ha='center', va='center', 
                    transform=axes[k].transAxes,
                    fontsize=12, alpha=0.5)

    fig.suptitle("Figure X. Star-forming galaxy r-band lightcurves (cleanR)\n"
                 "Expected morphology: noise-dominated variability under the same cleaning cuts",
                 fontsize=14, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = os.path.join(OUTDIR, "FigureX_Starforming_lightcurves_cleanR.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    # also save which objects were used (for advisor check / reproducibility)
    out_list = os.path.join(OUTDIR, "FigureX_Starforming_lightcurves_cleanR_used_list.csv")
    pd.DataFrame(used, columns=["TARGETID", "Fvar_percent", "N", "Time_span_days"]).to_csv(out_list, index=False)
    
    # Save failed targets list
    if failed:
        out_failed = os.path.join(OUTDIR, "FigureX_Starforming_lightcurves_cleanR_failed.csv")
        pd.DataFrame(failed, columns=["TARGETID"]).to_csv(out_failed, index=False)

    print("\n" + "="*70)
    print("Results Summary:")
    print("="*70)
    print(f"Saved plot: {out_png}")
    print(f"Saved used list: {out_list}")
    
    if used:
        print(f"\nSuccessfully plotted {len(used)} lightcurves:")
        for tid, fvar, n, dt in used:
            print(f"  {tid}: Fvar={fvar:.1f}%, N={n}, ΔT={dt:.0f}d")
    
    if failed:
        print(f"\nFailed to plot {len(failed)} targets: {failed}")
        if os.path.exists(out_failed):
            print(f"Saved failed list: {out_failed}")
    
    print(f"\nTotal attempted: {len(picked)}")
    print(f"Success rate: {100*len(used)/len(picked):.1f}%")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the stats CSV file exists and is accessible")
        print("2. Verify the STARFORMING directory exists and contains clean CSV files")
        print("3. Check the Class naming in the stats CSV (should include 'Star-forming')")
        print("4. Ensure you have the required permissions to read/write files")
        import traceback
        traceback.print_exc()
