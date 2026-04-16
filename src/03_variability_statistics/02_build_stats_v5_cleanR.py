# 02_build_stats_v5_cleanR_minfix.py
# Build variability statistics CSV from cleanR MINFIX outputs
# - Reads *_rband_clean.csv for each class
# - Computes flux-based Fvar (%)
# - Outputs AGN_variability_statistics_v5_cleanR_MINFIX.csv
#
# Compatible with your plotting script (Figure1/2/3):
#   order = ["Seyfert", "Composite", "LINER", "Star-forming"]

import os
import glob
import numpy as np
import pandas as pd

# -----------------------------
# PATHS
# -----------------------------
BASE = r"C:\Users\30126\Desktop\AGN"

# IMPORTANT: point to your MINFIX directory
CLEAN_ROOT = os.path.join(BASE, "ZTF_lightcurves_CLEAN_R_MINFIX")

# Output stats CSV (new name to avoid mixing with old run)
OUTCSV = os.path.join(BASE, "AGN_variability_statistics_v5_cleanR_MINFIX.csv")

# Classes present in MINFIX root
CLASSES = ["SEYFERT", "COMPOSITE", "LINER", "STARFORMING"]

# -----------------------------
# Helpers
# -----------------------------
def mag_to_flux(mag):
    # relative flux, constant cancels in Fvar
    return 10.0 ** (-0.4 * np.asarray(mag, dtype=float))

def magerr_to_fluxerr(mag, magerr):
    # df/f = ln(10)*0.4*magerr
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)
    f = mag_to_flux(mag)
    return f * (np.log(10.0) * 0.4) * magerr

def compute_fvar_percent(mag, magerr):
    """
    Flux-based fractional variability:
      Fvar = sqrt(S^2 - mean(err^2)) / mean(flux)
    Return percent (%). If invalid, return NaN.
    """
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    ok = np.isfinite(mag) & np.isfinite(magerr) & (magerr > 0)
    mag = mag[ok]
    magerr = magerr[ok]
    if mag.size < 10:
        return np.nan

    f = mag_to_flux(mag)
    ferr = magerr_to_fluxerr(mag, magerr)

    mean_f = np.mean(f)
    if not np.isfinite(mean_f) or mean_f <= 0:
        return np.nan

    s2 = np.var(f, ddof=1)          # sample variance
    mean_err2 = np.mean(ferr ** 2)

    excess = s2 - mean_err2
    if not np.isfinite(excess) or excess <= 0:
        return 0.0

    fvar = np.sqrt(excess) / mean_f
    return 100.0 * float(fvar)

def canon_class_name(cls_dir):
    """
    Map folder name -> plot-friendly class name
    """
    c = cls_dir.strip().upper()
    if c == "SEYFERT":
        return "Seyfert"
    if c == "COMPOSITE":
        return "Composite"
    if c == "LINER":
        return "LINER"
    if c == "STARFORMING":
        return "Star-forming"
    return cls_dir.title()

# -----------------------------
# Main
# -----------------------------
def main():
    if not os.path.isdir(CLEAN_ROOT):
        raise FileNotFoundError(f"CLEAN_ROOT not found: {CLEAN_ROOT}")

    rows = []
    total_files = 0
    kept_files = 0

    print("=" * 70)
    print("Build stats from cleanR MINFIX")
    print("=" * 70)
    print("CLEAN_ROOT:", CLEAN_ROOT)
    print("OUTCSV    :", OUTCSV)
    print("=" * 70)

    for cls in CLASSES:
        d = os.path.join(CLEAN_ROOT, cls)
        files = sorted(glob.glob(os.path.join(d, "*_rband_clean.csv")))
        print(f"[{cls}] files: {len(files)}  dir: {d}")
        total_files += len(files)

        class_label = canon_class_name(cls)

        for fp in files:
            bn = os.path.basename(fp)
            tid = bn.replace("_rband_clean.csv", "")

            try:
                df = pd.read_csv(fp)
            except Exception:
                continue

            # required columns
            need = {"mjd", "mag", "magerr"}
            if not need.issubset(df.columns):
                continue

            mjd = pd.to_numeric(df["mjd"], errors="coerce").to_numpy()
            mag = pd.to_numeric(df["mag"], errors="coerce").to_numpy()
            magerr = pd.to_numeric(df["magerr"], errors="coerce").to_numpy()

            ok = np.isfinite(mjd) & np.isfinite(mag) & np.isfinite(magerr) & (magerr > 0)
            mjd, mag, magerr = mjd[ok], mag[ok], magerr[ok]

            if len(mjd) < 10:
                continue

            n = int(len(mjd))
            time_span = float(np.max(mjd) - np.min(mjd))
            mean_mag = float(np.median(mag))  # median更稳健
            fvar = compute_fvar_percent(mag, magerr)

            rows.append(dict(
                TARGETID=str(tid),
                Class=class_label,
                N=n,
                Mean_mag=mean_mag,
                Time_span=time_span,
                Fvar_percent=fvar
            ))
            kept_files += 1

    out = pd.DataFrame(rows)

    # Basic sanity cleanup
    out["Fvar_percent"] = pd.to_numeric(out["Fvar_percent"], errors="coerce")
    out["N"] = pd.to_numeric(out["N"], errors="coerce")
    out["Mean_mag"] = pd.to_numeric(out["Mean_mag"], errors="coerce")
    out["Time_span"] = pd.to_numeric(out["Time_span"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Class", "Fvar_percent", "N", "Mean_mag", "Time_span"])

    out.to_csv(OUTCSV, index=False)

    print("\nDone ✅")
    print(f"Total files scanned : {total_files}")
    print(f"Files kept in stats : {kept_files}")
    print(f"Rows in OUTCSV      : {len(out)}")
    print("Output:", OUTCSV)

    # quick class counts
    if len(out) > 0:
        print("\nClass counts:")
        print(out["Class"].value_counts())

if __name__ == "__main__":
    main()
