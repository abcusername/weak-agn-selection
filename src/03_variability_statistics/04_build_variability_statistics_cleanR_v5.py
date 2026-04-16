# 04_build_variability_statistics_cleanR_v5_MINFIX.py
import os
import glob
import numpy as np
import pandas as pd

# ======= USER PATHS =======
BASE = r"C:\Users\30126\Desktop\AGN"

# >>> 改为 MINFIX 根目录 <<<
CLEAN_BASE = os.path.join(BASE, "ZTF_lightcurves_CLEAN_R_MINFIX")

# >>> 输出统计表也改名，避免覆盖旧文件 <<<
OUT_CSV = os.path.join(BASE, "AGN_variability_statistics_v5_cleanR_MINFIX.csv")

CLASS_DIRS = {
    "Seyfert":       os.path.join(CLEAN_BASE, "SEYFERT"),
    "Composite":     os.path.join(CLEAN_BASE, "COMPOSITE"),
    "LINER":         os.path.join(CLEAN_BASE, "LINER"),
    "Star-forming":  os.path.join(CLEAN_BASE, "STARFORMING"),
}

# ======= Fvar function (r-band mag space -> flux space) =======
def mag_to_flux(mag):
    return 10 ** (-0.4 * np.asarray(mag, dtype=float))

def compute_fvar_percent(mag, magerr):
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)

    m = np.isfinite(mag) & np.isfinite(magerr) & (magerr > 0)
    mag = mag[m]; magerr = magerr[m]
    if mag.size < 10:
        return np.nan

    f = mag_to_flux(mag)
    ferr = f * (np.log(10.0) * 0.4) * magerr

    mean_f = np.mean(f)
    if not np.isfinite(mean_f) or mean_f <= 0:
        return np.nan

    s2 = np.var(f, ddof=1)
    mean_err2 = np.mean(ferr**2)

    num = s2 - mean_err2
    if num <= 0:
        return 0.0
    fvar = np.sqrt(num) / mean_f
    return 100.0 * float(fvar)

rows = []
total_scanned = 0
total_kept = 0

print("="*70)
print("Build stats from cleanR MINFIX")
print("="*70)
print("CLEAN_BASE :", CLEAN_BASE)
print("OUT_CSV    :", OUT_CSV)
print("="*70)

for cls, d in CLASS_DIRS.items():
    files = sorted(glob.glob(os.path.join(d, "*_rband_clean.csv")))
    print(f"[{cls}] files: {len(files)}  dir: {d}")

    for fp in files:
        total_scanned += 1
        tid = os.path.basename(fp).replace("_rband_clean.csv", "")

        try:
            df = pd.read_csv(fp, dtype={"oid": str})
        except Exception:
            continue

        if not {"mjd","mag","magerr"}.issubset(df.columns):
            continue

        df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["mjd","mag","magerr"])
        if len(df) < 10:
            continue

        # 强制数值化
        mjd = pd.to_numeric(df["mjd"], errors="coerce").to_numpy()
        mag = pd.to_numeric(df["mag"], errors="coerce").to_numpy()
        magerr = pd.to_numeric(df["magerr"], errors="coerce").to_numpy()
        ok = np.isfinite(mjd) & np.isfinite(mag) & np.isfinite(magerr)
        mjd, mag, magerr = mjd[ok], mag[ok], magerr[ok]
        if len(mjd) < 10:
            continue

        fvar_p = compute_fvar_percent(mag, magerr)
        n = int(len(mjd))
        dt = float(np.max(mjd) - np.min(mjd))
        mean_mag = float(np.median(mag))

        rows.append(dict(
            TARGETID=str(tid),
            Class=cls,
            N=n,
            Time_span=dt,
            Mean_mag=mean_mag,
            Fvar_percent=fvar_p
        ))
        total_kept += 1

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)

print("\nDone ✅")
print("Total files scanned :", total_scanned)
print("Files kept in stats :", total_kept)
print("Rows in OUT_CSV      :", len(out))
print("Output:", OUT_CSV)
print("\nClass counts:")
print(out["Class"].value_counts(dropna=False))

