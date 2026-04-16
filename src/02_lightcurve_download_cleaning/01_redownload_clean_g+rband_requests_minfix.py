# 01_redownload_clean_band_requests_minfix.py
# ---------------------------------------------------
# Minimal-fix clean pipeline for ZTF light curves (g/r/i)
# Based on your r-band version, but band-generalized.

import os
import time
import numpy as np
import pandas as pd
import requests
from io import StringIO

# -----------------------------
# USER SETTINGS
# -----------------------------
BASE = r"C:\Users\30126\Desktop\AGN"

# Output root (new directory)
BASE_CLEAN = os.path.join(BASE, "ZTF_lightcurves_CLEAN_GRi_MINFIX")
os.makedirs(BASE_CLEAN, exist_ok=True)

TASKS = [
    dict(
        name="SEYFERT",
        input_csv=os.path.join(BASE, r"BPT_analysis_formula_12_13\DESI_SEYFERT_galaxies.csv"),
        out_dir=os.path.join(BASE_CLEAN, "SEYFERT"),
    ),
    dict(
        name="COMPOSITE",
        input_csv=os.path.join(BASE, r"BPT_analysis_formula_12_13\DESI_COMPOSITE_galaxies.csv"),
        out_dir=os.path.join(BASE_CLEAN, "COMPOSITE"),
    ),
    dict(
        name="LINER",
        input_csv=os.path.join(BASE, r"BPT_analysis_formula_12_13\DESI_LINER_galaxies.csv"),
        out_dir=os.path.join(BASE_CLEAN, "LINER"),
    ),
    dict(
        name="STARFORMING",
        input_csv=os.path.join(BASE, r"BPT_analysis_formula_12_13\DESI_STARFORMING_1000_control.csv"),
        out_dir=os.path.join(BASE_CLEAN, "STARFORMING"),
    ),
]

# Which bands to download (recommend start with ["g","r"])
BANDS = ["g"]   # change to ["g","r","i"] if you want i-band too

FORCE_REPROCESS = True
RADIUS_ARCSEC = 5.0

MAGERR_MAX = 0.5
MIN_POINTS_AFTER_CLEAN = 30
NSIG = 5.0

DROP_UPPERLIMITS = True
UPPERLIM_MARGIN = 0.0

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60
MAX_RETRY = 3
BACKOFF_BASE = 2.0

SLEEP_OK = 0.2
IRSA_LC_API = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"


# -----------------------------
# Utilities
# -----------------------------
def angsep_arcsec(ra1, dec1, ra2, dec2):
    ra1 = np.deg2rad(ra1); dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2); dec2 = np.deg2rad(dec2)
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(dra/2)**2
    c = 2*np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return np.rad2deg(c) * 3600.0

def sigma_clip_mad(mag, nsig=5.0):
    mag = np.asarray(mag, dtype=float)
    if mag.size < 10:
        return np.ones_like(mag, dtype=bool)
    med = np.nanmedian(mag)
    mad = np.nanmedian(np.abs(mag - med))
    if not np.isfinite(mad) or mad <= 0:
        sd = np.nanstd(mag)
        if not np.isfinite(sd) or sd <= 0:
            return np.ones_like(mag, dtype=bool)
        return np.abs(mag - med) < nsig * sd
    robust_sd = 1.4826 * mad
    return np.abs(mag - med) < nsig * robust_sd

def fetch_irsa_csv(ra, dec, radius_arcsec, band="r", target_id=None):
    radius_deg = radius_arcsec / 3600.0
    pos_value = f"CIRCLE {ra:.7f} {dec:.7f} {radius_deg:.7f}"
    params = {"POS": pos_value, "BANDNAME": band, "FORMAT": "csv"}
    headers = {"User-Agent": "Mozilla/5.0 (ZTF clean; contact: user)"}

    last_err = None
    for k in range(1, MAX_RETRY + 1):
        try:
            r = requests.get(IRSA_LC_API, params=params, headers=headers,
                             timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                raise RuntimeError(last_err)

            text = r.text.strip()
            if len(text) == 0 or text.lower().startswith("<!doctype html") or "<html" in text[:200].lower():
                last_err = "Non-CSV response (HTML/empty)"
                raise RuntimeError(last_err)

            df = pd.read_csv(StringIO(text))
            if df is None or len(df) == 0:
                return None, "EMPTY_TABLE"
            return df, None

        except Exception as e:
            last_err = str(e)
            sleep_s = (BACKOFF_BASE ** (k - 1)) + 0.2 * np.random.rand()
            if target_id:
                print(f"   {target_id} [{band}] retry {k}/{MAX_RETRY}, sleep {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    return None, last_err

def choose_central_oid(df, tra, tdec):
    need = {"oid", "ra", "dec"}
    if not need.issubset(set(df.columns)):
        return None, None
    seps = angsep_arcsec(tra, tdec, df["ra"].values, df["dec"].values)
    tmp = df.copy()
    tmp["_sep"] = seps
    g = tmp.groupby("oid")["_sep"].median()
    best_oid = g.idxmin()
    best_sep = float(g.min())
    return best_oid, best_sep

def clean_band_lc(sub):
    for c in ["mjd", "mag", "magerr"]:
        if c not in sub.columns:
            return None, f"missing {c}"

    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["mjd", "mag", "magerr"])

    if DROP_UPPERLIMITS and ("limitmag" in sub.columns):
        lm = pd.to_numeric(sub["limitmag"], errors="coerce")
        sub["limitmag"] = lm
        keep_det = lm.isna() | (sub["mag"] < (lm - UPPERLIM_MARGIN))
        sub = sub[keep_det]

    if "catflags" in sub.columns:
        sub = sub[sub["catflags"].fillna(0).astype(int) == 0]

    sub = sub[(sub["magerr"] > 0) & (sub["magerr"] <= MAGERR_MAX)]

    if len(sub) < MIN_POINTS_AFTER_CLEAN:
        return None, f"too_few_after_qc n={len(sub)}"

    keep = sigma_clip_mad(sub["mag"].values, nsig=NSIG)
    sub = sub[keep]

    if len(sub) < MIN_POINTS_AFTER_CLEAN:
        return None, f"too_few_after_clip n={len(sub)}"

    return sub.sort_values("mjd"), None


# -----------------------------
# Main
# -----------------------------
def process_one_class(task):
    name = task["name"]
    input_csv = task["input_csv"]
    out_dir = task["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    df0 = pd.read_csv(input_csv)
    ra_arr = df0["TARGET_RA"].values
    dec_arr = df0["TARGET_DEC"].values
    tid_arr = df0["TARGETID"].values

    total = len(df0)

    for band in BANDS:
        log_path = os.path.join(os.path.dirname(out_dir), f"cleaning_log_{name}_{band}.csv")

        ok = 0
        skip_done = 0
        logs = []
        start_time = time.time()

        print(f"\n[{name}] band={band} total={total}")
        print(f"[{name}] MAGERR_MAX={MAGERR_MAX}, DROP_UPPERLIMITS={DROP_UPPERLIMITS}, UPPERLIM_MARGIN={UPPERLIM_MARGIN}")
        print(f"[{name}] out_dir={out_dir} FORCE_REPROCESS={FORCE_REPROCESS}")

        for i in range(total):
            tid = str(tid_arr[i])
            tra = float(ra_arr[i])
            tdec = float(dec_arr[i])

            out_csv = os.path.join(out_dir, f"{tid}_{band}band_clean.csv")

            if (not FORCE_REPROCESS) and os.path.exists(out_csv):
                skip_done += 1
                continue

            if (i < 50 and (i + 1) % 5 == 0) or ((i + 1) % 50 == 0) or (i + 1 == total):
                pct = 100.0 * (i + 1) / total
                elapsed = time.time() - start_time
                rem = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else np.nan
                print(f"[{name}][{band}] {i+1}/{total} ({pct:5.1f}%) OK={ok} skip={skip_done} t={elapsed:.0f}s rem={rem:.0f}s")

            raw, err = fetch_irsa_csv(tra, tdec, RADIUS_ARCSEC, band=band, target_id=tid)
            if raw is None:
                logs.append(dict(TARGETID=tid, band=band, status="NO_DATA", reason=err or "empty"))
                continue

            best_oid, best_sep = choose_central_oid(raw, tra, tdec)
            if best_oid is None:
                logs.append(dict(TARGETID=tid, band=band, status="NO_OID", reason=f"cols={list(raw.columns)}"))
                continue

            sub = raw[raw["oid"] == best_oid].copy()
            clean, cerr = clean_band_lc(sub)
            if clean is None:
                logs.append(dict(TARGETID=tid, band=band, status="FAIL_CLEAN", reason=cerr,
                                 chosen_oid=str(best_oid), chosen_sep_arcsec=best_sep, n_oid=len(sub)))
                continue

            keep_cols = [c for c in ["mjd", "mag", "magerr", "oid", "ra", "dec", "fid", "catflags", "limitmag"] if c in clean.columns]
            clean[keep_cols].to_csv(out_csv, index=False)

            ok += 1
            logs.append(dict(TARGETID=tid, band=band, status="OK", reason="", chosen_oid=str(best_oid),
                             chosen_sep_arcsec=best_sep, n_clean=len(clean)))

            time.sleep(SLEEP_OK)

        pd.DataFrame(logs).to_csv(log_path, index=False)
        total_time = time.time() - start_time
        print(f"\n[{name}] band={band} DONE: OK={ok}/{total}, skip={skip_done}, time={total_time/60:.1f} min")
        print(f"log: {log_path}")

    return


if __name__ == "__main__":
    print("=" * 70)
    print("ZTF clean MINFIX (band-generalized)")
    print("=" * 70)
    print(f"Output root: {BASE_CLEAN}")
    print(f"BANDS: {BANDS}")
    print("=" * 70)

    for t in TASKS:
        process_one_class(t)

    print("\nAll done.")
