import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0) Fix Chinese font + minus sign
# =========================
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC",
    "Noto Sans CJK SC", "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 1) Paths: samples already in folder
# =========================
BASE = r"C:\Users\30126\Desktop\AGN"
SAMPLE_DIR = os.path.join(
    BASE, "TO_ADVISOR_CHECK_MINFIX",
    "lightcurves_cleanR_MINFIX", "SEYFERT_examples"
)

OUTDIR = os.path.join(BASE, "TO_ADVISOR_CHECK_MINFIX", "auto_classify_from_samples")
PLOTDIR = os.path.join(OUTDIR, "qc_plots")
os.makedirs(PLOTDIR, exist_ok=True)

OUTCSV = os.path.join(OUTDIR, "auto_classification_samples.csv")

# =========================
# 2) Helpers
# =========================
def mag_to_flux(mag):
    return 10 ** (-0.4 * mag)

def magerr_to_fluxerr(mag, magerr):
    f = mag_to_flux(mag)
    return f * (np.log(10) * 0.4) * magerr

def robust_mad(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return mad

def split_segments_by_gap(t, gap_days=60.0):
    """
    Split by big time gaps. Each segment returns indices into original arrays.
    """
    t = np.asarray(t, dtype=float)
    order = np.argsort(t)
    t_sorted = t[order]
    dt = np.diff(t_sorted)
    cut = np.where(dt > gap_days)[0]
    boundaries = np.r_[0, cut + 1, len(t_sorted)]
    segs = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        idx_sorted = np.arange(s, e)
        segs.append(order[idx_sorted])
    return segs

def kmeans_1d_two_clusters(x, n_iter=30, seed=0):
    """
    Minimal 1D k-means (k=2) without sklearn.
    Returns: (labels, (c0,c1), x_used)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return None, None, None

    c0, c1 = np.nanpercentile(x, [30, 70])

    for _ in range(n_iter):
        d0 = np.abs(x - c0)
        d1 = np.abs(x - c1)
        lab = (d1 < d0).astype(int)

        if np.any(lab == 0):
            c0_new = np.mean(x[lab == 0])
        else:
            c0_new = c0

        if np.any(lab == 1):
            c1_new = np.mean(x[lab == 1])
        else:
            c1_new = c1

        if np.isclose(c0_new, c0) and np.isclose(c1_new, c1):
            break
        c0, c1 = c0_new, c1_new

    return lab, (c0, c1), x

def compute_features(lc):
    """
    lc must contain mjd, mag, magerr columns.
    Compute residuals in flux space and diagnostic metrics.
    """
    t = lc["mjd"].to_numpy(dtype=float)
    mag = lc["mag"].to_numpy(dtype=float)
    merr = lc["magerr"].to_numpy(dtype=float)

    good = np.isfinite(t) & np.isfinite(mag) & np.isfinite(merr)
    t, mag, merr = t[good], mag[good], merr[good]

    n = len(t)
    if n < 30:
        return None

    f = mag_to_flux(mag)
    fe = magerr_to_fluxerr(mag, merr)

    w = 1.0 / np.maximum(fe, 1e-12) ** 2
    fhat = np.sum(w * f) / np.sum(w)

    # normalized residuals in flux
    z = (f - fhat) / np.maximum(fe, 1e-12)

    z_med = np.nanmedian(z)
    z_mad = robust_mad(z)
    z_sigma_rob = 1.4826 * z_mad if z_mad > 0 else np.nanstd(z)

    absz = np.abs(z - z_med)
    frac_gt6 = float(np.mean(absz > 6.0))
    frac_gt8 = float(np.mean(absz > 8.0))
    max_absz = float(np.nanmax(absz))

    # segment drift: compare segment medians
    segs = split_segments_by_gap(t, gap_days=60.0)
    seg_meds = []
    for idx in segs:
        if len(idx) >= 30:
            seg_meds.append(np.nanmedian(z[idx]))
    seg_meds = np.array(seg_meds, dtype=float)

    if len(seg_meds) >= 2 and np.isfinite(z_sigma_rob) and z_sigma_rob > 0:
        drift_amp = float((np.nanmax(seg_meds) - np.nanmin(seg_meds)) / z_sigma_rob)
    else:
        drift_amp = 0.0

    # layering/mismatch check in mag space
    km = kmeans_1d_two_clusters(mag, seed=0)
    if km[0] is None:
        layer_sep = 0.0
        layer_balance = 0.0
    else:
        lab, (c0, c1), x = km
        n0 = np.sum(lab == 0)
        n1 = np.sum(lab == 1)
        layer_balance = float(min(n0, n1) / max(n0, n1)) if max(n0, n1) > 0 else 0.0
        if n0 < 10 or n1 < 10:
            layer_sep = 0.0
        else:
            s0 = np.std(x[lab == 0])
            s1 = np.std(x[lab == 1])
            sp = np.sqrt((s0**2 + s1**2) / 2.0)
            layer_sep = float(np.abs(c0 - c1) / (sp + 1e-12))

    return dict(
        N=n,
        z_sigma_rob=float(z_sigma_rob),
        frac_gt6=frac_gt6,
        frac_gt8=frac_gt8,
        max_absz=max_absz,
        drift_amp=drift_amp,
        layer_sep=layer_sep,
        layer_balance=layer_balance,
        t=t, mag=mag, merr=merr, z=z
    )

def classify_from_features(feat):
    """
    Priority-based classification into 4 categories.
    """
    # 1) mismatch/layering
    if (feat["layer_sep"] >= 3.0) and (feat["layer_balance"] >= 0.20):
        return "疑似错配/分层"

    # 2) drift/segment bias
    if feat["drift_amp"] >= 3.0:
        return "疑似系统漂移/分段偏置"

    # 3) outlier dominated
    if (feat["frac_gt8"] >= 0.01) or (feat["max_absz"] >= 15.0):
        return "明显离群点主导"

    # 4) normal-like
    return "曲线形态正常、像核光变"

def make_qc_plot(t, mag, merr, z, title, out_png):
    fig = plt.figure(figsize=(10, 6))

    ax1 = plt.subplot(2, 1, 1)
    ax1.errorbar(t, mag, yerr=merr, fmt='.', markersize=2, linewidth=0.5)
    ax1.invert_yaxis()
    ax1.set_ylabel("r-band mag")
    ax1.set_title(title)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t, z, '.', markersize=3)
    ax2.axhline(0, linewidth=1)
    ax2.set_xlabel("MJD")
    ax2.set_ylabel("(f - fhat) / sigma_f")  # <-- no f-hat glyph!

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def parse_tid_from_filename(path):
    name = os.path.basename(path)
    m = re.match(r"(\d+)_advisor\.csv$", name)
    return int(m.group(1)) if m else None

# =========================
# 3) Main
# =========================
csvs = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*_advisor.csv")))
if len(csvs) == 0:
    raise FileNotFoundError(f"No *_advisor.csv found in: {SAMPLE_DIR}")

rows = []

for fp in csvs:
    tid = parse_tid_from_filename(fp)
    if tid is None:
        rows.append(dict(file=fp, status="BAD_NAME"))
        continue

    lc = pd.read_csv(fp)
    if not {"mjd", "mag", "magerr"}.issubset(lc.columns):
        rows.append(dict(TARGETID=tid, file=fp, status="BAD_COLUMNS"))
        continue

    feat = compute_features(lc)
    if feat is None:
        rows.append(dict(TARGETID=tid, file=fp, status="TOO_FEW_POINTS"))
        continue

    label = classify_from_features(feat)

    title = (
        f"SEYFERT {tid}  N={feat['N']}  "
        f"[{label}]  "
        f"out8={feat['frac_gt8']:.3f}  drift={feat['drift_amp']:.2f}  "
        f"layer={feat['layer_sep']:.2f}/{feat['layer_balance']:.2f}"
    )

    out_png = os.path.join(PLOTDIR, f"{tid}_qc.png")
    make_qc_plot(feat["t"], feat["mag"], feat["merr"], feat["z"], title, out_png)

    rows.append(dict(
        TARGETID=tid,
        file=fp,
        N=feat["N"],
        label=label,
        z_sigma_rob=feat["z_sigma_rob"],
        frac_gt6=feat["frac_gt6"],
        frac_gt8=feat["frac_gt8"],
        max_absz=feat["max_absz"],
        drift_amp=feat["drift_amp"],
        layer_sep=feat["layer_sep"],
        layer_balance=feat["layer_balance"],
        qc_png=out_png,
        status="OK"
    ))

out = pd.DataFrame(rows)
out.to_csv(OUTCSV, index=False, encoding="utf-8-sig")

print("Done.")
print("SAMPLE_DIR:", SAMPLE_DIR)
print("OUTCSV:", OUTCSV)
print("PLOTDIR:", PLOTDIR)
print("\nCounts by label:")
if "label" in out.columns:
    print(out["label"].value_counts(dropna=False))
else:
    print("No label column (all failed?)")
