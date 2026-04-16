import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- fix Chinese font on Windows ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# =================
# USER SETTINGS
# =================
BASE = r"C:\Users\30126\Desktop\AGN"  # <<< 改成你的 AGN 根目录
STATS_STEM = "AGN_variability_statistics_v7_cleanR_MINFIX_withPvar_FvarSys"  # v7 stats（不写扩展也行）

# MINFIX lightcurves
LC_BASE = os.path.join(BASE, "ZTF_lightcurves_CLEAN_R_MINFIX")  # contains STARFORMING/COMPOSITE folders

# outputs
OUTDIR = os.path.join(BASE, "TO_ADVISOR_CHECK_MINFIX", "03_SF_composite_highFvar_qc")
os.makedirs(OUTDIR, exist_ok=True)

# selection
TOP_K = 20
FVAR_THR = 10.0  # percent, per advisor example

RANDOM_SEED = 1
random.seed(RANDOM_SEED)

# tagging (do NOT remove)
MAD_K = 4.0
NEAR_LIMIT_DELTA = 0.3

# if file missing in MINFIX, skip
SKIP_MISSING_LC = True


# =================
# HELPERS
# =================
def resolve_stats_path(base, stem):
    direct = os.path.join(base, stem)
    if os.path.exists(direct):
        return direct

    exts = ["csv", "CSV", "txt", "TXT", "tsv", "TSV"]
    for ext in exts:
        p = os.path.join(base, f"{stem}.{ext}")
        if os.path.exists(p):
            return p

    cand = []
    for ext in exts:
        cand.extend(glob.glob(os.path.join(base, f"{stem}*.{ext}")))
    if len(cand) > 0:
        cand.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return cand[0]

    raise FileNotFoundError(f"Cannot find stats file for stem '{stem}' under: {base}")


def read_table_auto(path):
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        pass
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")


def classify_classname(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if "SEY" in s:
        return "Seyfert"
    if "LIN" in s:
        return "LINER"
    if "COMP" in s:
        return "Composite"
    if "STAR" in s:
        return "Star-forming"
    if s == "SF":
        return "Star-forming"
    return str(x).strip()


def robust_z_mad(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad <= 0 or np.isnan(mad):
        return np.zeros_like(x), med, mad
    z = (x - med) / (1.4826 * mad)
    return z, med, mad


def load_lc(class_folder, tid):
    # files are like: <LC_BASE>/<CLASS>/<TARGETID>_rband_clean.csv
    fn = os.path.join(LC_BASE, class_folder, f"{tid}_rband_clean.csv")
    if not os.path.exists(fn):
        return None, fn
    df = pd.read_csv(fn)
    return df, fn


def label_reason(n_near_limit, n_outlier, max_magerr, fvar, mean_mag):
    """
    A very simple heuristic tag for advisor-facing interpretation.
    You can refine later.
    """
    tags = []
    if n_near_limit >= 5:
        tags.append("near_limitmag")
    if max_magerr >= 0.15:
        tags.append("large_magerr")
    if n_outlier >= 15:
        tags.append("many_outliers")
    if (fvar >= 20) and (mean_mag < 18) and (n_near_limit == 0) and (max_magerr < 0.08):
        tags.append("real_candidate")
    if len(tags) == 0:
        tags.append("unclear")
    return "|".join(tags)


def plot_qc_one(df, meta, fn, out_png):
    df = df.sort_values("mjd").reset_index(drop=True)

    # tag outliers
    z, med, mad = robust_z_mad(df["mag"].values)
    df["is_outlier_tag"] = np.abs(z) > MAD_K

    # near limit
    if "limitmag" in df.columns:
        df["is_near_limit"] = (df["limitmag"] - df["mag"]) < NEAR_LIMIT_DELTA
    else:
        df["is_near_limit"] = False

    n_outlier = int(df["is_outlier_tag"].sum())
    n_near_limit = int(df["is_near_limit"].sum())
    oid_n = int(df["oid"].nunique()) if "oid" in df.columns else 0
    max_magerr = float(np.nanmax(df["magerr"].values)) if "magerr" in df.columns else np.nan

    # meta
    tid = str(meta.get("TARGETID"))
    cls = str(meta.get("Class_clean"))
    mean_mag = float(meta.get("Mean_mag", np.nan))
    fvar = float(meta.get("Fvar_percent", np.nan))
    N = int(meta.get("N", len(df)))
    dt = float(meta.get("Time_span", np.nan))

    reason = label_reason(n_near_limit, n_outlier, max_magerr, fvar, mean_mag)

    # plot
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    ax.errorbar(df["mjd"], df["mag"], yerr=df["magerr"], fmt=".", ms=2.2, lw=0.6, alpha=0.85)

    m1 = df["is_near_limit"].values
    if m1.any():
        ax.scatter(df.loc[m1, "mjd"], df.loc[m1, "mag"], s=16, facecolors="none",
                   edgecolors="orange", linewidths=0.9, label=f"near limitmag (<{NEAR_LIMIT_DELTA} mag)")

    m2 = df["is_outlier_tag"].values
    if m2.any():
        ax.scatter(df.loc[m2, "mjd"], df.loc[m2, "mag"], s=22, facecolors="none",
                   edgecolors="red", linewidths=1.0, label=f"outlier tag |z_MAD|>{MAD_K:g}")

    ax.set_xlabel("MJD")
    ax.set_ylabel("r-band mag")
    ax.invert_yaxis()

    ax.set_title(f"{cls} {tid}  Mean={mean_mag:.3f}  Fvar={fvar:.2f}%  N={N}  ΔT={dt:.0f} d", fontsize=10)

    catflags_unique = sorted(df["catflags"].unique().tolist()) if "catflags" in df.columns else []
    footer = (
        f"file: {os.path.basename(fn)} | oid_n={oid_n} | catflags={catflags_unique} | "
        f"n_outlier={n_outlier} | n_near_limit={n_near_limit} | max_magerr={max_magerr:.3f} | tag={reason}"
    )
    ax.text(0.01, 0.02, footer, transform=ax.transAxes, fontsize=8, alpha=0.92)

    if m1.any() or m2.any():
        ax.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    return df, {
        "TARGETID": tid,
        "Class": cls,
        "Mean_mag": mean_mag,
        "Fvar_percent": fvar,
        "N": N,
        "Time_span": dt,
        "oid_n": oid_n,
        "n_outlier": n_outlier,
        "n_near_limit": n_near_limit,
        "max_magerr": max_magerr,
        "reason_tag": reason,
        "lc_file": fn,
        "qc_png": out_png
    }


def merge_pngs_to_pdf(indir, outpdf, pattern="*_QC.png"):
    try:
        from PIL import Image
    except ImportError:
        print("[WARN] Pillow not installed, skip PDF merging. Install via: pip install pillow")
        return False

    pngs = sorted(glob.glob(os.path.join(indir, pattern)))
    if len(pngs) == 0:
        print("[WARN] No PNGs found to merge:", indir, pattern)
        return False

    imgs = [Image.open(p).convert("RGB") for p in pngs]
    imgs[0].save(outpdf, save_all=True, append_images=imgs[1:])
    print("Saved PDF:", outpdf)
    return True


# =================
# MAIN
# =================
def main():
    stats_path = resolve_stats_path(BASE, STATS_STEM)
    print("Using STATS:", stats_path)
    S = read_table_auto(stats_path)

    # normalize class
    S["Class_clean"] = S["Class"].apply(classify_classname)

    # numeric
    for c in ["N", "Time_span", "Mean_mag", "Fvar_percent"]:
        if c in S.columns:
            S[c] = pd.to_numeric(S[c], errors="coerce")

    # folders mapping
    class_to_folder = {
        "Star-forming": "STARFORMING",
        "Composite": "COMPOSITE"
    }

    summary_all = []

    for cls, folder in class_to_folder.items():
        sub = S[S["Class_clean"] == cls].copy()
        sub = sub[np.isfinite(sub["Fvar_percent"])].copy()

        # choose high-Fvar candidates
        hi = sub[sub["Fvar_percent"] >= FVAR_THR].copy()
        hi = hi.sort_values("Fvar_percent", ascending=False)

        if len(hi) >= TOP_K:
            pick = hi.head(TOP_K)
        else:
            # if not enough above threshold, fill with top Fvar
            pick = sub.sort_values("Fvar_percent", ascending=False).head(TOP_K)

        pick_tids = pick["TARGETID"].astype(str).tolist()

        # output dirs
        out_cls_dir = os.path.join(OUTDIR, cls.replace("-", "").replace(" ", ""))
        os.makedirs(out_cls_dir, exist_ok=True)

        print(f"\n[{cls}] selected {len(pick_tids)} targets (threshold={FVAR_THR}%) -> {out_cls_dir}")

        for tid in pick_tids:
            meta = pick[pick["TARGETID"].astype(str) == tid].iloc[0].to_dict()
            df_lc, fn = load_lc(folder, tid)

            if df_lc is None:
                msg = f"[WARN] Missing LC: {fn}"
                print(msg)
                if SKIP_MISSING_LC:
                    summary_all.append({
                        "TARGETID": tid,
                        "Class": cls,
                        "reason_tag": "missing_lc",
                        "lc_file": fn
                    })
                    continue
                else:
                    raise FileNotFoundError(fn)

            out_png = os.path.join(out_cls_dir, f"{tid}_QC.png")
            df_qc, summ = plot_qc_one(df_lc, meta, fn, out_png)

            # save tagged outliers table (only tagged points)
            outlier_df = df_qc[df_qc["is_outlier_tag"]].copy()
            out_outliers = os.path.join(out_cls_dir, f"{tid}_outliers_tagged.csv")
            outlier_df.to_csv(out_outliers, index=False)

            summ["outliers_csv"] = out_outliers
            summary_all.append(summ)

        # merge into pdf
        pdf_path = os.path.join(out_cls_dir, f"{cls}_highFvar_QC_booklet.pdf".replace("-", "").replace(" ", ""))
        merge_pngs_to_pdf(out_cls_dir, pdf_path, pattern="*_QC.png")

    # save summary
    out_summary = os.path.join(OUTDIR, "highFvar_qc_summary.csv")
    pd.DataFrame(summary_all).to_csv(out_summary, index=False)
    print("\nSaved summary:", out_summary)
    print("Done.")


if __name__ == "__main__":
    main()
