import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- fix Chinese font on Windows ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ==============
# USER SETTINGS
# ==============
BASE = r"C:\Users\30126\Desktop\AGN"   # <<< 改成你的 AGN 根目录

LC_DIR = os.path.join(BASE, "ZTF_lightcurves_CLEAN_R_MINFIX", "SEYFERT")
STAGE1 = os.path.join(BASE, "TO_ADVISOR_CHECK_MINFIX", "stage1_seyfert_bright.csv")

OUTDIR = os.path.join(BASE, "TO_ADVISOR_CHECK_MINFIX", "01_seyfert_gold_qc")
os.makedirs(OUTDIR, exist_ok=True)

# Pick how many QC targets
N_PICK = 30
RANDOM_SEED = 1

# Outlier tagging (do NOT remove): robust z using MAD in mag space
MAD_K = 4.0

# Near limitmag tagging: if (limitmag - mag) < this value -> near limit
NEAR_LIMIT_DELTA = 0.3  # mag within 0.3 mag of limitmag

# Build a PDF booklet after making PNGs
MAKE_PDF = True
PDF_NAME = "Seyfert_bright_QC_booklet.pdf"


# ======================
# HELPER FUNCTIONS
# ======================
def robust_z_mad(x):
    """Return robust z-score using MAD (mag space), along with median and MAD."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad <= 0 or np.isnan(mad):
        return np.zeros_like(x), med, mad
    z = (x - med) / (1.4826 * mad)  # 1.4826*MAD ~ sigma if Gaussian
    return z, med, mad


def load_lc(tid):
    fn = os.path.join(LC_DIR, f"{tid}_rband_clean.csv")
    if not os.path.exists(fn):
        return None, fn
    df = pd.read_csv(fn)
    return df, fn


def plot_one(tid, meta, out_png):
    df, fn = load_lc(tid)
    if df is None or len(df) < 5:
        return False, {"n_outlier": 0, "n_near_limit": 0, "oid_n": 0}

    df = df.sort_values("mjd").reset_index(drop=True)

    # --- outlier tag (do not remove) ---
    z, med, mad = robust_z_mad(df["mag"].values)
    df["is_outlier_tag"] = np.abs(z) > MAD_K

    # --- near limit tag ---
    if "limitmag" in df.columns:
        df["is_near_limit"] = (df["limitmag"] - df["mag"]) < NEAR_LIMIT_DELTA
    else:
        df["is_near_limit"] = False

    n_outlier = int(df["is_outlier_tag"].sum())
    n_near_limit = int(df["is_near_limit"].sum())
    oid_n = int(df["oid"].nunique()) if "oid" in df.columns else 0

    # --- build plot ---
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)

    # base points with errorbars
    ax.errorbar(df["mjd"], df["mag"], yerr=df["magerr"],
                fmt=".", ms=2.2, lw=0.6, alpha=0.85)

    # highlight near-limit points
    m1 = df["is_near_limit"].values
    if m1.any():
        ax.scatter(df.loc[m1, "mjd"], df.loc[m1, "mag"],
                   s=16, facecolors="none", edgecolors="orange", linewidths=0.9,
                   label=f"near limitmag (<{NEAR_LIMIT_DELTA} mag)")

    # highlight outliers (tag only)
    m2 = df["is_outlier_tag"].values
    if m2.any():
        ax.scatter(df.loc[m2, "mjd"], df.loc[m2, "mag"],
                   s=22, facecolors="none", edgecolors="red", linewidths=1.0,
                   label=f"outlier tag |z_MAD|>{MAD_K:g}")

    ax.set_xlabel("MJD")
    ax.set_ylabel("r-band mag")
    ax.invert_yaxis()

    # title with key meta
    mean_mag = float(meta.get("Mean_mag", np.nan))
    fvar = float(meta.get("Fvar_percent", np.nan))
    N = int(meta.get("N", len(df)))
    dt = float(meta.get("Time_span", np.nan))

    ax.set_title(
        f"Seyfert {tid}  Mean={mean_mag:.3f}  Fvar={fvar:.2f}%  N={N}  ΔT={dt:.0f} d",
        fontsize=10
    )

    # footer: file + oid/catflags + counts
    catflags_unique = sorted(df["catflags"].unique().tolist()) if "catflags" in df.columns else []
    footer = (
        f"file: {os.path.basename(fn)} | oid_n={oid_n} | catflags={catflags_unique} | "
        f"n_outlier={n_outlier} | n_near_limit={n_near_limit}"
    )
    ax.text(0.01, 0.02, footer, transform=ax.transAxes, fontsize=8, alpha=0.92)

    if m1.any() or m2.any():
        ax.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    return True, {"n_outlier": n_outlier, "n_near_limit": n_near_limit, "oid_n": oid_n}


def merge_pngs_to_pdf(indir, outpdf):
    """
    Merge *_QC.png into a single PDF.
    Requires pillow: pip install pillow
    """
    try:
        from PIL import Image
    except ImportError:
        print("[WARN] Pillow not installed, skip PDF merging. Install via: pip install pillow")
        return False

    pngs = sorted([f for f in os.listdir(indir) if f.lower().endswith("_qc.png")])
    if len(pngs) == 0:
        print("[WARN] No *_QC.png found, skip PDF merging.")
        return False

    imgs = []
    for f in pngs:
        path = os.path.join(indir, f)
        im = Image.open(path).convert("RGB")
        imgs.append(im)

    imgs[0].save(outpdf, save_all=True, append_images=imgs[1:])
    print("Saved PDF booklet:", outpdf)
    return True


def main():
    random.seed(RANDOM_SEED)

    S = pd.read_csv(STAGE1)

    # basic sanity filter (even if stage1 already did it)
    if "Mean_mag" in S.columns:
        S = S[S["Mean_mag"] < 18.0]
    if "N" in S.columns:
        S = S[S["N"] > 500]

    tids = S["TARGETID"].astype(str).tolist()
    if len(tids) == 0:
        print("No targets after filter. Check stage1 csv columns.")
        return

    pick = tids if len(tids) <= N_PICK else random.sample(tids, N_PICK)

    out_rows = []
    ok = 0
    for tid in pick:
        meta_row = S[S["TARGETID"].astype(str) == tid].iloc[0].to_dict()
        out_png = os.path.join(OUTDIR, f"{tid}_QC.png")
        success, info = plot_one(tid, meta_row, out_png)

        if success:
            ok += 1
            out_rows.append([tid, out_png, info["n_outlier"], info["n_near_limit"], info["oid_n"]])
        else:
            out_rows.append([tid, "FAILED", np.nan, np.nan, np.nan])

    df_index = pd.DataFrame(out_rows, columns=["TARGETID", "QC_png", "n_outlier", "n_near_limit", "oid_n"])
    idx_path = os.path.join(OUTDIR, "qc_index.csv")
    df_index.to_csv(idx_path, index=False)

    print(f"Done. QC plots saved: {ok}/{len(pick)} in {OUTDIR}")
    print("Index saved:", idx_path)

    # build PDF booklet
    if MAKE_PDF:
        outpdf = os.path.join(OUTDIR, PDF_NAME)
        merge_pngs_to_pdf(OUTDIR, outpdf)


if __name__ == "__main__":
    main()
