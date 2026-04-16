import os
import glob
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
STATS_STEM = "AGN_variability_statistics_v7_cleanR_MINFIX_withPvar_FvarSys"  # 你确认过的 v7 文件名（不含扩展也行）

OUTDIR = os.path.join(BASE, "TO_ADVISOR_CHECK_MINFIX", "02_noise_control")
os.makedirs(OUTDIR, exist_ok=True)

# magnitude bins (edges)
MAG_BINS = np.arange(13.0, 20.5, 1.0)  # 13-14,...,19-20
SF_Q = 0.95  # SF 95% quantile threshold

# >>> IMPORTANT CHANGE (B): lower min count so bright bins are not empty
MIN_N_IN_BIN = 10

# sys floors to process
SYS_LIST = ["sys000", "sys010", "sys020", "sys030", "sys050"]

# >>> IMPORTANT CHANGE (C): use chi2red threshold instead of mlog10p
CHI2RED_THR = 1.5  # you can try 2.0 as a more conservative alternative

# "Seyfert gold sample" filter (A): Mean_mag<18 and N>500
SEYFERT_GOLD_MEANMAG_MAX = 18.0
SEYFERT_GOLD_N_MIN = 500


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


def bin_labels(edges):
    return [f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(len(edges)-1)]


def compute_bin_index(x, edges):
    idx = np.digitize(x, edges) - 1
    idx[(x < edges[0]) | (x >= edges[-1])] = -1
    return idx


def wilson_interval(k, n, z=1.0):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z*np.sqrt((p*(1-p) + z*z/(4*n))/n)) / denom
    return (max(0.0, center-half), min(1.0, center+half))


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# =================
# LOAD
# =================
STATS_PATH = resolve_stats_path(BASE, STATS_STEM)
print("Using STATS:", STATS_PATH)

df = read_table_auto(STATS_PATH)
print("Loaded rows:", len(df), "cols:", len(df.columns))

# Required columns
need_cols = ["TARGETID", "Class", "N", "Mean_mag", "Fvar_percent"]
for c in need_cols:
    if c not in df.columns:
        raise RuntimeError(f"Missing column '{c}' in stats file. Found columns: {list(df.columns)[:30]} ...")

df["Class_clean"] = df["Class"].apply(classify_classname)
keep = df["Class_clean"].isin(["Seyfert", "LINER", "Composite", "Star-forming"])
df = df[keep].copy()

# numeric cleanup
df["N"] = pd.to_numeric(df["N"], errors="coerce")
df["Mean_mag"] = pd.to_numeric(df["Mean_mag"], errors="coerce")
df["Fvar_percent"] = pd.to_numeric(df["Fvar_percent"], errors="coerce")
df = df[np.isfinite(df["N"]) & np.isfinite(df["Mean_mag"]) & np.isfinite(df["Fvar_percent"])].copy()
df = df[(df["Mean_mag"] > 10) & (df["Mean_mag"] < 25)].copy()

# magnitude bins
edges = MAG_BINS
labels = bin_labels(edges)
df["mag_bin"] = compute_bin_index(df["Mean_mag"].values, edges)

# =================
# FIG 1: Fvar vs Mean_mag scatter (all)
# =================
plt.figure(figsize=(7.2, 5.2), dpi=160)
for cls in ["Seyfert", "LINER", "Composite", "Star-forming"]:
    sub = df[df["Class_clean"] == cls]
    plt.scatter(sub["Mean_mag"], sub["Fvar_percent"], s=6, alpha=0.5, label=f"{cls} (n={len(sub)})")
plt.xlabel("Mean r-band mag")
plt.ylabel("Fvar_percent")
plt.title("Fvar vs Mean_mag ")
plt.ylim(bottom=0)
plt.legend(fontsize=8, frameon=True)
save_fig(os.path.join(OUTDIR, "Fvar_vs_MeanMag_all_classes.png"))

# =================
# D: SF noise-floor threshold table (Fvar 95% quantile per mag-bin)
# =================
thr_col = f"SF_Fvar_q{int(SF_Q*100)}"
thr_rows = []
for b in range(len(labels)):
    sf = df[(df["Class_clean"] == "Star-forming") & (df["mag_bin"] == b)]
    if len(sf) >= MIN_N_IN_BIN:
        thr = float(np.nanquantile(sf["Fvar_percent"], SF_Q))
    else:
        thr = np.nan
    thr_rows.append([b, labels[b], len(sf), thr])

thr_df = pd.DataFrame(thr_rows, columns=["bin_id", "mag_bin", "n_SF", thr_col])
thr_df.to_csv(os.path.join(OUTDIR, "SF_thresholds_by_magbin_Fvar.csv"), index=False)

# =================
# FIG 2a: Fvar fraction using SF baseline threshold (ALL classes)
# =================
def compute_fvar_fraction_table(df_in, tag):
    rows = []
    for cls in ["Seyfert", "LINER", "Composite", "Star-forming"]:
        for b in range(len(labels)):
            sub = df_in[(df_in["Class_clean"] == cls) & (df_in["mag_bin"] == b)]
            thr = thr_df.loc[thr_df["bin_id"] == b, thr_col].values[0]
            if (len(sub) >= MIN_N_IN_BIN) and np.isfinite(thr):
                k = int((sub["Fvar_percent"] > thr).sum())
                n = int(len(sub))
                lo, hi = wilson_interval(k, n, z=1.0)
                frac = k / n
            else:
                k, n, frac, lo, hi = (np.nan, int(len(sub)), np.nan, np.nan, np.nan)
            rows.append([tag, cls, b, labels[b], n, k, frac, lo, hi])
    return pd.DataFrame(rows, columns=["tag", "Class", "bin_id", "mag_bin", "n", "k_above_thr", "frac", "frac_lo", "frac_hi"])

def plot_fraction(df_frac, title, outpng):
    plt.figure(figsize=(7.4, 5.2), dpi=160)
    x = np.arange(len(labels))
    for cls in ["Seyfert", "LINER", "Composite", "Star-forming"]:
        sub = df_frac[df_frac["Class"] == cls].sort_values("bin_id")
        y = sub["frac"].values.astype(float)
        ylo = sub["frac_lo"].values.astype(float)
        yhi = sub["frac_hi"].values.astype(float)
        m = np.isfinite(y)
        plt.plot(x[m], y[m], marker="o", linewidth=1.4, label=cls)
        plt.vlines(x[m], ylo[m], yhi[m], linewidth=1.0, alpha=0.8)
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Mean_mag bin")
    plt.ylabel(f"Fraction(Fvar > SF q{int(SF_Q*100)} in same mag bin)")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend(fontsize=8, frameon=True)
    save_fig(outpng)

frac_all = compute_fvar_fraction_table(df, tag="ALL")
frac_all.to_csv(os.path.join(OUTDIR, "fractions_by_magbin_FvarSF95_ALL.csv"), index=False)
plot_fraction(
    frac_all,
    "Variability fraction with magnitude-dependent threshold (SF baseline) — ALL",
    os.path.join(OUTDIR, "VariabilityFraction_magbin_SF95_Fvar_ALL.png")
)

# =================
# FIG 2b (A): Seyfert GOLD only (Mean_mag<18 & N>500), others remain full
# We implement: restrict only Seyfert rows, while LINER/Composite/SF unchanged.
# =================
df_gold = df.copy()
is_sey = df_gold["Class_clean"] == "Seyfert"
df_gold = pd.concat([
    df_gold[~is_sey],
    df_gold[is_sey & (df_gold["Mean_mag"] < SEYFERT_GOLD_MEANMAG_MAX) & (df_gold["N"] > SEYFERT_GOLD_N_MIN)]
], ignore_index=True)

frac_gold = compute_fvar_fraction_table(df_gold, tag="SEYFERT_GOLD")
frac_gold.to_csv(os.path.join(OUTDIR, "fractions_by_magbin_FvarSF95_SeyfertGold.csv"), index=False)
plot_fraction(
    frac_gold,
    f"Variability fraction (SF baseline) — Seyfert GOLD (Mean<{SEYFERT_GOLD_MEANMAG_MAX}, N>{SEYFERT_GOLD_N_MIN})",
    os.path.join(OUTDIR, "VariabilityFraction_magbin_SF95_Fvar_SeyfertGold.png")
)

# =================
# C: chi2red threshold fraction (by sys floors)
# =================
cov_rows = []

def compute_chi2red_fraction_by_sys(df_in, sys):
    chi2red_col = f"chi2red_{sys}"
    if chi2red_col not in df_in.columns:
        return None, None

    tmp = df_in.copy()
    tmp[chi2red_col] = pd.to_numeric(tmp[chi2red_col], errors="coerce")

    # coverage per class
    for cls in ["Seyfert", "LINER", "Composite", "Star-forming"]:
        sub = tmp[tmp["Class_clean"] == cls]
        valid = np.isfinite(sub[chi2red_col].values)
        cov_rows.append([sys, cls, int(valid.sum()), int(len(sub))])

    rows = []
    for cls in ["Seyfert", "LINER", "Composite", "Star-forming"]:
        for b in range(len(labels)):
            sub = tmp[(tmp["Class_clean"] == cls) & (tmp["mag_bin"] == b)]
            val = sub[chi2red_col].values
            val = val[np.isfinite(val)]
            n = int(len(val))
            if n >= MIN_N_IN_BIN:
                k = int((val > CHI2RED_THR).sum())
                lo, hi = wilson_interval(k, n, z=1.0)
                frac = k / n
            else:
                k, frac, lo, hi = (np.nan, np.nan, np.nan, np.nan)
            rows.append([sys, cls, b, labels[b], n, k, frac, lo, hi])
    out = pd.DataFrame(rows, columns=["sys", "Class", "bin_id", "mag_bin", "n_valid", "k_sig", "frac", "frac_lo", "frac_hi"])
    return out, chi2red_col

def plot_chi2red_fraction(subdf, sys, outpng):
    plt.figure(figsize=(7.4, 5.2), dpi=160)
    x = np.arange(len(labels))
    for cls in ["Seyfert", "LINER", "Composite", "Star-forming"]:
        s = subdf[subdf["Class"] == cls].sort_values("bin_id")
        y = s["frac"].values.astype(float)
        ylo = s["frac_lo"].values.astype(float)
        yhi = s["frac_hi"].values.astype(float)
        m = np.isfinite(y)
        plt.plot(x[m], y[m], marker="o", linewidth=1.4, label=cls)
        plt.vlines(x[m], ylo[m], yhi[m], linewidth=1.0, alpha=0.8)
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Mean_mag bin")
    plt.ylabel(f"Fraction(chi2red_{sys} > {CHI2RED_THR:g})")
    plt.title(f"chi2red-based variability fraction by mag bin ({sys})")
    plt.ylim(0, 1)
    plt.legend(fontsize=8, frameon=True)
    save_fig(outpng)

for sys in SYS_LIST:
    subdf, colname = compute_chi2red_fraction_by_sys(df, sys)
    if subdf is None:
        print(f"[WARN] Missing chi2red_{sys}, skip")
        continue
    subdf.to_csv(os.path.join(OUTDIR, f"fractions_by_magbin_chi2red_gt_{CHI2RED_THR:g}_{sys}.csv"), index=False)
    plot_chi2red_fraction(
        subdf, sys,
        os.path.join(OUTDIR, f"Chi2redFraction_magbin_gt_{CHI2RED_THR:g}_{sys}.png")
    )

cov_df = pd.DataFrame(cov_rows, columns=["sys", "Class", "n_valid", "n_total"])
cov_df.to_csv(os.path.join(OUTDIR, "coverage_by_sys_and_class.csv"), index=False)

print("\nDone.")
print("Outputs written to:", OUTDIR)
print("Key outputs:")
print(" - Fvar_vs_MeanMag_all_classes.png")
print(" - SF_thresholds_by_magbin_Fvar.csv  (send this table to advisor)")
print(" - VariabilityFraction_magbin_SF95_Fvar_ALL.png")
print(" - VariabilityFraction_magbin_SF95_Fvar_SeyfertGold.png")
print(" - Chi2redFraction_magbin_gt_*.png (sys000/010/020/030/050)")
