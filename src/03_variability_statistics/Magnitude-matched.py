import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
CSV_PATH = r"C:\Users\30126\Desktop\AGN\AGN_variability_statistics_v5_cleanR_MINFIX.csv"
FVAR_COL = "Fvar_percent"
MAG_COL  = "Mean_mag"
CLASS_COL = "Class"

# variability threshold (percent)
FVAR_THR = 5.0

# magnitude bin settings
BIN_W = 0.2
MAG_MIN, MAG_MAX = 14.0, 21.0

# matching settings
N_BOOT = 200        # repeats for uncertainty
RANDOM_SEED = 42

# which AGN group to compare with SF
AGN_CLASSES = ["liner", "seyfert"]   # you can add "composite" if desired
SF_CLASS = "star-forming"

# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

# normalize class labels
df[CLASS_COL] = df[CLASS_COL].astype(str).str.strip().str.lower()

# basic sanity
need_cols = {FVAR_COL, MAG_COL, CLASS_COL}
missing = need_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}. Columns={list(df.columns)}")

# keep finite rows
df = df[np.isfinite(df[FVAR_COL]) & np.isfinite(df[MAG_COL])].copy()

# define variable flag
df["is_var"] = df[FVAR_COL] >= FVAR_THR

df_sf  = df[df[CLASS_COL] == SF_CLASS].copy()
df_agn = df[df[CLASS_COL].isin(AGN_CLASSES)].copy()

print("========== BASIC ==========")
print("Total rows:", len(df))
print("SF rows   :", len(df_sf))
print("AGN rows  :", len(df_agn))
print(f"Threshold : {FVAR_THR:.1f}%")
print("===========================")

# =========================
# 1) BINNED COMPARISON
# =========================
bins = np.arange(MAG_MIN, MAG_MAX + BIN_W, BIN_W)
labels = (bins[:-1] + bins[1:]) / 2

def binned_fraction(d):
    d = d.copy()
    d["bin"] = pd.cut(d[MAG_COL], bins=bins, include_lowest=True)
    g = d.groupby("bin", observed=True)
    out = pd.DataFrame({
        "N": g.size(),
        "frac_var": g["is_var"].mean(),
        "mag_center": labels[:len(g.size())]  # safe alignment later
    })
    out["mag_center"] = [interval.mid for interval in out.index]
    return out.reset_index(drop=True)

sf_bin  = binned_fraction(df_sf)
agn_bin = binned_fraction(df_agn)

# merge on mag_center via nearest (bins align, so exact mid should align)
m = pd.merge(sf_bin, agn_bin, on="mag_center", how="outer", suffixes=("_SF", "_AGN")).sort_values("mag_center")

print("\n========== BINNED (Mean_mag bins) ==========")
print(m.head(10))

# =========================
# 2) MAGNITUDE-MATCHED COMPARISON (stratified resampling)
# =========================
rng = np.random.default_rng(RANDOM_SEED)

# assign bins to each row
df_sf["bin"] = pd.cut(df_sf[MAG_COL], bins=bins, include_lowest=True)
df_agn["bin"] = pd.cut(df_agn[MAG_COL], bins=bins, include_lowest=True)

# count per bin in AGN
agn_counts = df_agn["bin"].value_counts(dropna=False).sort_index()
sf_counts  = df_sf["bin"].value_counts(dropna=False).sort_index()

common_bins = [b for b in agn_counts.index if pd.notna(b) and (b in sf_counts.index)]
# use the MIN count per bin to match
target_counts = {}
for b in common_bins:
    na = int(agn_counts.get(b, 0))
    ns = int(sf_counts.get(b, 0))
    k = min(na, ns)
    if k > 0:
        target_counts[b] = k

if len(target_counts) == 0:
    raise RuntimeError("No overlapping magnitude bins between SF and AGN for matching!")

def one_matched_draw():
    # sample matched SF and matched AGN
    sf_parts = []
    agn_parts = []
    for b, k in target_counts.items():
        sf_b  = df_sf[df_sf["bin"] == b]
        agn_b = df_agn[df_agn["bin"] == b]
        sf_parts.append(sf_b.sample(n=k, replace=False, random_state=int(rng.integers(1e9))))
        agn_parts.append(agn_b.sample(n=k, replace=False, random_state=int(rng.integers(1e9))))
    sf_m  = pd.concat(sf_parts, ignore_index=True)
    agn_m = pd.concat(agn_parts, ignore_index=True)
    return sf_m, agn_m

sf_fracs = []
agn_fracs = []
ns_sf = []
ns_agn = []

for _ in range(N_BOOT):
    sf_m, agn_m = one_matched_draw()
    sf_fracs.append(sf_m["is_var"].mean())
    agn_fracs.append(agn_m["is_var"].mean())
    ns_sf.append(len(sf_m))
    ns_agn.append(len(agn_m))

sf_fracs = np.array(sf_fracs)
agn_fracs = np.array(agn_fracs)

print("\n========== MAG-MATCHED RESULT ==========")
print(f"Matched N (median): SF={int(np.median(ns_sf))}, AGN={int(np.median(ns_agn))}")
print(f"SF  frac_var: mean={sf_fracs.mean():.4f}, std={sf_fracs.std(ddof=1):.4f}")
print(f"AGN frac_var: mean={agn_fracs.mean():.4f}, std={agn_fracs.std(ddof=1):.4f}")
print(f"Delta(AGN-SF): mean={(agn_fracs-sf_fracs).mean():.4f}, std={(agn_fracs-sf_fracs).std(ddof=1):.4f}")

# =========================
# PLOTS (optional)
# =========================
# (A) bin-wise fractions
plt.figure(figsize=(9,5))
plt.plot(m["mag_center"], m["frac_var_SF"], marker="o", label="Star-forming (binned)")
plt.plot(m["mag_center"], m["frac_var_AGN"], marker="o", label="LINER+Seyfert (binned)")
plt.xlabel("Mean magnitude bin center")
plt.ylabel(f"Variability fraction (Fvar ≥ {FVAR_THR:.1f}%)")
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("Figure_matched_binned_fraction.png", dpi=300)
plt.show()

# (B) matched distribution of fractions (bootstrapped)
plt.figure(figsize=(7,5))
plt.hist(sf_fracs, bins=20, alpha=0.7, label="SF matched")
plt.hist(agn_fracs, bins=20, alpha=0.7, label="AGN matched")
plt.xlabel(f"Variability fraction (Fvar ≥ {FVAR_THR:.1f}%)")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("Figure_matched_fraction_bootstrap.png", dpi=300)
plt.show()
