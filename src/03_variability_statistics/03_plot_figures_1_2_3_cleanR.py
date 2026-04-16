# 03_plot_figures_1_2_3_cleanR_MINFIX.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Paths (MINFIX)
# =========================
BASE = r"C:\Users\30126\Desktop\AGN"
CSV = os.path.join(BASE, "AGN_variability_statistics_v5_cleanR_MINFIX.csv")
OUTDIR = os.path.join(BASE, "FIGS_CLEAN_R_MINFIX")
os.makedirs(OUTDIR, exist_ok=True)

print("=" * 70)
print("Plot figures from cleanR MINFIX stats")
print("CSV    :", CSV)
print("OUTDIR :", OUTDIR)
print("=" * 70)

df = pd.read_csv(CSV)

# =========================
# Clean / normalize
# =========================
num_cols = ["Fvar_percent", "N", "Mean_mag", "Time_span"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Class"] = df["Class"].astype(str)

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Class"] + num_cols)

order = ["Seyfert", "Composite", "LINER", "Star-forming"]

def norm_class(s):
    s = str(s).strip().lower()
    if s == "liner":
        return "LINER"
    if s == "seyfert":
        return "Seyfert"
    if s == "composite":
        return "Composite"
    if s in ["star-forming", "starforming", "star forming"]:
        return "Star-forming"
    # fallback
    return s.title()

df["Class"] = df["Class"].apply(norm_class)
df = df[df["Class"].isin(order)].copy()

print("\nClass counts:")
print(df["Class"].value_counts().reindex(order))

# =========================
# Figure 1: variability fraction bar
# =========================
thr = 5.0  # %
# 用 groupby().mean() 避免 pandas 的 include_groups 参数差异
g = (
    df.assign(is_var=(df["Fvar_percent"] >= thr).astype(float))
      .groupby("Class")["is_var"].mean()
      .reindex(order)
)

plt.figure(figsize=(6.5, 4.2))
plt.bar(g.index, g.values)
plt.ylabel(f"Variability fraction ($F_{{\\rm var}} \\geq {thr:.0f}\\%$)")
plt.ylim(0, 1)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

f1 = os.path.join(OUTDIR, "Figure1_variability_fraction_bar_cleanR_MINFIX.png")
plt.savefig(f1, dpi=300)
plt.close()

# =========================
# Figure 2: Fvar distribution (step hist)
# =========================
plt.figure(figsize=(7.2, 4.6))
bins = np.arange(0, 30.5, 0.5)  # MINFIX 后可能分布更正常，给到 30% 更稳妥

for cls in order:
    x = df.loc[df["Class"] == cls, "Fvar_percent"].values
    x = x[np.isfinite(x)]
    x = x[(x >= 0) & (x <= 30)]
    if len(x) == 0:
        continue
    plt.hist(x, bins=bins, density=True, histtype="step", linewidth=2, label=cls)

plt.xlabel(r"Fractional Variability $F_{\rm var}$ (%)")
plt.ylabel("Normalized density")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

f2 = os.path.join(OUTDIR, "Figure2_Fvar_distribution_cleanR_MINFIX.png")
plt.savefig(f2, dpi=300)
plt.close()

# =========================
# Figure 3: Fvar vs properties
# =========================
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)

for cls in order:
    sub = df[df["Class"] == cls].copy()
    if len(sub) == 0:
        continue

    # display cap (只影响画图，不影响统计表)
    sub = sub[(sub["Fvar_percent"] >= 0) & (sub["Fvar_percent"] <= 30)]

    axes[0].scatter(sub["Time_span"], sub["Fvar_percent"], s=10, alpha=0.55, label=cls)
    axes[1].scatter(sub["N"], sub["Fvar_percent"], s=10, alpha=0.55, label=cls)
    axes[2].scatter(sub["Mean_mag"], sub["Fvar_percent"], s=10, alpha=0.55, label=cls)

axes[0].set_title("(a) $F_{var}$ vs Time span")
axes[1].set_title("(b) $F_{var}$ vs N")
axes[2].set_title("(c) $F_{var}$ vs Mean magnitude")

axes[0].set_xlabel("Time span (days)")
axes[1].set_xlabel("Number of epochs (N)")
axes[2].set_xlabel("Mean magnitude")
axes[0].set_ylabel(r"$F_{var}$ (%)")

for ax in axes:
    ax.axhline(thr, ls="--", lw=1)
    ax.grid(True, alpha=0.25)

# make mag axis astronomical (bright on left)
axes[2].invert_xaxis()

axes[0].legend(frameon=True, loc="upper left")
plt.tight_layout()

f3 = os.path.join(OUTDIR, "Figure3_Fvar_vs_properties_cleanR_MINFIX.png")
plt.savefig(f3, dpi=300)
plt.close()

print("\nSaved:")
print(f1)
print(f2)
print(f3)
print("=" * 70)
print("Done ✅")
print("=" * 70)
