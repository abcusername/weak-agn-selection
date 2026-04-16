import os
import numpy as np
import pandas as pd

# ====== path ======
BASE = r"C:\Users\30126\Desktop\AGN"
TID = 39627790923863707
CSV = os.path.join(BASE, "TO_ADVISOR_CHECK_MINFIX",
                   "lightcurves_cleanR_MINFIX", "SEYFERT_examples",
                   f"{TID}_advisor.csv")

# ====== utils ======
def mag_to_flux(mag):
    return 10 ** (-0.4 * mag)

def magerr_to_fluxerr(mag, magerr):
    f = mag_to_flux(mag)
    return f * (np.log(10) * 0.4) * magerr

df = pd.read_csv(CSV)

# 只保留必要列（存在就用，不存在就跳过）
cols = ["mjd","mag","magerr","catflags","limitmag","ra","dec","oid"]
for c in cols:
    if c not in df.columns:
        df[c] = np.nan

# 基本清洗
m = np.isfinite(df["mjd"]) & np.isfinite(df["mag"]) & np.isfinite(df["magerr"]) & (df["magerr"] > 0)
d = df.loc[m].copy()

# 计算 flux 残差
f  = mag_to_flux(d["mag"].values)
fe = magerr_to_fluxerr(d["mag"].values, d["magerr"].values)

w = 1.0 / (fe**2)
fbar = np.sum(w * f) / np.sum(w)

r = (f - fbar) / fe   # normalized residuals in flux space

d["flux"] = f
d["fluxerr"] = fe
d["r_norm"] = r
d["abs_r"] = np.abs(r)
d["night"] = np.floor(d["mjd"]).astype(int)

# ====== 1) 输出最离谱点 ======
top = d.sort_values("abs_r", ascending=False).head(30)
out_top = os.path.join(os.path.dirname(CSV), f"{TID}_TOP30_outliers.csv")
top.to_csv(out_top, index=False)

print("\n=== TOP 30 outliers by |r| saved ===")
print(out_top)
print(top[["mjd","mag","magerr","r_norm","catflags","limitmag","night"]].head(15))

# ====== 2) night 级别诊断：是否“坏夜成片” ======
g = d.groupby("night").agg(
    n=("mag","size"),
    med_r=("r_norm","median"),
    mad_r=("r_norm", lambda x: np.median(np.abs(x - np.median(x))) ),
    max_abs_r=("abs_r","max"),
    med_mag=("mag","median"),
    med_magerr=("magerr","median"),
    frac_catflags_nonzero=("catflags", lambda x: np.mean(np.array(x)!=0) if np.any(pd.notna(x)) else np.nan)
).reset_index()

# 给一个简单的“坏夜评分”：max_abs_r 优先，其次 mad_r
g["score"] = g["max_abs_r"] + 0.5 * g["mad_r"]
g = g.sort_values("score", ascending=False)

out_night = os.path.join(os.path.dirname(CSV), f"{TID}_night_diagnostics.csv")
g.to_csv(out_night, index=False)

print("\n=== Night diagnostics saved ===")
print(out_night)
print("\nTop suspicious nights:")
print(g.head(10)[["night","n","max_abs_r","mad_r","med_r","med_mag","med_magerr","frac_catflags_nonzero"]])

# ====== 3) 结论提示（简单规则）=====
# 如果 top 10 outliers 有 >=5 个来自同一 night -> 很像“坏夜/系统”
top10 = d.sort_values("abs_r", ascending=False).head(10)
most_night = top10["night"].value_counts().head(1)
night_id = int(most_night.index[0])
cnt = int(most_night.iloc[0])

print("\n=== Quick hint ===")
print(f"Top10 outliers: most frequent night = {night_id}, count = {cnt}/10")
if cnt >= 5:
    print("=> Strong sign of night-level systematics (bad night / segment bias).")
else:
    print("=> Outliers are scattered; could be a few isolated bad points or real events.")
