import os
import numpy as np
import pandas as pd

# ==============================
# User settings (改这里就行)
# ==============================
# 你的统计总表（包含 Class, Mean_mag 的那个 csv）
# 例如：C:\Users\30126\Desktop\AGN\AGN_variability_statistics_v5_cleanR_MINFIX.csv
CSV_PATH = r"C:\Users\30126\Desktop\AGN\AGN_variability_statistics_v5_cleanR_MINFIX.csv"

# Mean_mag > 多少算“更暗”
MAG_THRESHOLD = 19.0

# ==============================
# Helper
# ==============================
def pick_column(df: pd.DataFrame, candidates):
    """从候选列名中选一个存在的列"""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_class(s: str) -> str:
    """把 Class 统一成简洁标签"""
    if s is None:
        return ""
    x = str(s).strip().lower()
    # 你数据里常见的几种写法都兼容
    if "star" in x:
        return "star-forming"
    if "sey" in x:
        return "seyfert"
    if "liner" in x:
        return "liner"
    if "comp" in x:
        return "composite"
    return x

def summarize(group_name: str, arr: np.ndarray, mag_threshold: float):
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        print(f"\n[{group_name}] No valid Mean_mag values.")
        return
    med = np.median(arr)
    frac_dark = np.mean(arr > mag_threshold)
    print(f"\n[{group_name}]")
    print(f"  N(valid Mean_mag) = {n}")
    print(f"  Median Mean_mag   = {med:.3f}")
    print(f"  Fraction Mean_mag > {mag_threshold:.1f} = {frac_dark:.4f} ({frac_dark*100:.2f}%)")

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # 兼容不同版本列名
    class_col = pick_column(df, ["Class", "class", "BPT_Class", "bpt_class"])
    meanmag_col = pick_column(df, ["Mean_mag", "MeanMag", "mean_mag", "meanmag", "mean_mag_r"])

    if class_col is None:
        raise KeyError(f"Cannot find class column. Existing columns: {list(df.columns)}")
    if meanmag_col is None:
        raise KeyError(f"Cannot find Mean_mag column. Existing columns: {list(df.columns)}")

    # 规范化
    df["_class_norm"] = df[class_col].apply(normalize_class)
    df["_meanmag"] = pd.to_numeric(df[meanmag_col], errors="coerce")

    # 仅保留需要的两组
    sf = df[df["_class_norm"] == "star-forming"].copy()
    agn = df[df["_class_norm"].isin(["liner", "seyfert"])].copy()

    # 打印总体信息（防止你误把别的类别混进来）
    print("========== BASIC CHECK ==========")
    print(f"CSV: {CSV_PATH}")
    print(f"Total rows: {len(df)}")
    print("Class counts (normalized):")
    print(df["_class_norm"].value_counts(dropna=False))
    print("=================================")

    # 缺失情况
    print("\nMean_mag missing rate:")
    print(f"  Star-forming : {sf['_meanmag'].isna().mean():.4f}  (N={len(sf)})")
    print(f"  LINER+Seyfert: {agn['_meanmag'].isna().mean():.4f}  (N={len(agn)})")

    # 统计
    summarize("Star-forming", sf["_meanmag"].to_numpy(), MAG_THRESHOLD)
    summarize("LINER + Seyfert", agn["_meanmag"].to_numpy(), MAG_THRESHOLD)

    # 给一个很直观的“是否明显更暗”的提示（非结论，只是帮助你快速判断）
    sf_med = np.nanmedian(sf["_meanmag"].to_numpy()) if len(sf) else np.nan
    agn_med = np.nanmedian(agn["_meanmag"].to_numpy()) if len(agn) else np.nan
    if np.isfinite(sf_med) and np.isfinite(agn_med):
        diff = sf_med - agn_med
        print("\n========== QUICK DIAG ==========")
        print(f"Median(Star-forming) - Median(LINER+Seyfert) = {diff:.3f} mag")
        if diff > 0.3:
            print("⚠️ Star-forming is notably fainter (median higher). Strong risk of noise-driven Fvar differences.")
            print("   => 建议先做 magnitude-matched（或在 Mean_mag 分箱后比较）再发导师。")
        else:
            print("✅ Medians are close (<=0.3 mag). Magnitude bias is less severe (still check tails).")
        print("================================")

