import os
import time
import pandas as pd
from ztfquery import lightcurve

# ======================================================
# 1. 路径设置
# ======================================================

input_csv = r"C:\Users\30126\Desktop\AGN\BPT_analysis_formula_12_13\DESI_SEYFERT_galaxies.csv"
output_dir = r"C:\Users\30126\Desktop\AGN\ZTF_lightcurves\SEYFERT"
os.makedirs(output_dir, exist_ok=True)

# ======================================================
# 2. 读取 DESI 样本
# ======================================================

df = pd.read_csv(input_csv)

ra = df["TARGET_RA"].values
dec = df["TARGET_DEC"].values
targetid = df["TARGETID"].values

print(f"Total targets to process: {len(df)}")

saved = 0
skipped = 0

# ======================================================
# 3. 批量下载 ZTF 光变曲线（ztfquery 1.28.0）
# ======================================================

for i in range(len(df)):

    print(f"\nDownloading TARGETID {targetid[i]}")

    try:
        q = lightcurve.LCQuery.from_position(
            ra=ra[i],
            dec=dec[i],
            radius_arcsec=10
        )

        lc = q.data

        if lc is None or len(lc) == 0:
            print("No ZTF data, skipped.")
            skipped += 1
            continue

        # ---------- 检查测光列 ----------
        if "mag" not in lc.columns or "magerr" not in lc.columns:
            print(f"Missing mag columns: {lc.columns}")
            skipped += 1
            continue

        # ---------- 质量筛选 ----------
        if "catflags" in lc.columns:
            lc = lc[lc["catflags"] == 0]

        if len(lc) < 10:
            print("Too few points (<10), skipped.")
            skipped += 1
            continue

        # ---------- 保存 ----------
        outname = os.path.join(
            output_dir,
            f"{targetid[i]}.csv"
        )

        lc[["mjd", "mag", "magerr", "filtercode"]].to_csv(
            outname,
            index=False
        )

        print(f"Saved: {outname}")
        saved += 1

        time.sleep(1.5)

    except Exception as e:
        print(f"Error: {e}")
        skipped += 1
        time.sleep(3)

# ======================================================
# 4. 总结
# ======================================================

print("\n====================================")
print("ZTF download finished")
print(f"Saved  : {saved}")
print(f"Skipped: {skipped}")
print("====================================")
