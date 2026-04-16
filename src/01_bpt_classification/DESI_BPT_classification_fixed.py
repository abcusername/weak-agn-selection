import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import matplotlib.ticker as ticker

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置图形参数
label_size = 20
params = {
    'legend.fontsize': 16,
    'axes.labelsize': label_size,
    'axes.titlesize': label_size + 2,
    'xtick.labelsize': label_size - 2,
    'ytick.labelsize': label_size - 2,
    'axes.linewidth': 2,
    'ytick.major.size': 6,
    'xtick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 2,
    'ytick.major.width': 2
}
plt.rcParams.update(params)
np.set_printoptions(precision=3)

def safe_log_ratio(numerator, denominator, epsilon=1e-10):
    """
    安全计算log10(numerator/denominator)，避免除零和无效值
    """
    safe_denom = np.maximum(denominator, epsilon)
    safe_num = np.maximum(numerator, epsilon)
    ratio = safe_num / safe_denom
    safe_ratio = np.maximum(ratio, epsilon)
    return np.log10(safe_ratio)

def calculate_additional_ratios(df):
    """
    计算DESI星表中其他的BPT比值
    """
    print("计算BPT诊断图所需比值...")
    
    # 计算[SII]/Hα (用于SII-BPT图)
    if all(col in df.columns for col in ['SII6716_FLUX', 'SII6731_FLUX', 'HALPHA_FLUX']):
        df['lgS2Ha'] = safe_log_ratio(
            df['SII6716_FLUX'] + df['SII6731_FLUX'], 
            df['HALPHA_FLUX']
        )
        print(f"已计算[SII]/Hα比值，有效值: {df['lgS2Ha'].notna().sum()}")
    
    # 计算[OI]/Hα
    if all(col in df.columns for col in ['OI6300_FLUX', 'HALPHA_FLUX']):
        df['lgO1Ha'] = safe_log_ratio(
            df['OI6300_FLUX'], 
            df['HALPHA_FLUX']
        )
        print(f"已计算[OI]/Hα比值，有效值: {df['lgO1Ha'].notna().sum()}")
    
    return df

def check_and_create_snr_columns(df):
    """
    检查并创建SNR列（如果需要）
    """
    print("检查信噪比(SNR)列...")
    
    # 导师指定的SNR列
    required_snr_cols = ['S2_SNR', 'Ha_SNR', 'Hb_SNR', 'O3_SNR']
    missing_snr_cols = [col for col in required_snr_cols if col not in df.columns]
    
    if missing_snr_cols:
        print(f"警告: 缺少SNR列: {missing_snr_cols}")
        print("尝试从FLUX和FLUX_IVAR计算SNR...")
        
        # 假设SNR = FLUX * sqrt(FLUX_IVAR)
        if 'HALPHA_FLUX' in df.columns and 'HALPHA_FLUX_IVAR' in df.columns:
            df['Ha_SNR'] = df['HALPHA_FLUX'] * np.sqrt(df['HALPHA_FLUX_IVAR'])
            print(f"  已计算Ha_SNR，平均值: {df['Ha_SNR'].mean():.2f}")
        
        if 'HBETA_FLUX' in df.columns and 'HBETA_FLUX_IVAR' in df.columns:
            df['Hb_SNR'] = df['HBETA_FLUX'] * np.sqrt(df['HBETA_FLUX_IVAR'])
            print(f"  已计算Hb_SNR，平均值: {df['Hb_SNR'].mean():.2f}")
        
        if 'OIII5007_FLUX' in df.columns and 'OIII5007_FLUX_IVAR' in df.columns:
            df['O3_SNR'] = df['OIII5007_FLUX'] * np.sqrt(df['OIII5007_FLUX_IVAR'])
            print(f"  已计算O3_SNR，平均值: {df['O3_SNR'].mean():.2f}")
        
        if all(col in df.columns for col in ['SII6716_FLUX', 'SII6716_FLUX_IVAR', 'SII6731_FLUX', 'SII6731_FLUX_IVAR']):
            s2_flux = df['SII6716_FLUX'] + df['SII6731_FLUX']
            s2_flux_ivar = 1.0 / (1.0/df['SII6716_FLUX_IVAR'] + 1.0/df['SII6731_FLUX_IVAR'])
            df['S2_SNR'] = s2_flux * np.sqrt(s2_flux_ivar)
            print(f"  已计算S2_SNR，平均值: {df['S2_SNR'].mean():.2f}")
    
    # 检查是否还有缺失，如果有则填充默认值
    for col in required_snr_cols:
        if col not in df.columns:
            print(f"  无法计算{col}，使用默认值100")
            df[col] = 100.0
    
    return df

def classify_agn_using_formula_12_13(df, snr_min=2.0):
    """
    严格根据导师的要求，使用公式(12)和(13)定义LINERs
    对于Seyfert，只需改变公式(13)的不等号方向
    
    公式(12): 0.72/[log([SII]/Hα)-0.32]+1.30 < log([OIII]/Hβ)
    或者: log([SII]/Hα) > 0.32
    
    公式(13): log([OIII]/Hβ) < 1.89×log([SII]/Hα)+0.76 (对于LINER)
    对于Seyfert: log([OIII]/Hβ) > 1.89×log([SII]/Hα)+0.76
    """
    print(f"\n使用公式(12)和(13)进行AGN细分 (SNRmin={snr_min})...")
    print("公式(12): 0.72/[log([SII]/Hα)-0.32]+1.30 < log([OIII]/Hβ) 或 log([SII]/Hα) > 0.32")
    print("公式(13): log([OIII]/Hβ) < 1.89×log([SII]/Hα)+0.76 (LINER)")
    print("Seyfert: 满足公式(12)且 log([OIII]/Hβ) > 1.89×log([SII]/Hα)+0.76")
    
    # 确保必要的列存在
    required_cols = ['lgS2Ha', 'lgO3Hb', 'S2_SNR', 'Ha_SNR', 'Hb_SNR', 'O3_SNR']
    for col in required_cols:
        if col not in df.columns:
            print(f"错误: 缺少必要列 {col}")
            # 尝试创建默认值
            if 'lgS2Ha' not in df.columns and 'lgO3Hb' not in df.columns:
                return df
            if col.endswith('_SNR'):
                df[col] = 100.0
    
    # 首先，所有星系初始化为未分类
    df['BPT_CLASS'] = 'UNCLASSIFIED'
    
    # 1. 先根据[NII]/Hα图进行初步分类
    print("\n步骤1: 基于[NII]/Hα图进行初步分类...")
    
    if 'lgN2Ha' in df.columns and 'lgO3Hb' in df.columns:
        # 计算是否在AGN区域 (公式6/11)
        n2ha = df['lgN2Ha'].values
        o3hb = df['lgO3Hb'].values
        
        # 检查是否在Kewley线上方 (AGN区域)
        mask_above_kewley = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            if n2ha[i] < 0.47:
                kewley_value = 0.61 / (n2ha[i] - 0.47) + 1.19
                mask_above_kewley[i] = o3hb[i] > kewley_value
            else:
                # n2ha >= 0.47，在AGN区域
                mask_above_kewley[i] = True
        
        # 检查是否在Kauffmann线下方 (恒星形成)
        mask_below_kauffmann = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            if n2ha[i] < 0.05:
                kauffmann_value = 0.61 / (n2ha[i] - 0.05) + 1.3
                mask_below_kauffmann[i] = o3hb[i] < kauffmann_value
            else:
                mask_below_kauffmann[i] = False
        
        # 检查是否在复合区域
        mask_composite = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            if n2ha[i] < 0.05:
                kauffmann_value = 0.61 / (n2ha[i] - 0.05) + 1.3
                kewley_value = 0.61 / (n2ha[i] - 0.47) + 1.19
                mask_composite[i] = (o3hb[i] > kauffmann_value) and (o3hb[i] < kewley_value)
            else:
                mask_composite[i] = False
        
        # 分配初步分类
        df.loc[mask_below_kauffmann, 'BPT_CLASS'] = 'STARFORMING'
        df.loc[mask_composite, 'BPT_CLASS'] = 'COMPOSITE'
        df.loc[mask_above_kewley & ~mask_below_kauffmann & ~mask_composite, 'BPT_CLASS'] = 'AGN_REGION'
    
    print(f"  恒星形成星系: {(df['BPT_CLASS'] == 'STARFORMING').sum()}")
    print(f"  复合星系: {(df['BPT_CLASS'] == 'COMPOSITE').sum()}")
    print(f"  AGN区域星系: {(df['BPT_CLASS'] == 'AGN_REGION').sum()}")
    
    # 2. 严格按导师的代码进行Seyfert和LINER分类
    print("\n步骤2: 使用公式(12)和(13)细分AGN为Seyfert和LINER...")
    
    # 获取AGN区域的星系
    agn_mask = df['BPT_CLASS'] == 'AGN_REGION'
    
    if agn_mask.sum() > 0 and 'lgS2Ha' in df.columns:
        # 获取相关数据
        s2_snr = df.loc[agn_mask, 'S2_SNR'].values
        ha_snr = df.loc[agn_mask, 'Ha_SNR'].values
        hb_snr = df.loc[agn_mask, 'Hb_SNR'].values
        o3_snr = df.loc[agn_mask, 'O3_SNR'].values
        lgS2Ha = df.loc[agn_mask, 'lgS2Ha'].values
        lgO3Hb = df.loc[agn_mask, 'lgO3Hb'].values
        
        # 应用导师的代码逻辑
        # 注意：导师代码中有个笔误 072 应该是 0.72
        # 公式(12): (lgO3Hb > 0.72/(lgS2Ha-0.32)+1.3) OR (lgS2Ha > 0.32)
        
        # 先计算公式(12)的条件
        condition_12 = np.zeros(len(lgS2Ha), dtype=bool)
        for i in range(len(lgS2Ha)):
            if lgS2Ha[i] != 0.32:  # 避免除零
                formula12_left = 0.72 / (lgS2Ha[i] - 0.32) + 1.3
                condition_12[i] = (lgO3Hb[i] > formula12_left) or (lgS2Ha[i] > 0.32)
            else:
                condition_12[i] = lgS2Ha[i] > 0.32  # 如果等于0.32，只检查第二部分
        
        # 公式(13)的右侧
        formula13_right = 1.89 * lgS2Ha + 0.76
        
        # Seyfert条件: SNR条件 AND 公式(12) AND lgO3Hb > formula13_right
        idx_seyfert_mask = (
            (s2_snr > snr_min) & 
            (ha_snr > snr_min) & 
            (hb_snr > snr_min) & 
            (o3_snr > snr_min) & 
            condition_12 & 
            (lgO3Hb > formula13_right)
        )
        
        # LINER条件: SNR条件 AND 公式(12) AND lgO3Hb < formula13_right
        idx_liner_mask = (
            (s2_snr > snr_min) & 
            (ha_snr > snr_min) & 
            (hb_snr > snr_min) & 
            (o3_snr > snr_min) & 
            condition_12 & 
            (lgO3Hb < formula13_right)
        )
        
        # 获取原始索引
        agn_indices = df[agn_mask].index
        
        # 应用分类
        seyfert_indices = agn_indices[idx_seyfert_mask]
        liner_indices = agn_indices[idx_liner_mask]
        
        df.loc[seyfert_indices, 'BPT_CLASS'] = 'SEYFERT'
        df.loc[liner_indices, 'BPT_CLASS'] = 'LINER'
        
        print(f"  使用[SII]/Hα诊断:")
        print(f"    分类为Seyfert: {len(seyfert_indices)}")
        print(f"    分类为LINER: {len(liner_indices)}")
        
        # 标记剩余的AGN为未细分AGN
        remaining_agn = agn_indices[~(idx_seyfert_mask | idx_liner_mask)]
        df.loc[remaining_agn, 'BPT_CLASS'] = 'AGN_UNCLASSIFIED'
        print(f"    未细分AGN: {len(remaining_agn)}")
    
    # 3. 最终统计
    print("\n最终分类统计:")
    for cls in ['STARFORMING', 'COMPOSITE', 'SEYFERT', 'LINER', 'AGN_UNCLASSIFIED', 'UNCLASSIFIED']:
        count = (df['BPT_CLASS'] == cls).sum()
        if count > 0:
            percentage = count / len(df) * 100
            print(f"  {cls:20s}: {count:6d} ({percentage:6.2f}%)")
    
    return df

def plot_formula_12_13_diagnostic(df, output_dir):
    """
    绘制专门展示公式(12)和(13)的诊断图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 左侧：[NII]/Hα诊断图
    colors = {
        'STARFORMING': 'blue',
        'COMPOSITE': 'green',
        'SEYFERT': 'red',
        'LINER': 'orange',
        'AGN_UNCLASSIFIED': 'purple',
        'UNCLASSIFIED': 'gray'
    }
    
    for bpt_class in ['STARFORMING', 'COMPOSITE', 'SEYFERT', 'LINER', 'AGN_UNCLASSIFIED']:
        mask = df['BPT_CLASS'] == bpt_class
        if mask.sum() > 0:
            ax1.scatter(df.loc[mask, 'lgN2Ha'], 
                       df.loc[mask, 'lgO3Hb'],
                       c=colors[bpt_class], 
                       label=f'{bpt_class} ({mask.sum()})',
                       alpha=0.6, s=20)
    
    # 绘制分类线
    x_n2 = np.linspace(-2.5, 0.47, 100)
    
    # Kauffmann线 (公式1) - 蓝色虚线
    x_kauffmann = x_n2[x_n2 < 0.05]
    y_kauffmann = 0.61 / (x_kauffmann - 0.05) + 1.3
    ax1.plot(x_kauffmann, y_kauffmann, 'b--', linewidth=2, label='Ka03 (Eq. 1)')
    
    # Kewley线 (公式5/11) - 红色实线
    x_kewley = x_n2[x_n2 < 0.47]
    y_kewley = 0.61 / (x_kewley - 0.47) + 1.19
    ax1.plot(x_kewley, y_kewley, 'r-', linewidth=2, label='Ke01 (Eq. 5/11)')
    
    ax1.set_xlabel(r'$\log\,\mathrm{[N\,II]/H\alpha}$', fontsize=16)
    ax1.set_ylabel(r'$\log\,\mathrm{[O\,III]/H\beta}$', fontsize=16)
    ax1.set_title('BPT诊断图: [NII]/Hα', fontsize=16)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-2.5, 1.0])
    ax1.set_ylim([-2.5, 1.5])
    
    # 右侧：[SII]/Hα诊断图（重点展示公式12和13）
    if 'lgS2Ha' in df.columns:
        for bpt_class in ['STARFORMING', 'COMPOSITE', 'SEYFERT', 'LINER', 'AGN_UNCLASSIFIED']:
            mask = (df['BPT_CLASS'] == bpt_class) & df['lgS2Ha'].notna()
            if mask.sum() > 0:
                ax2.scatter(df.loc[mask, 'lgS2Ha'], 
                           df.loc[mask, 'lgO3Hb'],
                           c=colors[bpt_class], 
                           label=f'{bpt_class} ({mask.sum()})',
                           alpha=0.6, s=20)
        
        # 绘制公式(12)和(13)的线
        x_s2 = np.linspace(-1.5, 0.5, 100)
        
        # 公式(12): 0.72/[log([SII]/Hα)-0.32]+1.30
        x_s2_safe = x_s2[x_s2 != 0.32]
        y_formula12 = 0.72 / (x_s2_safe - 0.32) + 1.30
        ax2.plot(x_s2_safe, y_formula12, 'k-', linewidth=3, label='Eq. 12: 0.72/(x-0.32)+1.30')
        
        # 公式(13): 1.89×log([SII]/Hα)+0.76
        y_formula13 = 1.89 * x_s2 + 0.76
        ax2.plot(x_s2, y_formula13, 'k--', linewidth=3, label='Eq. 13: 1.89x+0.76')
        
        # 垂直虚线 x=0.32
        ax2.axvline(x=0.32, color='gray', linestyle=':', linewidth=2, label='x=0.32')
        
        # 标记LINER区域（满足公式12且满足公式13）
        intersection_mask = (x_s2 > 0.32) & (x_s2 < 0.5)
        if np.any(intersection_mask):
            x_fill = x_s2[intersection_mask]
            y_12_fill = 0.72 / (x_fill - 0.32) + 1.30
            y_13_fill = 1.89 * x_fill + 0.76
            
            ax2.fill_between(x_fill, y_12_fill, y_13_fill, 
                           where=(y_12_fill < y_13_fill), 
                           color='orange', alpha=0.3, label='LINER region')
        
        ax2.set_xlabel(r'$\log\,\mathrm{[S\,II]/H\alpha}$', fontsize=16)
        ax2.set_ylabel(r'$\log\,\mathrm{[O\,III]/H\beta}$', fontsize=16)
        ax2.set_title('BPT诊断图: [SII]/Hα', fontsize=16)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-1.5, 0.5])
        ax2.set_ylim([-1.0, 1.5])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'BPT_formula_12_13_diagnostic.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"公式(12)和(13)诊断图已保存至: {output_path}")
    plt.show()
    plt.close()

def verify_classification_logic(df, snr_min=2.0):
    """
    验证分类逻辑，确保与导师的要求一致
    """
    print("\n" + "="*60)
    print("验证分类逻辑")
    print("="*60)
    
    # 检查Seyfert和LINER的分类逻辑
    seyfert_mask = df['BPT_CLASS'] == 'SEYFERT'
    liner_mask = df['BPT_CLASS'] == 'LINER'
    
    print(f"Seyfert星系数量: {seyfert_mask.sum()}")
    print(f"LINER星系数量: {liner_mask.sum()}")
    
    # 验证几个样本是否符合公式
    if seyfert_mask.sum() > 0:
        print("\n随机检查5个Seyfert星系:")
        seyfert_sample = df[seyfert_mask].sample(min(5, seyfert_mask.sum()))
        for idx, row in seyfert_sample.iterrows():
            s2ha = row['lgS2Ha']
            o3hb = row['lgO3Hb']
            
            # 计算公式(12)条件
            if s2ha != 0.32:
                formula12_left = 0.72 / (s2ha - 0.32) + 1.3
                condition_12 = (o3hb > formula12_left) or (s2ha > 0.32)
            else:
                condition_12 = s2ha > 0.32
            
            # 计算公式(13)右侧
            formula13_right = 1.89 * s2ha + 0.76
            
            print(f"  ID {row['TARGETID'] if 'TARGETID' in row else idx}:")
            print(f"    log([SII]/Hα) = {s2ha:.3f}")
            print(f"    log([OIII]/Hβ) = {o3hb:.3f}")
            print(f"    公式(12)条件满足: {condition_12}")
            print(f"    公式(13)右侧: {formula13_right:.3f}")
            print(f"    o3hb > formula13_right: {o3hb > formula13_right}")
            print()
    
    if liner_mask.sum() > 0:
        print("\n随机检查5个LINER星系:")
        liner_sample = df[liner_mask].sample(min(5, liner_mask.sum()))
        for idx, row in liner_sample.iterrows():
            s2ha = row['lgS2Ha']
            o3hb = row['lgO3Hb']
            
            # 计算公式(12)条件
            if s2ha != 0.32:
                formula12_left = 0.72 / (s2ha - 0.32) + 1.3
                condition_12 = (o3hb > formula12_left) or (s2ha > 0.32)
            else:
                condition_12 = s2ha > 0.32
            
            # 计算公式(13)右侧
            formula13_right = 1.89 * s2ha + 0.76
            
            print(f"  ID {row['TARGETID'] if 'TARGETID' in row else idx}:")
            print(f"    log([SII]/Hα) = {s2ha:.3f}")
            print(f"    log([OIII]/Hβ) = {o3hb:.3f}")
            print(f"    公式(12)条件满足: {condition_12}")
            print(f"    公式(13)右侧: {formula13_right:.3f}")
            print(f"    o3hb < formula13_right: {o3hb < formula13_right}")
            print()

def main():
    """
    主函数
    """
    # 设置路径
    data_path = r"C:\Users\30126\Desktop\AGN\DESI_mass_emline_z0p05_cat.csv"
    output_dir = r"C:\Users\30126\Desktop\AGN\BPT_analysis_formula_12_13"
    
    print("正在加载DESI星表...")
    df = pd.read_csv(data_path)
    print(f"成功加载 {len(df)} 个星系")
    
    # 检查必要的列是否存在
    required_cols = ['lgN2Ha', 'lgO3Hb']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"警告: 缺少以下必要的列: {missing_cols}")
        return
    
    # 计算其他比值
    df = calculate_additional_ratios(df)
    
    # 检查并创建SNR列
    df = check_and_create_snr_columns(df)
    
    # 使用公式(12)和(13)进行分类
    df = classify_agn_using_formula_12_13(df, snr_min=2.0)
    
    # 绘制诊断图
    print("\n绘制公式(12)和(13)诊断图...")
    plot_formula_12_13_diagnostic(df, output_dir)
    
    # 验证分类逻辑
    verify_classification_logic(df, snr_min=2.0)
    
    # 最终统计
    print("\n" + "="*60)
    print("最终分类统计")
    print("="*60)
    
    total_counts = df['BPT_CLASS'].value_counts()
    for cls, count in total_counts.items():
        percentage = count / len(df) * 100
        print(f"  {cls:20s}: {count:6d} ({percentage:6.2f}%)")
    
    # 保存结果
    result_file = os.path.join(output_dir, 'DESI_BPT_classification_formula_12_13.csv')
    df.to_csv(result_file, index=False)
    print(f"\n分类结果已保存至: {result_file}")
    
    # 保存各类星系的子集
    for cls in ['STARFORMING', 'COMPOSITE', 'SEYFERT', 'LINER', 'AGN_UNCLASSIFIED']:
        cls_df = df[df['BPT_CLASS'] == cls]
        if len(cls_df) > 0:
            cls_file = os.path.join(output_dir, f'DESI_{cls}_galaxies.csv')
            cls_df.to_csv(cls_file, index=False)
            print(f"  {cls}星系列表已保存至: {cls_file}")
    
    print(f"\n所有结果已保存至: {output_dir}")
    
    return df

if __name__ == '__main__':
    df = main()
