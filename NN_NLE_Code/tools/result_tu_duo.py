import matplotlib.pyplot as plt
import numpy as np
import os
import re

# ==============================================================================
# 1. 全局配置区域 (在此定义你要把哪几组实验放在一张图里对比)
# ==============================================================================

# [基础路径]
base_root = r'D:\paperwork\Experiment_Data'

# [对比列表]: 请填入文件夹名称 (也就是提取脚本中的 target_scenario)
# 脚本会依次读取这些文件夹下的 results_xxx.md 文件
scenarios_list = [
    '5Gsyms_10km',  # 对应实线 (-)
    '10Gsyms_10km',  # 对应虚线 (--)
    '20Gsyms_10km'  # 对应点划线 (-.)
]

# [线型分配]: 对应上面的列表顺序
scenario_linestyles = ['-', '--', '-.', ':']

# [保存位置]: 图片保存的文件夹 (建议放在一个汇总文件夹)
save_dir = r'D:\paperwork\Experiment_Data\Comparison_Results'
output_file_name = '10km.png'

# [参数范围]: ROP (横坐标)
rop = np.arange(-27, -14, 1)

# [绘图设置]: 字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 11})


# ==============================================================================
# 2. 多场景数据自动加载引擎
# ==============================================================================

def load_all_scenarios(base_path, scenarios):
    """
    遍历场景列表，读取每个场景的 .md 文件，并将数据合并到一个大字典中。
    变量名会自动重命名为: 原名_场景名 (例如: sqnr_dnn_5Gsyms_20km)
    """
    master_data = {}
    print("开始加载多组实验数据...")

    for sc_name in scenarios:
        # 1. 构建路径: D:\...\5Gsyms_20km\results_5Gsyms_20km.md
        file_path = os.path.join(base_path, sc_name, f'results_{sc_name}.md')

        if not os.path.exists(file_path):
            print(f"  [Error] 缺失文件: {file_path}")
            print(f"  -> 请先针对 [{sc_name}] 运行 indicator_extraction.py 脚本！")
            continue  # 跳过该文件

        print(f"  ->正在读取: {sc_name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 2. 正则解析
        pattern = r"(\w+)\s*=\s*np\.array\(\[\s*(.*?)\s*\]\)"
        matches = re.findall(pattern, content, re.DOTALL)

        count = 0
        for var_name, values_str in matches:
            try:
                # 转为 numpy 数组
                vals = [float(v) for v in values_str.replace('\n', '').split(',')]

                # 【关键步骤】: 重命名变量，加上后缀
                # sqnr_dnn -> sqnr_dnn_5Gsyms_20km
                unique_key = f"{var_name}_{sc_name}"
                master_data[unique_key] = np.array(vals)
                count += 1
            except:
                pass

        # print(f"    提取了 {count} 个变量")

    return master_data


# 执行加载
all_data = load_all_scenarios(base_root, scenarios_list)


# 辅助函数: 安全获取数据
def get_data(prefix, model, scenario):
    key = f"{prefix}_{model}_{scenario}"
    if key in all_data:
        return all_data[key]
    else:
        # 如果数据缺失，返回全0数组防止报错
        # print(f"  [Warning] 缺少数据: {key}")
        return np.zeros_like(rop)


# ==============================================================================
# 3. 绘图逻辑
# ==============================================================================
if not all_data:
    print("错误: 没有加载到任何数据，请检查路径。")
    exit()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 模型颜色配置 (Model -> Color/Marker)
model_styles = {
    'dnn': {'c': 'blue', 'm': 'o', 'name': 'DNN-NLE'},
    'volterra': {'c': 'red', 'm': 's', 'name': 'Volterra'},
    'dfe': {'c': 'green', 'm': '^', 'name': 'DFE'}
}

print("正在绘图...")

# 双层循环: 外层循环场景(线型)，内层循环模型(颜色)
for i, sc_name in enumerate(scenarios_list):
    # 以此决定线型: 第一个场景用 '-', 第二个用 '--'
    line_style = scenario_linestyles[i % len(scenario_linestyles)]

    # 场景显示的后缀 (用于图例)
    # 提取 "5Gsyms" 这样的短标识可能更好看，这里直接用文件夹名
    label_suffix = sc_name

    # --- 绘制 SQNR (左图) ---
    for model in ['dnn', 'volterra', 'dfe']:
        y = get_data('sqnr', model, sc_name)
        s = model_styles[model]

        # 仅在第一组数据时打印完整图例，或者每组都打印(会导致图例很长)
        # 这里采用: "DNN-NLE (5Gsyms_20km)" 的格式
        label_str = f"{s['name']} ({label_suffix})"

        ax1.plot(rop, y, color=s['c'], marker=s['m'], linestyle=line_style,
                 linewidth=2, markersize=6, label=label_str)

    # --- 绘制 EVM (右图) ---
    for model in ['dnn', 'volterra', 'dfe']:
        y = get_data('evm', model, sc_name)
        s = model_styles[model]
        label_str = f"{s['name']} ({label_suffix})"

        ax2.plot(rop, y, color=s['c'], marker=s['m'], linestyle=line_style,
                 linewidth=2, markersize=6, label=label_str)

# ==============================================================================
# 4. 图表美化与保存
# ==============================================================================

# SQNR 设置
ax1.set_title('PCM SQNR Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Received Optical Power (dBm)', fontsize=12)
ax1.set_ylabel('PCM SQNR (dB)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_xticks(rop)
# 图例分列显示，避免太长遮挡
ax1.legend(loc='upper left', ncol=2, fontsize=9)

# EVM 设置
ax2.set_title('rms EVM Performance Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Received Optical Power (dBm)', fontsize=12)
ax2.set_ylabel('rms EVM (%)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_xticks(rop)
ax2.axhline(y=12.5, color='black', linestyle='--', linewidth=1.5, label='16QAM Limit')
ax2.legend(loc='upper right', ncol=2, fontsize=9)

plt.tight_layout()

# 确保保存目录存在
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except:
        pass

full_save_path = os.path.join(save_dir, output_file_name)
plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
print(f"处理完成！高清对比图已保存至: {full_save_path}")
plt.show()