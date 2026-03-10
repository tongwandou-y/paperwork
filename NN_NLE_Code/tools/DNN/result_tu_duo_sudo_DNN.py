import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import re

# ==============================================================================
# 1. 全局配置区域
# ==============================================================================

base_root = r'D:\paperwork\Experiment_Data_仿真'

# [对比列表]：不同速率 (20km固定)
scenarios_list = [
    '10Gsyms_20km',
    '20Gsyms_20km',
    # '5Gsyms_20km'
]

# [保存位置]
save_dir = r'D:\paperwork\Experiment_Data_仿真\Comparison_Results'

# [文件名接口]
output_filename_sqnr = 'PCM_SQNR_不同速率_20km.png'
output_filename_evm = 'rms_EVM_不同速率_20km.png'
output_filename_ber = 'BER_PAM4_不同速率_20km.png'

# [参数范围]：您可以随意调整这个范围，只要在 -27 到 -15 之间即可
# 例如：改为 -22 到 -15
rop = np.arange(-27, -14, 1)

# [关键常量]：定义原始数据文件(.md)中数据的起始点
# 您提到 .md 文件中数据是从 -27dBm 开始的
DATA_START_DBM = -27

# [绘图设置]
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 11})


# ==============================================================================
# 2. 数据加载引擎
# ==============================================================================
def load_all_scenarios(base_path, scenarios):
    master_data = {}
    print("开始加载多组实验数据...")
    for sc_name in scenarios:
        file_path = os.path.join(base_path, sc_name, f'results_{sc_name}.md')
        if not os.path.exists(file_path):
            print(f"  [Error] 缺失文件: {file_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用正则提取数组内容
        pattern = r"(\w+)\s*=\s*np\.array\(\[\s*(.*?)\s*\]\)"
        matches = re.findall(pattern, content, re.DOTALL)

        for var_name, values_str in matches:
            try:
                # 简单清洗数据
                cleaned_values = values_str.replace("'", "").replace('"', "")
                # 尝试转换为浮点数
                vals = [float(v) for v in cleaned_values.replace('\n', '').split(',')]
                unique_key = f"{var_name}_{sc_name}"
                master_data[unique_key] = np.array(vals)
            except:
                pass
    return master_data


all_data = load_all_scenarios(base_root, scenarios_list)


def get_data(prefix, model, scenario):
    """
    获取数据并根据当前的 rop 范围进行切片
    """
    key = f"{prefix}_{model}_{scenario}"
    full_data = all_data.get(key)

    if full_data is None:
        # 如果没有数据，返回对应长度的 0 数组
        return np.zeros(len(rop), dtype=float)

    # --- [核心修复] 数据对齐逻辑 ---
    # 计算 rop[0] (比如 -22) 相对于 DATA_START_DBM (-27) 的偏移量
    # -22 - (-27) = 5，说明我们要从数组的第 5 个索引开始取
    start_index = int(rop[0] - DATA_START_DBM)
    end_index = start_index + len(rop)

    # 边界检查，防止索引越界
    if start_index < 0 or end_index > len(full_data):
        print(f"[Warning] 请求的 ROP 范围 {rop[0]}~{rop[-1]}dBm 超出了数据文件范围!")
        # 降级处理：返回全0
        return np.zeros(len(rop), dtype=float)

    # 返回切片后的数据，使其长度与 rop 一致
    return full_data[start_index: end_index]


if not all_data:
    print("错误: 没有加载到数据，请检查路径。")
    exit()


# ==============================================================================
# 3. 差值计算与标注函数
# ==============================================================================

def annotate_max_diff(ax, metric_prefix, unit_label):
    """
    在图中标注 DNN 和 Volterra 的最大差值
    """
    print(f"\n{'=' * 20} {metric_prefix.upper()} 差值详情 {'=' * 20}")

    offset_map = [0, 2, -2]

    for i, sc_name in enumerate(scenarios_list):
        y_dnn = get_data(metric_prefix, 'dnn', sc_name)
        y_vol = get_data(metric_prefix, 'volterra', sc_name)

        # 计算绝对差值
        diffs = np.abs(y_dnn - y_vol)
        max_idx = np.argmax(diffs)
        max_diff = diffs[max_idx]
        target_rop = rop[max_idx]
        y_d_max = y_dnn[max_idx]
        y_v_max = y_vol[max_idx]

        # 终端打印
        print(f"场景: {sc_name} | Max Diff: {max_diff:.4f} {unit_label} @ {target_rop} dBm")

        # 绘图: 双向箭头
        y_bottom = min(y_d_max, y_v_max)
        y_top = max(y_d_max, y_v_max)

        ax.annotate(
            '', xy=(target_rop, y_bottom), xytext=(target_rop, y_top),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5, shrinkA=0, shrinkB=0)
        )

        # 绘图: 虚线引出
        text_x_pos = target_rop + 1.2
        # 动态调整文本位置，防止画出图外
        if text_x_pos > rop[-1]:
            text_x_pos = target_rop - 1.2

        mid_y = (y_bottom + y_top) / 2
        ax.plot([target_rop, text_x_pos], [mid_y, mid_y], color='black', linestyle='--', linewidth=1)

        # 绘图: 文本
        text_y_pos = mid_y + offset_map[i % 3]
        label_text = f"~{max_diff:.1f}{unit_label}"
        ax.text(text_x_pos, text_y_pos, label_text, color='black',
                fontsize=11, fontweight='bold', va='center')


# ==============================================================================
# 4. 样式配置 (修复版)
# ==============================================================================

model_styles = {
    'dnn': {'c': 'blue', 'm': 'o', 'name': 'DNN-NLE'},
    'volterra': {'c': 'red', 'm': 's', 'name': 'Volterra'},
    'dfe': {'c': 'green', 'm': '^', 'name': 'DFE'}
}

# 使用字典将样式直接绑定到具体的场景名称上，避免索引错位
scenario_styles = {
    '5Gsyms_20km': {'ls': '-', 'mfc': 'auto', 'label': '5Gsyms'},
    '10Gsyms_20km': {'ls': '--', 'mfc': 'white', 'label': '10Gsyms'},
    '20Gsyms_20km': {'ls': ':', 'mfc': 'auto', 'label': '20Gsyms'}
}


def plot_lines_on_ax(ax, data_prefix, use_log10=False):
    """
    通用绘图函数
    """
    for sc_name in scenarios_list:
        # 直接通过场景名称获取样式，不再依赖 i % 3
        style_cfg = scenario_styles.get(sc_name)
        if not style_cfg:
            continue

        ls = style_cfg['ls']
        mfc = style_cfg['mfc']

        for model in ['dnn', 'volterra', 'dfe']:
            y = get_data(data_prefix, model, sc_name)

            if use_log10:
                safe_y = np.where(y > 0, y, 1e-9)
                plot_y = np.log10(safe_y)
            else:
                plot_y = y

            ms = model_styles[model]
            ax.plot(rop, plot_y,
                    color=ms['c'],
                    marker=ms['m'],
                    linestyle=ls,
                    linewidth=1.8,
                    markersize=6,
                    markerfacecolor=ms['c'] if mfc == 'auto' else 'white',
                    markeredgewidth=1.5
                    )


def create_custom_legend(is_evm=False):
    # 1. 准备左列元素 (固定为3个模型)
    col1 = [
        Line2D([0], [0], color='blue', marker='o', lw=0, label='DNN-NLE'),
        Line2D([0], [0], color='red', marker='s', lw=0, label='Volterra'),
        Line2D([0], [0], color='green', marker='^', lw=0, label='DFE')
    ]

    # 2. 准备右列元素 (动态数量的速率 + 可选的Limit)
    col2 = []
    for sc_name in scenarios_list:
        cfg = scenario_styles[sc_name]
        face_color = 'black' if cfg['mfc'] == 'auto' else 'white'
        col2.append(
            Line2D([0], [0], color='black', marker='o', linestyle=cfg['ls'],
                   markerfacecolor=face_color, label=cfg['label'])
        )

    if is_evm:
        col2.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='16QAM Limit'))

    # 3. 智能对齐两列长度 (用透明占位符填补短的一边到底部)
    max_len = max(len(col1), len(col2))
    blank = Line2D([0], [0], color='none', marker='None', label=' ')  # 完全隐形的占位符

    while len(col1) < max_len:
        col1.append(blank)
    while len(col2) < max_len:
        col2.append(blank)

    # 4. 合并输出 (matplotlib 会自动将前半部分放在左列，后半部分放在右列)
    return col1 + col2

# 确保保存目录存在
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except:
        pass

# ==============================================================================
# 5. 绘制第一张图: PCM SQNR
# ==============================================================================
print("正在绘制 SQNR 图...")
plt.figure(figsize=(9, 6))
ax1 = plt.gca()

plot_lines_on_ax(ax1, 'sqnr', use_log10=False)
annotate_max_diff(ax1, 'sqnr', 'dB')

ax1.set_title('PCM SQNR Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Received Optical Power (dBm)', fontsize=12)
ax1.set_ylabel('PCM SQNR (dB)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_xticks(rop)
ax1.legend(handles=create_custom_legend(is_evm=False), loc='upper left', ncol=2, fontsize=10, frameon=True)

save_path_sqnr = os.path.join(save_dir, output_filename_sqnr)
plt.tight_layout()
plt.savefig(save_path_sqnr, dpi=300, bbox_inches='tight')
print(f"SQNR图已保存: {save_path_sqnr}")
plt.close()

# ==============================================================================
# 6. 绘制第二张图: rms EVM
# ==============================================================================
print("正在绘制 EVM 图...")
plt.figure(figsize=(9, 6))
ax2 = plt.gca()

plot_lines_on_ax(ax2, 'evm', use_log10=False)
annotate_max_diff(ax2, 'evm', '%')

ax2.set_title('rms EVM Performance Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Received Optical Power (dBm)', fontsize=12)
ax2.set_ylabel('rms EVM (%)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_xticks(rop)
ax2.axhline(y=12.5, color='black', linestyle='--', linewidth=1.5)
ax2.legend(handles=create_custom_legend(is_evm=True), loc='upper right', ncol=2, fontsize=12, frameon=True)

save_path_evm = os.path.join(save_dir, output_filename_evm)
plt.tight_layout()
plt.savefig(save_path_evm, dpi=300, bbox_inches='tight')
print(f"EVM图已保存: {save_path_evm}")
plt.close()

# ==============================================================================
# 7. 绘制第三张图: BER (PAM4)
# ==============================================================================
print("正在绘制 BER 图...")
plt.figure(figsize=(9, 6))
ax3 = plt.gca()

# 使用 'ber_pam4' 前缀，并开启 log10 转换
plot_lines_on_ax(ax3, 'ber_pam4', use_log10=False)

ax3.set_title('BER (PAM4) Performance Comparison', fontsize=14, fontweight='bold')
ax3.set_xlabel('Received Optical Power (dBm)', fontsize=12)
# 纵坐标题注设为 Received PAM4 BER
ax3.set_ylabel('Received PAM4 BER', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.set_xticks(rop)
# 使用普通图例 (不带 Limit 线)
ax3.legend(handles=create_custom_legend(is_evm=False), loc='upper right', ncol=2, fontsize=10, frameon=True)

save_path_ber = os.path.join(save_dir, output_filename_ber)
plt.tight_layout()
plt.savefig(save_path_ber, dpi=300, bbox_inches='tight')
print(f"BER图已保存: {save_path_ber}")
plt.close()
