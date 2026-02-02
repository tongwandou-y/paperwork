import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import re

# ==============================================================================
# 1. 全局配置区域
# ==============================================================================

base_root = r'D:\paperwork\Experiment_Data'

# [对比列表]：不同速率 (20Gsyms固定)
scenarios_list = [
    '20Gsyms_10km',
    '20Gsyms_20km',
    '20Gsyms_30km'
]

# [保存位置]
save_dir = r'D:\paperwork\Experiment_Data\Comparison_Results'

# [文件名接口]
output_filename_sqnr = 'PCM_SQNR_不同光纤长度_20Gsyms.png'
output_filename_evm = 'rms_EVM_不同光纤长度_20Gsyms.png'
output_filename_ber = 'BER_PAM4_不同光纤长度_20Gsyms.png'

# [参数范围]：您可以随意调整这个范围
# 例如：改为 np.arange(-22, -14, 1) 测试自动切片功能
rop = np.arange(-22, -14, 1)

# [关键常量]：定义原始数据文件(.md)中数据的起始点
# 这是为了让脚本知道 -22dBm 对应数组的第几个元素
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
                # 简单清洗数据：去掉可能的引号
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
    获取数据并根据当前的 rop 范围进行智能切片 (Slicing)
    """
    key = f"{prefix}_{model}_{scenario}"
    full_data = all_data.get(key)

    if full_data is None:
        # 如果没有数据，返回对应长度的 0 数组
        return np.zeros(len(rop), dtype=float)

    # --- [核心修复] 数据对齐逻辑 ---
    # 计算 rop[0] (当前画图起始点) 相对于 DATA_START_DBM (文件数据起始点) 的偏移索引
    start_index = int(rop[0] - DATA_START_DBM)
    end_index = start_index + len(rop)

    # 边界检查
    if start_index < 0 or end_index > len(full_data):
        print(f"[Warning] {scenario}: 请求的 ROP 范围 {rop[0]}~{rop[-1]}dBm 超出了数据范围!")
        return np.zeros(len(rop), dtype=float)

    # 返回切片后的数据
    return full_data[start_index: end_index]


if not all_data:
    print("错误: 没有加载到数据，请检查路径。")
    exit()


# ==============================================================================
# 3. 差值计算与标注函数
# ==============================================================================

def annotate_max_diff(ax, metric_prefix, unit_label):
    print(f"\n{'=' * 20} {metric_prefix.upper()} 差值详情 {'=' * 20}")
    offset_map = [0, 2, -2]

    for i, sc_name in enumerate(scenarios_list):
        y_dnn = get_data(metric_prefix, 'dnn', sc_name)
        y_vol = get_data(metric_prefix, 'volterra', sc_name)

        diffs = np.abs(y_dnn - y_vol)
        max_idx = np.argmax(diffs)
        max_diff = diffs[max_idx]
        target_rop = rop[max_idx]
        y_d_max = y_dnn[max_idx]
        y_v_max = y_vol[max_idx]

        print(f"场景: {sc_name} | Max Diff: {max_diff:.4f} {unit_label} @ {target_rop} dBm")

        y_bottom = min(y_d_max, y_v_max)
        y_top = max(y_d_max, y_v_max)

        ax.annotate(
            '', xy=(target_rop, y_bottom), xytext=(target_rop, y_top),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5, shrinkA=0, shrinkB=0)
        )

        text_x_pos = target_rop + 1.2
        if text_x_pos > rop[-1]:
            text_x_pos = target_rop - 1.2

        mid_y = (y_bottom + y_top) / 2
        ax.plot([target_rop, text_x_pos], [mid_y, mid_y], color='black', linestyle='--', linewidth=1)

        text_y_pos = mid_y + offset_map[i % 3]
        label_text = f"~{max_diff:.1f}{unit_label}"
        ax.text(text_x_pos, text_y_pos, label_text, color='black',
                fontsize=11, fontweight='bold', va='center')


# ==============================================================================
# 4. 样式配置
# ==============================================================================

model_styles = {
    'dnn': {'c': 'blue', 'm': 'o', 'name': 'DNN-NLE'},
    'volterra': {'c': 'red', 'm': 's', 'name': 'Volterra'},
    'dfe': {'c': 'green', 'm': '^', 'name': 'DFE'}
}

scenario_styles = [
    {'ls': '-', 'mfc': 'auto'},  # 5G
    {'ls': '--', 'mfc': 'none'},  # 10G
    {'ls': ':', 'mfc': 'auto'}  # 20G
]


def plot_lines_on_ax(ax, data_prefix, use_log10=False):
    for i, sc_name in enumerate(scenarios_list):
        style_cfg = scenario_styles[i % 3]
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
    legend_elements = []
    # 模型
    legend_elements.append(Line2D([0], [0], color='blue', marker='o', lw=0, label='DNN-NLE'))
    legend_elements.append(Line2D([0], [0], color='red', marker='s', lw=0, label='Volterra'))
    legend_elements.append(Line2D([0], [0], color='green', marker='^', lw=0, label='DFE'))
    # 空行
    legend_elements.append(Line2D([0], [0], color='white', label=' '))
    # 距离
    legend_elements.append(
        Line2D([0], [0], color='black', marker='o', linestyle='-', markerfacecolor='black', label='10km'))
    legend_elements.append(
        Line2D([0], [0], color='black', marker='o', linestyle='--', markerfacecolor='white', label='20km'))
    legend_elements.append(
        Line2D([0], [0], color='black', marker='o', linestyle=':', markerfacecolor='black', label='30km'))

    if is_evm:
        legend_elements.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='16QAM Limit'))

    return legend_elements


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

# 【修正】开启 use_log10=True，确保误码率曲线正常显示
plot_lines_on_ax(ax3, 'ber_pam4', use_log10=False)

ax3.set_title('BER (PAM4) Performance Comparison', fontsize=14, fontweight='bold')
ax3.set_xlabel('Received Optical Power (dBm)', fontsize=12)
ax3.set_ylabel('Received PAM4 BER (log10)', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.set_xticks(rop)
ax3.legend(handles=create_custom_legend(is_evm=False), loc='upper right', ncol=2, fontsize=10, frameon=True)

save_path_ber = os.path.join(save_dir, output_filename_ber)
plt.tight_layout()
plt.savefig(save_path_ber, dpi=300, bbox_inches='tight')
print(f"BER图已保存: {save_path_ber}")
plt.close()