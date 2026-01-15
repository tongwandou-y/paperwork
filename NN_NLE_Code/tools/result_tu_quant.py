import matplotlib.pyplot as plt
import numpy as np
import os
import re
from datetime import datetime

# ==============================================================================
# 1. 全局配置区域 (请根据您的实验需求修改此处)
# ==============================================================================

# [基础路径]：存放所有实验数据的根目录
base_root = r'D:\paperwork\Experiment_Data'

# [核心变量]: 实验场景的前缀 (脚本会自动拼接 _{bit}bit 来寻找文件夹)
# 例如: 如果文件夹名为 '20Gsyms_30km_4bit', 这里填 '20Gsyms_30km'
scenario_prefix = '20Gsyms_30km'

# [变量范围]: 需要提取的量化比特列表 (X轴)
quantization_bits = [4, 6, 8, 10, 12]

# [关键接口]: 需要提取的目标接收光功率 (ROP), 单位 dBm
# 修改这里！例如: -20, -18 等
target_rop = -21

# [保存路径]: 结果保存的文件夹 (默认保存在根目录下，也可指定具体路径)
save_dir = r'D:\paperwork\Experiment_Data\Quant_Results'

# [文件名模板]: 实验报告的命名规则 (注意 {} 会被替换为 target_rop)
file_template_nn = 'Report_PRBS31_test_DNN_{}dBm.txt'
file_template_volterra = 'Report_PRBS31_test_Volterra_{}dBm.txt'
file_template_dfe = 'Report_PRBS31_test_DFE_{}dBm.txt'

# ==============================================================================
# 2. 数据提取逻辑
# ==============================================================================

# 准备容器存储数据
data_store = {
    'NN': [],
    'Volterra': [],
    'DFE': []
}


def get_evm_from_file(filepath):
    """从单个文件中提取 rms EVM"""
    if not os.path.exists(filepath):
        print(f"  [Warning] 文件缺失: {filepath}")
        return None  # 或者返回 0

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 正则匹配 rms EVM
    match = re.search(r'rms EVM\s*:\s*([0-9.eE+-]+)', content)
    if match:
        return float(match.group(1))
    else:
        print(f"  [Warning] 未找到 EVM 数据: {filepath}")
        return 0.0


print(f"开始提取数据 (ROP = {target_rop} dBm)...")

for bit in quantization_bits:
    # 拼凑当前 bit 的文件夹名称, e.g., 20Gsyms_30km_4bit
    current_scenario = f"{scenario_prefix}_{bit}bit"
    current_exp_dir = os.path.join(base_root, current_scenario)

    print(f"正在处理: {current_scenario} ...")

    # 构建三个模型的具体文件路径
    path_nn = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'NN', file_template_nn.format(target_rop))
    path_volterra = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'Volterra',
                                 file_template_volterra.format(target_rop))
    path_dfe = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'DFE',
                            file_template_dfe.format(target_rop))

    # 提取并存入列表 (如果文件不存在填入 0 或 None)
    val_nn = get_evm_from_file(path_nn)
    val_vol = get_evm_from_file(path_volterra)
    val_dfe = get_evm_from_file(path_dfe)

    # 如果是 None (文件不存在), 这里暂存为 0 以保证列表长度一致，方便画图
    data_store['NN'].append(val_nn if val_nn is not None else 0)
    data_store['Volterra'].append(val_vol if val_vol is not None else 0)
    data_store['DFE'].append(val_dfe if val_dfe is not None else 0)

# ==============================================================================
# 3. 生成 Markdown 文件 (按照要求格式对齐)
# ==============================================================================
output_filename_base = f"相同接收光功率下不同量化比特的EVM_{target_rop}dBm"
md_file_path = os.path.join(save_dir, f"{output_filename_base}.md")

print(f"正在生成报告: {md_file_path}")

with open(md_file_path, 'w', encoding='utf-8') as f:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 写入标题和元数据
    f.write(f"{output_filename_base}\n\n")
    f.write(f"> **生成时间:** {current_time}\n>\n")
    f.write(f"> **接收光功率**：{target_rop}dBm\n\n")

    f.write("```\n")

    # 格式化输出数据，使用 <10.4f 实现左对齐，保留4位小数
    # 表头(可选，为了清晰我这里不写表头直接写数据，符合您的示例)

    for model_name, values in data_store.items():
        # 将列表中的数字转换为格式化的字符串
        # 您的示例: NN: 5.0461 , 12.1193
        formatted_values = " , ".join([f"{v:.4f}" for v in values])

        # 写入行，模型名称占用 10 个字符宽度以保持对齐
        f.write(f"{model_name:<10}:\t{formatted_values}\n")

    f.write("```\n\n")

print("Markdown 文件生成完毕。")

# ==============================================================================
# 4. 绘图逻辑 (参考附图风格)
# ==============================================================================
png_file_path = os.path.join(save_dir, f"{output_filename_base}.png")
print(f"正在绘图: {png_file_path}")

# 设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(figsize=(8, 6))

# 设置背景颜色 (模仿附图的淡绿色背景，可选，如果不想要可以注释掉)
# ax.set_facecolor('#E8F0E4')
# fig.patch.set_facecolor('#E8F0E4')

# 定义样式
styles = {
    'NN': {'c': 'steelblue', 'm': 'o', 'lbl': 'DNN'},  # 对应附图 Uniform 风格
    'Volterra': {'c': 'peru', 'm': 's', 'lbl': 'Volterra'},  # 对应附图 A-Law 风格
    'DFE': {'c': 'forestgreen', 'm': '^', 'lbl': 'DFE'}  # 对应附图 Mu-Law 风格
}

# 绘制线条
for model in ['NN', 'Volterra', 'DFE']:  # 注意这里键名要和 data_store 一致
    ax.plot(quantization_bits, data_store[model],
            color=styles[model]['c'],
            marker=styles[model]['m'],
            linewidth=2,
            markersize=8,
            label=styles[model]['lbl'])

# 设置坐标轴
ax.set_xlabel('Quantization bits', fontsize=14, fontweight='bold')
ax.set_ylabel('EVM (%)', fontsize=14, fontweight='bold')

# 设置刻度 (X轴强制显示为整数 bits)
ax.set_xticks(quantization_bits)
ax.tick_params(axis='both', which='major', labelsize=12)

# 设置网格
ax.grid(True, linestyle='--', alpha=0.5, color='gray')

# 设置图例 (带边框和阴影效果)
legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, borderpad=0.5)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.0)

# 设置边框加粗 (参考附图风格)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 保存与显示
plt.tight_layout()
plt.savefig(png_file_path, dpi=300, bbox_inches='tight')
print(f"图片已保存至: {png_file_path}")
plt.show()