# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import re
# from datetime import datetime
#
# # ==============================================================================
# # 1. 全局配置区域 (请根据您的实验需求修改此处)
# # ==============================================================================
#
# # [基础路径]：存放所有实验数据的根目录
# base_root = r'D:\paperwork\Experiment_Data\Quant'
#
# # [核心变量]: 实验场景的前缀 (脚本会自动拼接 _{bit}bit 来寻找文件夹)
# # 例如: 如果文件夹名为 '20Gsyms_30km_4bit', 这里填 '20Gsyms_30km'
# scenario_prefix = '20Gsyms_30km'
#
# # [变量范围]: 需要提取的量化比特列表 (X轴)
# quantization_bits = [3, 4, 5, 6, 7, 8, 9, 10]
#
# # [关键接口]: 需要提取的目标接收光功率 (ROP), 单位 dBm
# # 修改这里！例如: -20, -18 等
# target_rop = -20
#
# # [保存路径]: 结果保存的文件夹 (默认保存在根目录下，也可指定具体路径)
# save_dir = r'D:\paperwork\Experiment_Data\Quant\Quant_Results'
#
# # [文件名模板]: 实验报告的命名规则 (注意 {} 会被替换为 target_rop)
# file_template_nn = 'Report_PRBS31_test_DNN_{}dBm.txt'
# file_template_volterra = 'Report_PRBS31_test_Volterra_{}dBm.txt'
# file_template_dfe = 'Report_PRBS31_test_DFE_{}dBm.txt'
#
# # ==============================================================================
# # 2. 数据提取逻辑
# # ==============================================================================
#
# # 准备容器存储数据
# data_store = {
#     'NN': [],
#     'Volterra': [],
#     'DFE': []
# }
#
#
# def get_evm_from_file(filepath):
#     """从单个文件中提取 rms EVM"""
#     if not os.path.exists(filepath):
#         print(f"  [Warning] 文件缺失: {filepath}")
#         return None  # 或者返回 0
#
#     with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
#         content = f.read()
#
#     # 正则匹配 rms EVM
#     match = re.search(r'rms EVM\s*:\s*([0-9.eE+-]+)', content)
#     if match:
#         return float(match.group(1))
#     else:
#         print(f"  [Warning] 未找到 EVM 数据: {filepath}")
#         return 0.0
#
#
# print(f"开始提取数据 (ROP = {target_rop} dBm)...")
#
# for bit in quantization_bits:
#     # 拼凑当前 bit 的文件夹名称, e.g., 20Gsyms_30km_4bit
#     current_scenario = f"{scenario_prefix}_{bit}bit"
#     current_exp_dir = os.path.join(base_root, current_scenario)
#
#     print(f"正在处理: {current_scenario} ...")
#
#     # 构建三个模型的具体文件路径
#     path_nn = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'NN', file_template_nn.format(target_rop))
#     path_volterra = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'Volterra',
#                                  file_template_volterra.format(target_rop))
#     path_dfe = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'DFE',
#                             file_template_dfe.format(target_rop))
#
#     # 提取并存入列表 (如果文件不存在填入 0 或 None)
#     val_nn = get_evm_from_file(path_nn)
#     val_vol = get_evm_from_file(path_volterra)
#     val_dfe = get_evm_from_file(path_dfe)
#
#     # 如果是 None (文件不存在), 这里暂存为 0 以保证列表长度一致，方便画图
#     data_store['NN'].append(val_nn if val_nn is not None else 0)
#     data_store['Volterra'].append(val_vol if val_vol is not None else 0)
#     data_store['DFE'].append(val_dfe if val_dfe is not None else 0)
#
# # ==============================================================================
# # 3. 生成 Markdown 文件 (按照要求格式对齐)
# # ==============================================================================
# output_filename_base = f"相同接收光功率下不同量化比特的EVM_{target_rop}dBm"
# md_file_path = os.path.join(save_dir, f"{output_filename_base}.md")
#
# print(f"正在生成报告: {md_file_path}")
#
# with open(md_file_path, 'w', encoding='utf-8') as f:
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
#     # 写入标题和元数据
#     f.write(f"{output_filename_base}\n\n")
#     f.write(f"> **生成时间:** {current_time}\n>\n")
#     f.write(f"> **接收光功率**：{target_rop}dBm\n\n")
#     f.write(f"> **量化比特范围**：{quantization_bits}\n\n")
#
#     f.write("```\n")
#
#     # 格式化输出数据，使用 <10.4f 实现左对齐，保留4位小数
#     # 表头(可选，为了清晰我这里不写表头直接写数据，符合您的示例)
#
#     for model_name, values in data_store.items():
#         # 将列表中的数字转换为格式化的字符串
#         # 您的示例: NN: 5.0461 , 12.1193
#         formatted_values = " , ".join([f"{v:.4f}" for v in values])
#
#         # 写入行，模型名称占用 10 个字符宽度以保持对齐
#         f.write(f"{model_name:<10}:\t{formatted_values}\n")
#
#     f.write("```\n\n")
#
# print("Markdown 文件生成完毕。")
#
# # ==============================================================================
# # 4. 绘图逻辑 (参考附图风格)
# # ==============================================================================
# png_file_path = os.path.join(save_dir, f"{output_filename_base}.png")
# print(f"正在绘图: {png_file_path}")
#
# # 设置字体
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams.update({'font.size': 12})
#
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # 设置背景颜色 (模仿附图的淡绿色背景，可选，如果不想要可以注释掉)
# # ax.set_facecolor('#E8F0E4')
# # fig.patch.set_facecolor('#E8F0E4')
#
# # 定义样式
# styles = {
#     'NN': {'c': 'steelblue', 'm': 'o', 'lbl': 'DNN'},  # 对应附图 Uniform 风格
#     'Volterra': {'c': 'peru', 'm': 's', 'lbl': 'Volterra'},  # 对应附图 A-Law 风格
#     'DFE': {'c': 'forestgreen', 'm': '^', 'lbl': 'DFE'}  # 对应附图 Mu-Law 风格
# }
#
# # 绘制线条
# for model in ['NN', 'Volterra', 'DFE']:  # 注意这里键名要和 data_store 一致
#     ax.plot(quantization_bits, data_store[model],
#             color=styles[model]['c'],
#             marker=styles[model]['m'],
#             linewidth=2,
#             markersize=8,
#             label=styles[model]['lbl'])
#
# # 设置坐标轴
# ax.set_xlabel('Quantization bits', fontsize=14, fontweight='bold')
# ax.set_ylabel('EVM (%)', fontsize=14, fontweight='bold')
#
# # 设置刻度 (X轴强制显示为整数 bits)
# ax.set_xticks(quantization_bits)
# ax.tick_params(axis='both', which='major', labelsize=12)
#
# # 设置网格
# ax.grid(True, linestyle='--', alpha=0.5, color='gray')
#
# # 设置图例 (带边框和阴影效果)
# legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, borderpad=0.5)
# legend.get_frame().set_edgecolor('black')
# legend.get_frame().set_linewidth(1.0)
#
# # 设置边框加粗 (参考附图风格)
# for spine in ax.spines.values():
#     spine.set_linewidth(1.5)
#
# # 保存与显示
# plt.tight_layout()
# plt.savefig(png_file_path, dpi=300, bbox_inches='tight')
# print(f"图片已保存至: {png_file_path}")
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import re
# from datetime import datetime
# # 【关键修改】引入 PchipInterpolator，专门用于单调数据的平滑，防止过冲和负数
# from scipy.interpolate import PchipInterpolator
#
# # ==============================================================================
# # 1. 全局配置区域 (请根据您的实验需求修改此处)
# # ==============================================================================
#
# # [基础路径]：存放所有实验数据的根目录
# base_root = r'D:\paperwork\Experiment_Data\Quant'
#
# # [核心变量]: 实验场景的前缀
# scenario_prefix = '20Gsyms_30km'
#
# # [变量范围]: 需要提取的量化比特列表 (X轴)
# # 确保这里是排好序的
# quantization_bits = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#
# # [关键接口]: 需要提取的目标接收光功率 (ROP), 单位 dBm
# target_rop = -20
#
# # [保存路径]: 结果保存的文件夹
# save_dir = r'D:\paperwork\Experiment_Data\Quant\Quant_Results'
#
# # [文件名模板]
# file_template_nn = 'Report_PRBS31_test_DNN_{}dBm.txt'
# file_template_volterra = 'Report_PRBS31_test_Volterra_{}dBm.txt'
# file_template_dfe = 'Report_PRBS31_test_DFE_{}dBm.txt'
#
# # ==============================================================================
# # 2. 数据提取逻辑
# # ==============================================================================
#
# # 准备容器存储数据
# data_store = {
#     'NN': [],
#     'Volterra': [],
#     'DFE': []
# }
#
#
# def get_evm_from_file(filepath):
#     """从单个文件中提取 rms EVM"""
#     if not os.path.exists(filepath):
#         return None
#     with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
#         content = f.read()
#     match = re.search(r'rms EVM\s*:\s*([0-9.eE+-]+)', content)
#     if match:
#         return float(match.group(1))
#     else:
#         return 0.0
#
#
# print(f"开始提取数据 (ROP = {target_rop} dBm)...")
#
# for bit in quantization_bits:
#     current_scenario = f"{scenario_prefix}_{bit}bit"
#     current_exp_dir = os.path.join(base_root, current_scenario)
#     print(f"正在处理: {current_scenario} ...")
#
#     path_nn = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'NN', file_template_nn.format(target_rop))
#     path_volterra = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'Volterra',
#                                  file_template_volterra.format(target_rop))
#     path_dfe = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'DFE',
#                             file_template_dfe.format(target_rop))
#
#     val_nn = get_evm_from_file(path_nn)
#     val_vol = get_evm_from_file(path_volterra)
#     val_dfe = get_evm_from_file(path_dfe)
#
#     data_store['NN'].append(val_nn if val_nn is not None else 0)
#     data_store['Volterra'].append(val_vol if val_vol is not None else 0)
#     data_store['DFE'].append(val_dfe if val_dfe is not None else 0)
#
# # ==============================================================================
# # 3. 生成 Markdown 文件
# # ==============================================================================
# if not os.path.exists(save_dir):
#     try:
#         os.makedirs(save_dir)
#     except:
#         pass
#
# output_filename_base = f"相同接收光功率下不同量化比特的EVM_{target_rop}dBm"
# md_file_path = os.path.join(save_dir, f"{output_filename_base}.md")
#
# with open(md_file_path, 'w', encoding='utf-8') as f:
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     f.write(f"{output_filename_base}\n\n")
#     f.write(f"> **生成时间:** {current_time}\n>\n")
#     f.write(f"> **接收光功率**：{target_rop}dBm\n\n")
#     f.write("```\n")
#     for model_name, values in data_store.items():
#         formatted_values = " , ".join([f"{v:.4f}" for v in values])
#         f.write(f"{model_name:<10}:\t{formatted_values}\n")
#     f.write("```\n\n")
#
# # ==============================================================================
# # 4. 绘图逻辑 (使用 Pchip 修正负数和震荡问题)
# # ==============================================================================
# png_file_path = os.path.join(save_dir, f"{output_filename_base}.png")
# print(f"正在绘图: {png_file_path}")
#
# # 设置字体
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams.update({'font.size': 12})
#
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # 定义样式
# styles = {
#     'NN': {'c': 'steelblue', 'm': 'o', 'lbl': 'DNN'},
#     'Volterra': {'c': 'peru', 'm': 's', 'lbl': 'Volterra'},
#     'DFE': {'c': 'forestgreen', 'm': '^', 'lbl': 'DFE'}
# }
#
# # 绘制线条 (包含平滑处理)
# for model in ['NN', 'Volterra', 'DFE']:
#     x = np.array(quantization_bits)
#     y = np.array(data_store[model])
#
#     # 过滤掉全0的数据（防止空数据绘图报错）
#     if np.all(y == 0):
#         print(f"提示: {model} 数据全为0，跳过绘制或仅绘制点。")
#
#     # 检查是否有足够的数据点进行插值 (至少需要2个点)
#     if len(x) > 2:
#         # 1. 创建更密集的 X 轴数据 (用于绘制平滑曲线)
#         x_smooth = np.linspace(x.min(), x.max(), 300)
#
#         try:
#             # 【关键修改】使用 PchipInterpolator 代替 make_interp_spline
#             # Pchip 可以保持单调性，避免剧烈震荡和负数
#             interpolator = PchipInterpolator(x, y)
#             y_smooth = interpolator(x_smooth)
#
#             # 【双重保险】强制将小于0的值修正为0 (虽然Pchip通常不需要，但以防万一)
#             y_smooth[y_smooth < 0] = 0
#
#             # 3. 绘制平滑曲线 (作为连线)
#             ax.plot(x_smooth, y_smooth,
#                     color=styles[model]['c'],
#                     linewidth=2,
#                     label=styles[model]['lbl'])
#
#             # 4. 绘制原始数据点 (Marker)
#             ax.scatter(x, y,
#                        color=styles[model]['c'],
#                        marker=styles[model]['m'],
#                        s=60,  # 点稍微大一点更好看
#                        zorder=5)
#
#         except Exception as e:
#             print(f"平滑处理失败 ({model}): {e}，绘制为折线")
#             ax.plot(x, y, color=styles[model]['c'], marker=styles[model]['m'],
#                     linewidth=2, markersize=8, label=styles[model]['lbl'])
#     else:
#         # 数据点太少，直接画折线
#         ax.plot(x, y, color=styles[model]['c'], marker=styles[model]['m'],
#                 linewidth=2, markersize=8, label=styles[model]['lbl'])
#
# # 设置坐标轴
# ax.set_xlabel('Quantization bits', fontsize=14, fontweight='bold')
# ax.set_ylabel('EVM (%)', fontsize=14, fontweight='bold')
#
# # 设置刻度
# ax.set_xticks(quantization_bits)
# # Y轴强制从0开始显示，避免显示负数区域
# ax.set_ylim(bottom=0)
#
# ax.tick_params(axis='both', which='major', labelsize=12)
#
# # 设置网格
# ax.grid(True, linestyle='--', alpha=0.5, color='gray')
#
# # 设置图例
# legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, borderpad=0.5)
# legend.get_frame().set_edgecolor('black')
# legend.get_frame().set_linewidth(1.0)
#
# # 边框加粗
# for spine in ax.spines.values():
#     spine.set_linewidth(1.5)
#
# plt.tight_layout()
# plt.savefig(png_file_path, dpi=300, bbox_inches='tight')
# print(f"图片已保存至: {png_file_path}")
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import os
import re
from datetime import datetime
# 引入 PchipInterpolator 用于平滑曲线
from scipy.interpolate import PchipInterpolator

# ==============================================================================
# 1. 全局配置区域
# ==============================================================================

# [基础路径]
base_root = r'D:\paperwork\Experiment_Data\Quant-最优'

# [核心变量]: 实验场景的前缀
scenario_prefix = '20Gsyms_30km'

# [变量范围]: X轴 (量化比特)
quantization_bits = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# [关键接口]: 目标接收光功率 (ROP)
target_rop = -20

# [保存路径]
save_dir = r'D:\paperwork\Experiment_Data\Quant-最优\Quant_Results'

# [文件名模板]
file_template_nn = 'Report_PRBS31_test_DNN_{}dBm.txt'
file_template_volterra = 'Report_PRBS31_test_Volterra_{}dBm.txt'
file_template_dfe = 'Report_PRBS31_test_DFE_{}dBm.txt'

# ==============================================================================
# 2. 数据提取逻辑
# ==============================================================================

# 准备容器
evm_store = {'NN': [], 'Volterra': [], 'DFE': []}
ser_store = {'NN': [], 'Volterra': [], 'DFE': []}


def get_metrics_from_file(filepath):
    """从单个文件中提取 EVM 和 SER"""
    evm_val = 0.0
    ser_val = 1.0  # SER 默认为 1 (表示误码率极高或文件缺失)

    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 提取 EVM
        match_evm = re.search(r'rms EVM\s*:\s*([0-9.eE+-]+)', content)
        if match_evm:
            evm_val = float(match_evm.group(1))

        # 提取 SER (PAM4)
        match_ser = re.search(r'SER \(PAM4\)\s*:\s*([0-9.eE+-]+)', content)
        if match_ser:
            ser_val = float(match_ser.group(1))

    return evm_val, ser_val


print(f"开始提取数据 (ROP = {target_rop} dBm)...")

for bit in quantization_bits:
    current_scenario = f"{scenario_prefix}_{bit}bit"
    current_exp_dir = os.path.join(base_root, current_scenario)
    print(f"正在处理: {current_scenario} ...")

    # 构建路径
    path_nn = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'NN', file_template_nn.format(target_rop))
    path_volterra = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'Volterra',
                                 file_template_volterra.format(target_rop))
    path_dfe = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'DFE',
                            file_template_dfe.format(target_rop))

    # 提取数据
    evm_nn, ser_nn = get_metrics_from_file(path_nn)
    evm_vol, ser_vol = get_metrics_from_file(path_volterra)
    evm_dfe, ser_dfe = get_metrics_from_file(path_dfe)

    # 存入列表
    evm_store['NN'].append(evm_nn)
    evm_store['Volterra'].append(evm_vol)
    evm_store['DFE'].append(evm_dfe)

    ser_store['NN'].append(ser_nn)
    ser_store['Volterra'].append(ser_vol)
    ser_store['DFE'].append(ser_dfe)

# ==============================================================================
# 3. 生成 Markdown 文件
# ==============================================================================
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except:
        pass

md_filename = f"Quantization_Metrics_{target_rop}dBm.md"
md_path = os.path.join(save_dir, md_filename)

with open(md_path, 'w', encoding='utf-8') as f:
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"# 量化比特性能分析报告\n\n> 时间: {t}\n> ROP: {target_rop} dBm\n\n")

    # 表1: EVM
    f.write("## 1. rms EVM (%)\n```\n")
    for k, v in evm_store.items():
        vals = " , ".join([f"{x:.4f}" for x in v])
        f.write(f"{k:<10}:\t{vals}\n")
    f.write("```\n\n")

    # 表2: SER
    f.write("## 2. SER (PAM4)\n```\n")
    for k, v in ser_store.items():
        vals = " , ".join([f"{x:.2e}" for x in v])
        f.write(f"{k:<10}:\t{vals}\n")
    f.write("```\n\n")


# ==============================================================================
# 4. 通用绘图函数 (支持多条阈值线)
# ==============================================================================
def plot_chart(data_dict, title_suffix, ylabel, filename_suffix, is_log_scale=False, threshold_list=None):
    """
    threshold_list: 列表，每个元素是字典 {'val': float, 'label': str, 'ls': str, 'color': str}
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    styles = {
        'NN': {'c': 'steelblue', 'm': 'o', 'lbl': 'DNN'},
        'Volterra': {'c': 'peru', 'm': 's', 'lbl': 'Volterra'},
        'DFE': {'c': 'forestgreen', 'm': '^', 'lbl': 'DFE'}
    }

    x = np.array(quantization_bits)

    for model in ['NN', 'Volterra', 'DFE']:
        y = np.array(data_dict[model])

        # 绘图逻辑
        if len(x) > 2 and not is_log_scale:
            try:
                x_smooth = np.linspace(x.min(), x.max(), 300)
                interpolator = PchipInterpolator(x, y)
                y_smooth = interpolator(x_smooth)
                if not is_log_scale: y_smooth[y_smooth < 0] = 0

                ax.plot(x_smooth, y_smooth, color=styles[model]['c'], linewidth=2, label=styles[model]['lbl'])
                ax.scatter(x, y, color=styles[model]['c'], marker=styles[model]['m'], s=60, zorder=5)
            except:
                ax.plot(x, y, color=styles[model]['c'], marker=styles[model]['m'], linewidth=2,
                        label=styles[model]['lbl'])
        else:
            ax.plot(x, y, color=styles[model]['c'], marker=styles[model]['m'], linewidth=2, markersize=8,
                    label=styles[model]['lbl'])

    # --- [新增] 绘制多条阈值线 ---
    if threshold_list:
        for th in threshold_list:
            ax.axhline(
                y=th['val'],
                color=th.get('color', 'black'),
                linestyle=th.get('ls', '--'),
                linewidth=1.5,
                label=th['label']
            )

    # 设置属性
    ax.set_xlabel('Quantization bits', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xticks(quantization_bits)
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')

    # Y轴设置
    if is_log_scale:
        ax.set_yscale('log')
    else:
        ax.set_ylim(bottom=0)

    # 图例
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, borderpad=0.5)
    legend.get_frame().set_edgecolor('black')

    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 保存
    full_path = os.path.join(save_dir, f"{filename_suffix}_{target_rop}dBm.png")
    plt.tight_layout()
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存: {full_path}")
    plt.close()


# ==============================================================================
# 5. 执行绘图
# ==============================================================================

# 1. 绘制 EVM 图
print("正在绘制 EVM 图...")
thresholds_evm = [
    {'val': 12.5, 'label': '16QAM Limit', 'ls': '--', 'color': 'black'}
]
plot_chart(evm_store, "EVM Performance", "EVM (%)", "EVM_Quant", is_log_scale=False, threshold_list=thresholds_evm)

# 2. 绘制 SER 图 (同时包含 SD-FEC 和 HD-FEC)
print("正在绘制 SER 图...")
thresholds_ser = [
    {'val': 2.0e-2, 'label': r'SD-FEC $2.0 \times 10^{-2}$', 'ls': '--', 'color': 'black'},  # 虚线
    {'val': 3.8e-3, 'label': r'HD-FEC $3.8 \times 10^{-3}$', 'ls': '-.', 'color': 'black'}  # 点划线，以区分
]
plot_chart(ser_store, "SER Performance", "SER (PAM4)", "SER_Quant", is_log_scale=True, threshold_list=thresholds_ser)