import matplotlib.pyplot as plt
import numpy as np
import os
import re

# ==============================================================================
# 1. 全局配置区域
# ==============================================================================

# [基础路径]
base_root = r'D:\paperwork\Experiment_Data'

# [核心变量]: 当前要画图的实验场景
target_scenario = '30Gsyms_20km'

# [参数范围]: ROP (横坐标)
rop = np.arange(-27, -14, 1)

# [绘图设置]: 字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 11})


# ==============================================================================
# 2. 自动化数据读取引擎 (基于“物理分块”的绝对精准提取)
# ==============================================================================

def load_data_robust(scenario_name):
    """
    通过将文件物理切割为不同章节来读取数据，
    彻底杜绝正则匹配跨越章节的问题。
    """
    folder_path = os.path.join(base_root, scenario_name)
    md_file_path = os.path.join(folder_path, f'results_{scenario_name}.md')

    if not os.path.exists(md_file_path):
        print(f"[Error] 找不到数据文件: {md_file_path}")
        return None

    print(f"正在读取数据文件: {md_file_path}")

    with open(md_file_path, 'r', encoding='utf-8') as f:
        full_content = f.read()

    data_dict = {}

    # 1. 使用正则将全文按 "## " 切割成多个独立的段落块
    # flags=re.MULTILINE 确保 ^ 能匹配每一行的行首
    segments = re.split(r'^##\s+', full_content, flags=re.MULTILINE)

    # 定义我们需要寻找的指标 (关键词)
    target_sections = ["PCM SQNR", "rms EVM", "BER (PAM4)"]

    for segment in segments:
        if not segment.strip(): continue  # 跳过空段落

        # 获取该段落的第一行作为标题
        lines = segment.split('\n')
        header_line = lines[0].strip()

        # 判断这个段落属于哪个指标
        active_metric = None
        for target in target_sections:
            if target in header_line:
                # 【核心防御机制】：
                # 如果我们在找 BER (PAM4)，必须确保标题里没有 "16-QAM"
                # 这能百分之百防止读错章节
                if target == "BER (PAM4)" and "16-QAM" in header_line:
                    continue

                active_metric = target
                break

        # 如果当前段落是我们需要的指标
        if active_metric:
            # print(f"  -> 锁定章节: {header_line}") # 调试确认用

            # 在这个独立的段落字符串内提取数组
            # 这里的 regex 只会作用于当前段落，绝对不会跑出去
            array_pattern = r"(\w+)\s*=\s*np\.array\(\[\s*(.*?)\s*\]\)"
            arrays = re.findall(array_pattern, segment, re.DOTALL)

            for var_name, values_str in arrays:
                try:
                    # 使用 eval 安全解析数值列表 (支持科学计数法 4.9e-2 和除法 10/100)
                    # 清理可能存在的单引号
                    clean_str = values_str.replace("'", "").replace('"', "")
                    parsed_list = eval(f"[{clean_str}]", {"__builtins__": None}, {})

                    float_array = np.array(parsed_list, dtype=float)

                    # 存入字典
                    unique_key = f"{var_name}_{scenario_name}"
                    data_dict[unique_key] = float_array
                except Exception:
                    pass  # 忽略解析失败的变量

    return data_dict


# 执行读取
all_data = load_data_robust(target_scenario)
if not all_data:
    print("错误：未提取到任何数据。")
    exit()


def get_data(prefix, model):
    key = f"{prefix}_{model}_{target_scenario}"
    if key in all_data:
        return all_data[key]
    else:
        # 找不到数据时返回全0，避免报错
        return np.zeros_like(rop, dtype=float)


# ==============================================================================
# 3. 通用单图绘制函数
# ==============================================================================

styles = {
    'dnn': {'c': 'blue', 'm': 'o', 'ls': '-', 'lbl': 'DNN-NLE'},
    'volterra': {'c': 'red', 'm': 's', 'ls': '--', 'lbl': 'Volterra'},
    'dfe': {'c': 'green', 'm': '^', 'ls': '-.', 'lbl': 'DFE'}
}


def plot_single_figure(metric_prefix, title, ylabel, filename_suffix, use_log10=False):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for model in ['dnn', 'volterra', 'dfe']:
        y_data = get_data(metric_prefix, model)

        # if use_log10:
        #     # [Log10 安全处理]
        #     # 过滤 <= 0 的非法值 (BER=0 时 log10 会变成 -inf)
        #     # 我们将其替换为 1e-9 (log10后为 -9)，代表极小误码率
        #     safe_y = np.where(y_data > 0, y_data, 1e-9)
        #     plot_y = np.log10(safe_y)
        # else:
        plot_y = y_data

        s = styles[model]
        ax.plot(rop, plot_y, color=s['c'], marker=s['m'], linestyle=s['ls'],
                linewidth=2, markersize=6, label=s['lbl'])

    # 设置标题和标签
    ax.set_title(f'{title} ({target_scenario})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Received Optical Power (dBm)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(rop)
    ax.legend(loc='best')

    # 保存
    save_dir = os.path.join(base_root, target_scenario)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    output_filename = f"result_{target_scenario}_{filename_suffix}.png"
    save_path = os.path.join(save_dir, output_filename)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[{filename_suffix}] 图片已保存至: {save_path}")
    plt.close()


# ==============================================================================
# 4. 执行绘图
# ==============================================================================

# 1. PCM SQNR
plot_single_figure(
    metric_prefix='sqnr',
    title='PCM SQNR Performance',
    ylabel='PCM SQNR (dB)',
    filename_suffix='SQNR'
)

# 2. rms EVM
plot_single_figure(
    metric_prefix='evm',
    title='rms EVM Performance',
    ylabel='rms EVM (%)',
    filename_suffix='EVM'
)

# 3. BER (PAM4)
plot_single_figure(
    metric_prefix='ber_pam4',  # 这里对应提取脚本里的变量前缀
    title='BER (PAM4) Performance',
    ylabel='Received PAM4 BER',
    filename_suffix='BER_PAM4',
    use_log10=True
)