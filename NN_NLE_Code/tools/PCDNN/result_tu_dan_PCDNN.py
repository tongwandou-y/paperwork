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
target_scenario = '20Gsyms_20km'

# [参数范围]: ROP (横坐标)
rop = np.arange(-24, -14, 1)

# [新增功能]: 手动选择要画的实验曲线 (必须与你 .md 文件中包含的组别一致)
SELECTED_GROUPS = [
    'DFE',
    'Volterra',
    'DNN',
    'SH_DNN',
    'PP_CDNN'
]

# [坐标系控制]: 设置对应指标是否开启对数坐标系 (True 开 / False 关)
USE_LOG_Y_SQNR = False   # SQNR 默认线性坐标
USE_LOG_Y_EVM  = False   # EVM 默认线性坐标
USE_LOG_Y_BER  = True    # BER 默认对数坐标

# [绘图设置]: 字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 11})


# ==============================================================================
# 2. 自动化数据读取引擎
# ==============================================================================

def load_data_robust(scenario_name):
    folder_path = os.path.join(base_root, scenario_name)
    md_file_path = os.path.join(folder_path, f'results_{scenario_name}.md')

    if not os.path.exists(md_file_path):
        print(f"[Error] 找不到数据文件: {md_file_path}")
        return None

    print(f"正在读取数据文件: {md_file_path}")

    with open(md_file_path, 'r', encoding='utf-8') as f:
        full_content = f.read()

    data_dict = {}
    segments = re.split(r'^##\s+', full_content, flags=re.MULTILINE)
    target_sections = ["PCM SQNR", "rms EVM", "BER (PAM4)"]

    for segment in segments:
        if not segment.strip(): continue

        lines = segment.split('\n')
        header_line = lines[0].strip()

        active_metric = None
        for target in target_sections:
            if target in header_line:
                if target == "BER (PAM4)" and "16-QAM" in header_line:
                    continue
                active_metric = target
                break

        if active_metric:
            array_pattern = r"(\w+)\s*=\s*np\.array\(\[\s*(.*?)\s*\]\)"
            arrays = re.findall(array_pattern, segment, re.DOTALL)

            for var_name, values_str in arrays:
                try:
                    clean_str = values_str.replace("'", "").replace('"', "")
                    parsed_list = eval(f"[{clean_str}]", {"__builtins__": None}, {})
                    float_array = np.array(parsed_list, dtype=float)
                    unique_key = f"{var_name}_{scenario_name}"
                    data_dict[unique_key] = float_array
                except Exception:
                    pass

    return data_dict


all_data = load_data_robust(target_scenario)
if not all_data:
    print("错误：未提取到任何数据。")
    exit()


def get_data(prefix, model):
    key = f"{prefix}_{model}_{target_scenario}"
    if key in all_data:
        return all_data[key]
    else:
        print(f"[警告] 缺失数据: {key}，将使用全 0 替代。")
        return np.zeros_like(rop, dtype=float)


# ==============================================================================
# 3. 通用单图绘制函数 (含动态组别选择)
# ==============================================================================

# 构建全量样式库
ALL_STYLES = {
    'DFE':      {'c': '#2ca02c', 'm': '^', 'ls': '--', 'lbl': 'DFE'},
    'Volterra': {'c': '#1f77b4', 'm': 's', 'ls': '-.', 'lbl': 'Volterra'},
    'DNN':      {'c': 'gray',    'm': 'v', 'ls': ':',  'lbl': 'DNN'},
    'SH_DNN':   {'c': 'orange',  'm': 'd', 'ls': '--', 'lbl': 'SH-DNN'},
    'PP_CDNN':  {'c': '#d62728', 'm': 'o', 'ls': '-',  'lbl': 'PP-CDNN'}
}

# 动态过滤当前需要的样式
styles = {k: ALL_STYLES[k] for k in SELECTED_GROUPS if k in ALL_STYLES}

if not styles:
    print("[致命错误] 选中的画图组别为空，请检查 SELECTED_GROUPS 变量！")
    exit()

def plot_single_figure(metric_prefix, title, ylabel, filename_suffix, use_log_y=False):
    fig, ax = plt.subplots(figsize=(8, 6))

    # 动态读取选中的模型
    models_to_plot = list(styles.keys())

    for model in models_to_plot:
        y_data = get_data(metric_prefix, model)

        # 若开启对数轴，需处理数值 <= 0 的情况，防止 matplotlib 绘图报错
        if use_log_y:
            plot_y = np.where(y_data <= 0, 1e-9, y_data)
        else:
            plot_y = y_data

        s = styles[model]
        ax.plot(rop, plot_y, color=s['c'], marker=s['m'], linestyle=s['ls'],
                linewidth=2, markersize=6, label=s['lbl'])

    # --- 坐标系控制逻辑 ---
    if use_log_y:
        ax.set_yscale('log')
        # 限制 Y 轴下界为 10^-6，防止被 1e-9 占满纵向空间
        ax.set_ylim(bottom=1e-6, top=1)

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
    filename_suffix='SQNR',
    use_log_y=USE_LOG_Y_SQNR
)

# 2. rms EVM
plot_single_figure(
    metric_prefix='evm',
    title='rms EVM Performance',
    ylabel='rms EVM (%)',
    filename_suffix='EVM',
    use_log_y=USE_LOG_Y_EVM
)

# 3. BER (PAM4)
plot_single_figure(
    metric_prefix='ber_pam4',
    title='BER (PAM4) Performance',
    ylabel='Received PAM4 BER',
    filename_suffix='BER_PAM4',
    use_log_y=USE_LOG_Y_BER
)