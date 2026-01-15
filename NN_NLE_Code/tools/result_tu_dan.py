import matplotlib.pyplot as plt
import numpy as np
import os
import re

# ==============================================================================
# 1. 全局配置区域 (只需修改 target_scenario)
# ==============================================================================

# [基础路径]
base_root = r'D:\paperwork\Experiment_Data'

# [核心变量]: 当前要画图的实验场景
# 修改这里，脚本会自动去读这个文件夹下的 .md 文件，并把图存在这里
target_scenario = '20Gsyms_30km_10bit'

# [参数范围]: ROP (横坐标)
rop = np.arange(-27, -14, 1)

# [绘图设置]: 字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 11})


# ==============================================================================
# 2. 自动化数据读取引擎 (核心升级)
# ==============================================================================

def load_data_from_md(scenario_name):
    """
    自动从 .md 文件中提取数据，并自动为变量加上后缀。
    例如: 读取 sqnr_dnn -> 返回 sqnr_dnn_20Gsyms_30km
    """
    # 1. 构建文件路径
    folder_path = os.path.join(base_root, scenario_name)
    md_file_path = os.path.join(folder_path, f'results_{scenario_name}.md')

    if not os.path.exists(md_file_path):
        print(f"[Error] 找不到数据文件: {md_file_path}")
        print("请先运行 indicator_extraction.py 生成该文件！")
        return None

    print(f"正在读取数据文件: {md_file_path}")

    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. 正则表达式匹配: 寻找 变量名 = np.array([ 数字 ]) 的结构
    # re.DOTALL 允许 . 匹配换行符
    pattern = r"(\w+)\s*=\s*np\.array\(\[\s*(.*?)\s*\]\)"
    matches = re.findall(pattern, content, re.DOTALL)

    data_dict = {}

    # 3. 解析数据并重命名
    for var_name, values_str in matches:
        # 去掉换行和空格，按逗号分割转为浮点数
        try:
            values = [float(v) for v in values_str.replace('\n', '').split(',')]

            # 【关键修改】: 加上场景后缀，例如 sqnr_dnn_20Gsyms_30km
            new_var_name = f"{var_name}_{scenario_name}"
            data_dict[new_var_name] = np.array(values)

            # print(f"  -> 已提取: {new_var_name} (长度: {len(values)})") # 调试用
        except ValueError:
            print(f"  [Warning] 无法解析变量 {var_name}")

    return data_dict


# --- 执行读取 ---
all_data = load_data_from_md(target_scenario)

# 如果没读到数据，直接停止
if not all_data:
    exit()


# 为了方便下面绘图代码书写，我们定义一个辅助函数来获取数据
# 这样下面写代码时，看起来就像在使用变量一样
def get_data(prefix, model):
    # 拼接键名: e.g., "sqnr" + "_" + "dnn" + "_" + "20Gsyms_30km"
    key = f"{prefix}_{model}_{target_scenario}"
    if key in all_data:
        return all_data[key]
    else:
        print(f"[Error] 缺少数据: {key}")
        return np.zeros_like(rop)


# ==============================================================================
# 3. 绘图逻辑 (使用提取的数据)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 样式字典
styles = {
    'dnn': {'c': 'blue', 'm': 'o', 'ls': '-', 'lbl': 'DNN-NLE'},
    'volterra': {'c': 'red', 'm': 's', 'ls': '--', 'lbl': 'Volterra'},
    'dfe': {'c': 'green', 'm': '^', 'ls': '-.', 'lbl': 'DFE'}
}

# --- 图 1: PCM SQNR ---
# 直接通过 key 从字典取值，无需定义几十个中间变量
for model in ['dnn', 'volterra', 'dfe']:
    y_data = get_data('sqnr', model)
    s = styles[model]
    ax1.plot(rop, y_data, color=s['c'], marker=s['m'], linestyle=s['ls'],
             linewidth=2, markersize=6, label=s['lbl'])

ax1.set_title(f'PCM SQNR Performance ({target_scenario})', fontsize=14, fontweight='bold')  # 标题带上场景名
ax1.set_xlabel('Received Optical Power (dBm)', fontsize=12)
ax1.set_ylabel('PCM SQNR (dB)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_xticks(rop)
ax1.legend(loc='upper left')

# --- 图 2: rms EVM ---
for model in ['dnn', 'volterra', 'dfe']:
    y_data = get_data('evm', model)
    s = styles[model]
    ax2.plot(rop, y_data, color=s['c'], marker=s['m'], linestyle=s['ls'],
             linewidth=2, markersize=6, label=s['lbl'])

ax2.axhline(y=12.5, color='black', linestyle='--', linewidth=1.5, label='16QAM Limit')

ax2.set_title(f'rms EVM Performance ({target_scenario})', fontsize=14, fontweight='bold')
ax2.set_xlabel('Received Optical Power (dBm)', fontsize=12)
ax2.set_ylabel('rms EVM (%)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_xticks(rop)
ax2.legend(loc='upper right')

# ==============================================================================
# 4. 保存结果
# ==============================================================================
plt.tight_layout()

# 动态生成保存路径: result_20Gsyms_30km.png
save_dir = os.path.join(base_root, target_scenario)
output_filename = f"result_{target_scenario}.png"
save_path = os.path.join(save_dir, output_filename)

if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except:
        pass

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"成功！图片已保存至: {save_path}")
print(f"使用的变量名示例: sqnr_dnn_{target_scenario}")

plt.show()