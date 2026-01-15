import os
import re
from datetime import datetime  # 引入时间模块

# ==============================================================================
# 1. 全局配置区域 (所有可修改的参数都在这里)
# ==============================================================================

# [基础路径]：存放所有实验数据的根目录 (一般不用动)
base_root = r'D:\paperwork\Experiment_Data'

# [核心变量]: 当前要处理的实验场景名称 (对应文件夹名，自动拼接到路径中)
# 修改这里即可切换不同实验组，例如: '20Gsyms_30km', '5Gsyms_20km' 等
target_scenario = '20Gsyms_30km_10bit'

# [参数范围]: X dBm 变化的范围
# 根据示例数据推断，数组是从“低质量(-27)”到“高质量(-15)”排列的
# 如果需要从 -15 到 -27，请改为 range(-15, -28, -1)
# Stop (-14): 不包含在内（截止到它前面的一个数）
x_values = list(range(-27, -14))

# [文件名模板]: 定义实验报告的命名规则
# {} 会被自动替换为上面的 dBm 数值
file_template_nn = 'Report_PRBS31_test_DNN_{}dBm.txt'
file_template_volterra = 'Report_PRBS31_test_Volterra_{}dBm.txt'
file_template_dfe = 'Report_PRBS31_test_DFE_{}dBm.txt'

# ==============================================================================
# 2. 自动路径构建 (根据上方配置自动生成)
# ==============================================================================

# 构建该实验的根目录: D:\paperwork\Experiment_Data\20Gsyms_30km
current_exp_dir = os.path.join(base_root, target_scenario)

# 路径1: NN (DNN) 实验数据文件夹路径
path_nn = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'NN')

# 路径2: DFE 实验数据文件夹路径
path_dfe = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'DFE')

# 路径3: Volterra 实验数据文件夹路径
path_volterra = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'Volterra')

# 自动生成输出文件名: results_20Gsyms_30km.md
output_filename = f'results_{target_scenario}.md'
# 路径4: 结果输出文件的完整路径 (包含文件名.md)
output_file_path = os.path.join(current_exp_dir, output_filename)

print(f"当前处理场景: {target_scenario}")
print(f"输出文件路径: {output_file_path}")
print("-" * 30)

# ==============================================================================
# 3. 详细参数映射配置
# ==============================================================================

# 定义实验组与文件名的映射关系
# key: 脚本内部标识, value: (文件夹路径, 文件名模板, 输出变量后缀)
experiments = {
    'NN': (path_nn, file_template_nn, 'dnn'),
    'Volterra': (path_volterra, file_template_volterra, 'volterra'),
    'DFE': (path_dfe, file_template_dfe, 'dfe')
}

# 定义需要提取的指标及其正则表达式
# Regex 说明:
# ([0-9.eE+-]+) 捕获浮点数（包括科学计数法）
# (\d+) 捕获整数
metrics_config = [
    {
        'name': 'PCM SQNR',
        'var_prefix': 'sqnr',
        'unit': 'Data (dB)',  # <--- 已恢复单位
        'pattern': r'PCM SQNR\s*:\s*([0-9.eE+-]+)',
        'type': float
    },
    {
        'name': 'rms EVM',
        'var_prefix': 'evm',
        'unit': 'Data (%)',   # <--- 已恢复单位
        'pattern': r'rms EVM\s*:\s*([0-9.eE+-]+)',
        'type': float
    },
    {
        'name': 'SER (PAM4)',
        'var_prefix': 'ser',
        'unit': 'Data',       # <--- 已恢复单位
        'pattern': r'SER \(PAM4\)\s*:\s*([0-9.eE+-]+)',
        'type': float
    },
    {
        'name': 'BER (16-QAM)',
        'var_prefix': 'ber',
        'unit': 'Data',       # <--- 已恢复单位
        'pattern': r'BER \(16-QAM\)\s*:\s*([0-9.eE+-]+)',
        'type': float
    },
    {
        'name': 'Total Errors',
        'var_prefix': 'errors',
        'unit': 'Data (Count)', # <--- 已恢复单位
        'pattern': r'Total Errors\s*:\s*(\d+)\s*/',  # 匹配斜杠前的数字
        'type': int
    }
]

# ==============================================================================
# 4. 数据提取逻辑
# ==============================================================================

def parse_file(filepath, patterns):
    """读取文件并根据正则提取所有指标"""
    data = {}
    if not os.path.exists(filepath):
        # 调试用，如果文件找不到可以打开下面的注释
        # print(f"警告: 文件不存在 -> {filepath}")
        return None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    for metric in patterns:
        match = re.search(metric['pattern'], content)
        if match:
            val_str = match.group(1)
            try:
                val = metric['type'](val_str)
                data[metric['name']] = val
            except ValueError:
                data[metric['name']] = 0  # 转换失败归0
        else:
            data[metric['name']] = 0  # 未匹配到归0

    return data


# 存储所有结果的字典容器
# 结构: all_data[metric_name][exp_type] = [val1, val2, ...]
all_data = {m['name']: {exp: [] for exp in experiments} for m in metrics_config}

print("开始提取数据...")

for x in x_values:
    # print(f"正在处理 X = {x} dBm ...")
    for exp_type, (folder_path, filename_tmpl, _) in experiments.items():
        # 构建完整文件路径
        filename = filename_tmpl.format(x)
        full_path = os.path.join(folder_path, filename)

        # 提取数据
        results = parse_file(full_path, metrics_config)

        # 存入列表
        if results:
            for m in metrics_config:
                all_data[m['name']][exp_type].append(results[m['name']])
        else:
            # 如果文件不存在，填入0或者NaN保持数组长度一致
            for m in metrics_config:
                all_data[m['name']][exp_type].append(0)

# ==============================================================================
# 5. 生成 Markdown 输出文件
# ==============================================================================

print(f"正在生成 Markdown 报告: {output_file_path}")

with open(output_file_path, 'w', encoding='utf-8') as f:
    # 写入 Markdown 头部信息
    f.write(f"## 实验数据提取日志: {target_scenario}\n\n")
    # 获取当前详细时间 (年-月-日 时:分:秒)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 写入 Markdown 头部引用信息
    f.write(f"> **范围:** {x_values[0]} dBm 到 {x_values[-1]} dBm\n")
    f.write(f"> **生成时间:** {current_time}\n\n")

    # 循环写入每个指标的块
    for metric in metrics_config:
        m_name = metric['name']
        m_prefix = metric['var_prefix']
        m_unit = metric['unit'] # 获取单位

        # 写入二级标题 (包含单位)
        f.write(f"## {m_name} - {m_unit}\n\n")

        # 写入代码块开始标记
        f.write("```python\n")

        # 按照 NN, Volterra, DFE 的顺序写入数据
        for exp_type in ['NN', 'Volterra', 'DFE']:
            _, _, var_suffix = experiments[exp_type]
            values_list = all_data[m_name][exp_type]

            # 格式化数值
            if metric['type'] == int:
                # 整数不带小数点
                vals_str = ", ".join([str(v) for v in values_list])
            else:
                # 浮点数保留4位
                vals_str = ", ".join([f"{v:.4f}" for v in values_list])

            # 写入变量赋值语句 (生成通用变量名，方便复制到画图脚本)
            var_name = f"{m_prefix}_{var_suffix}"
            f.write(f"{var_name} = np.array([\n    {vals_str}\n])\n")

        # 写入代码块结束标记
        f.write("```\n\n")
        f.write("---\n\n")  # 添加分隔线

print("完成！请查看生成的 .md 文件。")