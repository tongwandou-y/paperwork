import os
import re
from datetime import datetime  # 引入时间模块

# ==============================================================================
# 1. 全局配置区域
# ==============================================================================

# [基础路径]
base_root = r'D:\paperwork\Experiment_Data'

# [核心变量]: 当前要处理的实验场景名称
target_scenario = '30Gsyms_20km'

# [参数范围]: X dBm 变化的范围
x_values = list(range(-27, -14))

# [文件名模板]
file_template_nn = 'Report_PRBS31_test_DNN_{}dBm.txt'
file_template_volterra = 'Report_PRBS31_test_Volterra_{}dBm.txt'
file_template_dfe = 'Report_PRBS31_test_DFE_{}dBm.txt'

# ==============================================================================
# 2. 自动路径构建
# ==============================================================================

current_exp_dir = os.path.join(base_root, target_scenario)
path_nn = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'NN')
path_dfe = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'DFE')
path_volterra = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'Volterra')

output_filename = f'results_{target_scenario}.md'
output_file_path = os.path.join(current_exp_dir, output_filename)

print(f"当前处理场景: {target_scenario}")
print(f"输出文件路径: {output_file_path}")
print("-" * 30)

# ==============================================================================
# 3. 详细参数映射配置 (深度修复正则)
# ==============================================================================

experiments = {
    'NN': (path_nn, file_template_nn, 'dnn'),
    'Volterra': (path_volterra, file_template_volterra, 'volterra'),
    'DFE': (path_dfe, file_template_dfe, 'dfe')
}

metrics_config = [
    # 1. SER (PAM4)
    {
        'name': 'SER (PAM4)',
        'var_prefix': 'ser',
        'unit': 'Data',
        'pattern': r'SER \(PAM4\)\s*:\s*([0-9.eE+-]+)',
        'type': 'raw_str',
        'default': '1.0'
    },
    # 2. BER (PAM4) - 【修复】增加防误触机制
    {
        'name': 'BER (PAM4)',
        'var_prefix': 'ber_pam4',
        'unit': 'Data',
        # 逻辑：查找 "BER (PAM4)"，但前面不能有 "16-QAM "，且 BER 前面应该是单词边界
        'pattern': r'(?<!16-QAM )(?<!\w)BER \(PAM4\)\s*:\s*([0-9.eE+-]+)',
        'type': 'raw_str',
        'default': '1.0'
    },
    # 3. PAM4 Total Errors
    {
        'name': 'PAM4 Total Errors',
        'var_prefix': 'errors_pam4',
        'unit': 'Data (String)',
        'pattern': r'PAM4 Total Errors\s*:\s*([0-9]+\s*/\s*[0-9]+)',
        'type': 'raw_str',
        'default': '0'
    },
    # 4. PCM SQNR
    {
        'name': 'PCM SQNR',
        'var_prefix': 'sqnr',
        'unit': 'Data (dB)',
        'pattern': r'PCM SQNR\s*:\s*([0-9.eE+-]+)',
        'type': float,
        'default': 0.0
    },
    # 5. rms EVM
    {
        'name': 'rms EVM',
        'var_prefix': 'evm',
        'unit': 'Data (%)',
        'pattern': r'rms EVM\s*:\s*([0-9.eE+-]+)',
        'type': float,
        'default': 0.0
    },
    # 6. BER (16-QAM) - 【修复】增加防误触机制
    {
        'name': 'BER (16-QAM)',
        'var_prefix': 'ber',
        'unit': 'Data',
        # 逻辑：查找 "BER (16-QAM)"，但前面不能有 "PAM4 "
        'pattern': r'(?<!PAM4 )(?<!\w)BER \(16-QAM\)\s*:\s*([0-9.eE+-]+)',
        'type': 'raw_str',
        'default': '1.0'
    },
    # 7. Total Errors (保持之前的修复)
    {
        'name': 'Total Errors',
        'var_prefix': 'errors',
        'unit': 'Data (String)',
        # 逻辑：前面不能有 "PAM4 "
        'pattern': r'(?<!PAM4 )Total Errors\s*:\s*([0-9]+\s*/\s*[0-9]+)',
        'type': 'raw_str',
        'default': '0'
    }
]

# ==============================================================================
# 4. 数据提取逻辑
# ==============================================================================

def parse_file(filepath, patterns):
    """读取文件并根据正则提取所有指标"""
    data = {}
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    for metric in patterns:
        match = re.search(metric['pattern'], content)
        if match:
            val_str = match.group(1)
            # 如果是 float 类型，进行转换；如果是 raw_str，直接保留原始字符串
            if metric['type'] == float:
                try:
                    data[metric['name']] = float(val_str)
                except ValueError:
                    data[metric['name']] = metric['default']
            else:
                # 'raw_str' 直接存字符串
                data[metric['name']] = val_str
        else:
            # 未匹配到时的默认值
            data[metric['name']] = metric['default']

    return data


# 存储所有结果
all_data = {m['name']: {exp: [] for exp in experiments} for m in metrics_config}

print("开始提取数据...")

for x in x_values:
    for exp_type, (folder_path, filename_tmpl, _) in experiments.items():
        filename = filename_tmpl.format(x)
        full_path = os.path.join(folder_path, filename)
        results = parse_file(full_path, metrics_config)

        if results:
            for m in metrics_config:
                all_data[m['name']][exp_type].append(results[m['name']])
        else:
            # 文件不存在，填默认值
            for m in metrics_config:
                all_data[m['name']][exp_type].append(m['default'])

# ==============================================================================
# 5. 生成 Markdown 输出文件
# ==============================================================================

print(f"正在生成 Markdown 报告: {output_file_path}")

with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(f"## 实验数据提取日志: {target_scenario}\n\n")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"> **范围:** {x_values[0]} dBm 到 {x_values[-1]} dBm\n")
    f.write(f"> **生成时间:** {current_time}\n\n")

    for metric in metrics_config:
        m_name = metric['name']
        m_prefix = metric['var_prefix']
        m_unit = metric['unit']

        f.write(f"## {m_name} - {m_unit}\n\n")
        f.write("```python\n")

        for exp_type in ['NN', 'Volterra', 'DFE']:
            _, _, var_suffix = experiments[exp_type]
            values_list = all_data[m_name][exp_type]

            # 格式化逻辑
            if metric['type'] == float:
                # 浮点数：保留4位小数
                vals_str = ", ".join([f"{v:.4f}" for v in values_list])
            else:
                # raw_str：直接输出字符串，不加引号
                # 这样 10 / 100 就会在 python 代码中直接作为除法运算
                vals_str = ", ".join([str(v) for v in values_list])

            var_name = f"{m_prefix}_{var_suffix}"
            f.write(f"{var_name} = np.array([\n    {vals_str}\n])\n")

        f.write("```\n\n")
        f.write("---\n\n")

print("完成！请查看生成的 .md 文件。")