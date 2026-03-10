import os
import re
from datetime import datetime

# ==============================================================================
# 1. 全局配置区域
# ==============================================================================

base_root = r'D:\paperwork\Experiment_Data'
target_scenario = '20Gsyms_20km'
x_values = list(range(-24, -14))

file_template_dnn = 'Report_PRBS31_test_DNN_{}dBm.txt'
file_template_volterra = 'Report_PRBS31_test_Volterra_{}dBm.txt'
file_template_dfe = 'Report_PRBS31_test_DFE_{}dBm.txt'

# ==============================================================================
# 2. 自动路径构建与映射配置
# ==============================================================================

current_exp_dir = os.path.join(base_root, target_scenario)

path_pam_only = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'DNN', 'ablation_pam_only')
path_pam_pcm = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'DNN', 'ablation_pam_pcm')
path_volterra = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'Volterra')
path_dfe = os.path.join(current_exp_dir, 'RX_Matlab_Result_Reports_txt', 'DFE')

output_filename = f'results_{target_scenario}.md'
output_file_path = os.path.join(current_exp_dir, output_filename)

print(f"当前处理场景: {target_scenario}")
print(f"输出文件路径: {output_file_path}")
print("-" * 30)

experiments = {
    'pam_only': (path_pam_only, file_template_dnn, 'pam_only'),
    'pam_pcm': (path_pam_pcm, file_template_dnn, 'pam_pcm'),
    'Volterra': (path_volterra, file_template_volterra, 'Volterra'),
    'DFE': (path_dfe, file_template_dfe, 'DFE')
}

# ==============================================================================
# 3. 指标正则配置
# ==============================================================================
metrics_config = [
    {
        'name': 'SER (PAM4)',
        'var_prefix': 'ser',
        'unit': 'Data',
        'pattern': r'SER \(PAM4\)\s*:\s*([^\s]+)',
        'type': 'raw_str',
        'default': '1.0'
    },
    {
        'name': 'BER (PAM4)',
        'var_prefix': 'ber_pam4',
        'unit': 'Data',
        'pattern': r'(?<!16-QAM )(?<!\w)BER \(PAM4\)\s*:\s*([^\s]+)',
        'type': 'raw_str',
        'default': '1.0'
    },
    {
        'name': 'PAM4 Total Errors',
        'var_prefix': 'errors_pam4',
        'unit': 'Data (String)',
        'pattern': r'PAM4 Total Errors\s*:\s*([a-zA-Z0-9.-]+\s*/\s*[a-zA-Z0-9.-]+)',
        'type': 'raw_str',
        'default': '0'
    },
    {
        'name': 'PCM SQNR',
        'var_prefix': 'sqnr',
        'unit': 'Data (dB)',
        'pattern': r'PCM SQNR\s*:\s*([^\s]+)',
        'type': float,
        'default': 0.0
    },
    {
        'name': 'rms EVM',
        'var_prefix': 'evm',
        'unit': 'Data (%)',
        'pattern': r'rms EVM\s*:\s*([^\s]+)',
        'type': float,
        'default': 0.0
    },
    {
        'name': 'BER (16-QAM)',
        'var_prefix': 'ber',
        'unit': 'Data',
        'pattern': r'(?<!PAM4 )(?<!\w)BER \(16-QAM\)\s*:\s*([^\s]+)',
        'type': 'raw_str',
        'default': '1.0'
    },
    {
        'name': 'Total Errors',
        'var_prefix': 'errors',
        'unit': 'Data (String)',
        'pattern': r'(?<!PAM4 )Total Errors\s*:\s*([a-zA-Z0-9.-]+\s*/\s*[a-zA-Z0-9.-]+)',
        'type': 'raw_str',
        'default': '0'
    }
]

# ==============================================================================
# 4. 数据提取逻辑
# ==============================================================================

def parse_file(filepath, patterns):
    data = {}

    if not os.path.exists(filepath):
        print(f"\n[致命警告] 无法找到文件 (触发了全盘默认值): \n --> {filepath}")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
            content = f.read()

        if not content.strip():
            print(f"\n[致命警告] 文件存在，但内容为空: \n --> {filepath}")
            return None

    except Exception as e:
        print(f"\n[致命警告] 尝试读取文件时崩溃 ({e}): \n --> {filepath}")
        return None

    for metric in patterns:
        match = re.search(metric['pattern'], content)
        if match:
            val_str = match.group(1).strip()
            if metric['type'] == float:
                try:
                    data[metric['name']] = float(val_str)
                except ValueError:
                    data[metric['name']] = metric['default']
            else:
                data[metric['name']] = val_str
        else:
            print(f"  [细微警告] 文件中未能匹配到指标 '{metric['name']}': {os.path.basename(filepath)}")
            data[metric['name']] = metric['default']

    return data


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
            for m in metrics_config:
                all_data[m['name']][exp_type].append(m['default'])

# ==============================================================================
# 5. 生成 Markdown 输出文件
# ==============================================================================

print(f"\n正在生成 Markdown 报告: {output_file_path}")

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

        # [已修复]：改为动态读取字典 keys，杜绝硬编码导致的 KeyError
        for exp_type in experiments.keys():
            _, _, var_suffix = experiments[exp_type]
            values_list = all_data[m_name][exp_type]

            if metric['type'] == float:
                vals_str = ", ".join([f"{float(v):.4f}" for v in values_list])
            else:
                vals_str = ", ".join([str(v) for v in values_list])

            var_name = f"{m_prefix}_{var_suffix}"
            f.write(f"{var_name} = np.array([\n    {vals_str}\n])\n")

        f.write("```\n\n")
        f.write("---\n\n")

print("完成！请查看控制台警告日志，并检查生成的 .md 文件。")