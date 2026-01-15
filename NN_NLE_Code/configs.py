# DNNV3-0/configs.py

import ml_collections
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_config(target_power=None):
    """
    获取实验配置参数
    :param target_power: 指定接收光功率 (dBm)。
                         - 如果传入数值: 覆盖默认值 (用于 batch_runner 批量运行)
                         - 如果为 None: 使用默认值 (用于 train.py 单独调试)
    """
    config = ml_collections.ConfigDict()

    ## =========================================================
    ## 1. 基础物理与模型设置 (Basic Settings)
    ## =========================================================

    # 光纤长度设置
    config.fiber_length = '30km'

    # 接收光功率 (dBm)
    # 如果传入了 target_power，就用传入的值；否则用默认值 (例如 -20)
    # 这样既支持脚本批量跑，也支持你单独运行 train.py
    if target_power is not None:
        received_optical_power = target_power
    else:
        received_optical_power = -20  # 默认值 (单独调试时使用)

    # 设置量化比特数
    quant_bits = 8
    # 存入 config 对象
    config.quant = quant_bits
    # 模型类型选择: 'DNN' 或 'CNN'
    config.model_type = 'DNN'

    # PRBS 序列名称
    train_prbs = 'PRBS23'
    test_prbs = 'PRBS31'

    ## =========================================================
    ## 2. 目录与路径设置 (Directories & Paths)
    ## =========================================================

    # 基础目录配置
    # 输入目录: 存放 MATLAB Part 1 生成的 .mat 标签和数据
    input_dir = r'E:\yinshibo\paperwork\Experiment_Data\20Gsyms_30km\NN_Input_Data_mat'
    # 输出目录: 存放 Python 处理完供 MATLAB Part 2 使用的数据
    output_dir = r'E:\yinshibo\paperwork\Experiment_Data\20Gsyms_30km\NN_Output_Data_mat'
    # Loss 日志存放目录
    loss_log_dir = r'E:\yinshibo\paperwork\Experiment_Data\20Gsyms_30km\NN_Loss_Log_txt'

    # 自动创建不存在的目录
    if not os.path.exists(loss_log_dir):
        os.makedirs(loss_log_dir)

    # --- 文件路径自动构造 ---

    # 1. 训练输入文件 (给 train.py 用)
    # 格式示例: Data_For_NN_PRBS15_8bit_train_-18.mat
    config.train_data_file = os.path.join(input_dir, f'Data_For_NN_{train_prbs}_{quant_bits}bit_train_{received_optical_power}.mat')

    # 2. 测试输入文件 (给 run_equalization.py 用)
    # 格式示例: Data_For_NN_PRBS23_8bit_test_-18.mat
    config.test_data_file = os.path.join(input_dir, f'Data_For_NN_{test_prbs}_{quant_bits}bit_test_{received_optical_power}.mat')

    # 3. 测试输出文件 (给 run_equalization.py 保存结果用)
    # 格式示例: NN_Output_test_DNN_8bit_-18.mat 或 NN_Output_test_CNN_8bit_-18.mat
    config.test_output_file = os.path.join(output_dir,
                                           f'NN_Output_test_{config.model_type}_{quant_bits}bit_{received_optical_power}.mat')

    # 4. 实验名称 (用于 Checkpoint 文件夹命名)
    # 动态组合：光纤长度 + 模型类型 + Quant + 功率 (避免不同实验互相覆盖)
    config.experiment_name = f'16QAM-{config.fiber_length}-{config.model_type}-{quant_bits}bit_{received_optical_power}dBm'

    # 5. Loss 日志文件完整路径
    # 格式示例: Loss_Log_16QAM-15km-CNN-8bit_-20dBm.txt
    config.loss_log_file = os.path.join(loss_log_dir, f'Loss_Log_{config.experiment_name}.txt')

    ## =========================================================
    ## 3. 训练超参
    ## =========================================================

    config.device = device
    config.batch_size = 64
    config.epoch = 100
    config.learn_rate = 0.001  # Adam优化器，学习率建议设小
    config.init_type = 'orthogonal'  # 权重初始化方法

    ## =========================================================
    ## 4. 均衡器核心参数 (滑动窗口半径)
    ## =========================================================

    # seq_len: 滑动窗口的半宽
    # ---------------------------------------------------------
    # 含义说明:
    #   - 总窗口大小 (TAPS) = 2 * seq_len + 1
    #   - 例如: seq_len=9 -> 总窗口=19。这样中心抽头可以精确对应当前符号。
    #   - 对应 DNN 的输入维度 input_dim。
    #
    # 【跑通删】重要限制条件:
    # 【跑通删】  - mod(config.seq_len * Mm, quant) 必须满足帧同步对齐要求
    # ---------------------------------------------------------
    config.seq_len = 8

    # 注意：d_model 在原代码中是OFDM概念，这里不再使用。
    # 模型的输入维度现在完全由 (2 * config.seq_len + 1) 决定。

    return config


# 为了兼容旧的 train.py/test.py 结构，我们仍然保留这个字典
Configs = {
    'drof_dnn_train': get_config(),
    'drof_dnn_test': get_config()  # 训练和测试使用相同的配置
}

