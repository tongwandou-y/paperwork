# DNNV3-0/configs.py

import ml_collections
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _build_ablation_tag(use_pam, use_pcm):
    """根据损失开关生成消融标签。"""
    if use_pam and (not use_pcm):
        return 'ablation_pam_only'
    if use_pam and use_pcm:
        return 'ablation_pam_pcm'
    return f'ablation_custom_p{int(use_pam)}_m{int(use_pcm)}'


def get_config(target_power=None, ablation_profile=None):
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
    config.fiber_length = '20km'

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
    ## 3. 训练超参
    ## =========================================================

    config.device = device
    config.batch_size = 64
    config.epoch = 300           # 增大训练轮次，给CosineAnnealing更多空间
    config.learn_rate = 0.001    # Adam初始学习率
    config.init_type = 'orthogonal'  # 权重初始化方法

    ## =========================================================
    ## 3.1 消融与损失权重开关
    ## =========================================================
    # 是否启用各损失项
    config.use_loss_pam = True
    config.use_loss_pcm = True
    if ablation_profile is not None:
        config.use_loss_pam = bool(ablation_profile.get('use_loss_pam', config.use_loss_pam))
        config.use_loss_pcm = bool(ablation_profile.get('use_loss_pcm', config.use_loss_pcm))

    # 一致性损失在当前版本已移除（后续改为级联双头再验证）
    config.use_loss_cons = False

    # 损失权重
    config.loss_alpha = 1.0  # L_pcm 权重
    # 最优模型判据: 'pam_loss' | 'total_loss' | 'hybrid'
    # 建议通信性能优先时使用 pam_loss
    config.best_model_metric = 'pam_loss'

    ## =========================================================
    ## 3.2 自适应损失权重与断点控制
    ## =========================================================
    # 使用可学习的不确定性权重自动平衡 L_pam / L_pcm
    config.loss_weight_strategy = 'auto_uncertainty'

    # 自适应权重的稳定系数（越小越稳健，建议 0.01~0.2）
    config.auto_uncertainty_reg = 0.05

    # 多任务稳定性增强：对各loss做EMA尺度归一化，减少量纲差异带来的权重抖动
    config.use_loss_ema_normalization = True
    config.loss_ema_momentum = 0.98
    config.loss_norm_eps = 1e-8

    # 主任务优先：对PAM项给予固定优先系数，确保SER/BER目标不被辅助任务稀释
    config.pam_priority_factor = 1.20

    # 辅助任务限幅：限制 (PCM+CONS) 贡献不超过 PAM项的一定比例（自适应，不是分段）
    config.aux_to_pam_max_ratio = 0.80

    ## =========================================================
    ## 2. 目录与路径设置 (Directories & Paths)
    ## =========================================================
    # 基础目录配置
    # 输入目录: 存放 MATLAB Part 1 生成的 .mat 标签和数据
    input_dir = r'E:\yinshibo\paperwork\Experiment_Data\20Gsyms_20km\NN_Input_Data_mat'
    # 输出目录: 存放 Python 处理完供 MATLAB Part 2 使用的数据
    output_root_dir = r'E:\yinshibo\paperwork\Experiment_Data\20Gsyms_20km\NN_Output_Data_mat'
    # Loss 日志存放目录
    loss_log_root_dir = r'E:\yinshibo\paperwork\Experiment_Data\20Gsyms_20km\NN_Loss_Log_txt'

    # 消融标签：作为现有保存根路径的下级目录，不破坏原有目录体系
    config.ablation_tag = _build_ablation_tag(config.use_loss_pam, config.use_loss_pcm)

    output_dir = os.path.join(output_root_dir, config.ablation_tag)
    loss_log_dir = os.path.join(loss_log_root_dir, config.ablation_tag)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(loss_log_dir, exist_ok=True)

    # --- 文件路径自动构造 ---
    # 1. 训练输入文件 (给 train.py 用)
    config.train_data_file = os.path.join(input_dir, f'Data_For_NN_{train_prbs}_train_{received_optical_power}.mat')
    # 2. 测试输入文件 (给 run_equalization.py 用)
    config.test_data_file = os.path.join(input_dir, f'Data_For_NN_{test_prbs}_test_{received_optical_power}.mat')
    # 3. 测试输出文件 (给 run_equalization.py 保存结果用)
    config.test_output_file = os.path.join(output_dir, f'NN_Output_test_{config.model_type}_{received_optical_power}.mat')

    # 4. 实验名称（加入消融标签，保证checkpoint互不覆盖）
    config.experiment_name = f'16QAM-{config.fiber_length}-{config.model_type}-{config.ablation_tag}_{received_optical_power}dBm'
    # 5. Loss 日志文件完整路径
    config.loss_log_file = os.path.join(loss_log_dir, f'Loss_Log_{config.experiment_name}.txt')

    # True: 忽略已有 checkpoint，从头开始训练
    # 由于模型结构已改变，必须从头开始训练
    config.force_restart = True

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
    # 重要限制条件:
    #   - mod(config.seq_len * Mm, quant) 必须满足帧同步对齐要求
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

