# DNNV3-0/configs.py

import ml_collections
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _sanitize_tag(tag):
    """将实验标签标准化为安全目录名。"""
    return str(tag).strip().lower().replace(' ', '_').replace('-', '_')


def _build_experiment_tag(model_name, profile_name=None):
    """
    生成实验子目录标签（用于输出与日志分组）。
    优先使用 batch_runner 传入的 profile.name，确保与实验命名一致。
    """
    if profile_name:
        return _sanitize_tag(profile_name)

    model_key = str(model_name).upper()
    if model_key == 'DNN':
        return 'baseline_dnn'
    if model_key == 'SH_DNN':
        return 'sh_dnn'
    if model_key == 'PP_CDNN':
        return 'pp_cdnn'
    return _sanitize_tag(model_name)


def _infer_data_mode(model_file, model_class):
    """
    根据模型名自动推断数据流模式:
    - 'block': 块级输入/输出（如 PP_CDNN）
    - 'symbol': 传统逐符号输入/输出（如 DNN）
    """
    model_key = f"{model_file}|{model_class}".lower()
    if ('pp_cdnn' in model_key) or ('pp-cdnn' in model_key):
        return 'block'
    if ('sh_dnn' in model_key) or ('sh-dnn' in model_key):
        return 'block'
    return 'symbol'


def get_config(target_power=None, ablation_profile=None):
    """
    获取实验配置参数
    :param target_power: 指定接收光功率 (dBm)。
                         - 如果传入数值: 覆盖默认值 (用于 batch_runner 批量运行)
                         - 如果为 None: 使用默认值 (用于 train.py 单独调试)
    """
    config = ml_collections.ConfigDict()

    # =========================================================
    # 0. 对比测评模式（公平性口径）
    # =========================================================
    # strict_uniform:
    #   - 统一选模口径（全部按 pam_loss）
    #   - 统一多任务超参默认值（不过度为 PP_CDNN 定向增强）
    # best_effort:
    #   - 允许按模型定向优化超参（突出各自可达上限）
    comparison_mode = 'strict_uniform'
    if ablation_profile is not None and 'comparison_mode' in ablation_profile:
        comparison_mode = str(ablation_profile['comparison_mode']).strip().lower()
    if comparison_mode not in {'strict_uniform', 'best_effort'}:
        raise ValueError(f"未知 comparison_mode: {comparison_mode}")
    config.comparison_mode = comparison_mode

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

    # --------------------------------------------------------------
    # 之后切换模型只需要修改 config.model_name
    # 模型文件位于 NN_NLE_Code/models/model/
    # --------------------------------------------------------------

    # 模型选择（你只需改这一个变量）
    # 内置: 'PP_CDNN' / 'DNN'
    # 也支持自定义模型名（默认映射为 <name>.py + <name> 类）
    config.model_name = 'PP_CDNN'
    if ablation_profile is not None and 'model_name' in ablation_profile:
        config.model_name = str(ablation_profile['model_name'])
    model_registry = {
        'PP_CDNN': ('PP_CDNN.py', 'PP_CDNN'),
        'DNN': ('DNN.py', 'DNN'),
        'SH_DNN': ('SH_DNN.py', 'SH_DNN'),
    }
    config.model_file, config.model_class = model_registry.get(
        config.model_name,
        (f"{config.model_name}.py", config.model_name)
    )
    # 若需要自动发现类名，可改为 None
    # config.model_class = None
    # 用于实验名与输出文件命名的模型标签（默认取文件名去后缀）
    config.model_type = os.path.splitext(config.model_file)[0]
    # 自动根据模型推断数据流模式，确保用户只改模型名即可切换全流程
    config.data_mode = _infer_data_mode(config.model_file, config.model_class)

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
    # PAM输出激活：建议使用 tanh 与 MATLAB 的阈值判决口径保持一致
    # 可选: 'tanh' | 'identity'
    config.pam_output_activation = 'tanh'
    # 推理阶段是否额外 clip 到 [-1, 1]，通常作为安全兜底
    config.inference_clip_pam = True

    ## =========================================================
    ## 3.1 任务定义（按模型强绑定）
    ## =========================================================

    # 由模型定义任务
    # DNN    : 传统黑盒，symbol->symbol，任务= pam
    # SH_DNN : 块级单头，任务= pam
    # PP_CDNN: 块级双头，任务= pam_pcm
    model_name_upper = str(config.model_name).upper()
    if model_name_upper in {'DNN', 'SH_DNN'}:
        config.task_type = 'pam'
    elif model_name_upper == 'PP_CDNN':
        config.task_type = 'pam_pcm'
    else:
        # 对未知自定义模型采用保守策略：symbol模式->pam，block模式->pam_pcm
        config.task_type = 'pam_pcm' if config.data_mode == 'block' else 'pam'

    # 损失权重
    config.loss_alpha = 1.0  # L_pcm 权重
    # 最优模型判据: 'pam_loss' | 'total_loss' | 'hybrid'
    # - pam_loss: 优先SER/BER
    # - hybrid  : 在PAM与PCM之间折中，通常更利于SQNR/EVM
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

    # 辅助任务限幅：限制 PCM 贡献不超过 PAM项的一定比例（自适应，不是分段）
    config.aux_to_pam_max_ratio = 0.80

    # 在 PAM+PCM 任务中，按 comparison_mode 选择策略：
    # - strict_uniform: 保持统一口径，不对 PP_CDNN 做额外“特供”强化
    # - best_effort: 允许对 PP_CDNN 做定向优化，追求模型上限
    if config.task_type == 'pam_pcm':
        if config.comparison_mode == 'best_effort':
            config.loss_alpha = 2.0
            config.pam_priority_factor = 1.00
            config.aux_to_pam_max_ratio = 1.50
            config.loss_ema_momentum = 0.95
            config.best_model_metric = 'hybrid'
        else:
            config.loss_alpha = 1.0
            config.pam_priority_factor = 1.20
            config.aux_to_pam_max_ratio = 0.80
            config.loss_ema_momentum = 0.98
            config.best_model_metric = 'pam_loss'

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

    # 实验标签：作为现有保存根路径的下级目录，不破坏原有目录体系
    # 规则：优先使用 batch_runner 里的 profile.name；否则按模型/开关自动生成。
    profile_name = None
    if ablation_profile is not None:
        profile_name = ablation_profile.get('name', None)
    config.ablation_tag = _build_experiment_tag(
        model_name=config.model_name,
        profile_name=profile_name
    )

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

