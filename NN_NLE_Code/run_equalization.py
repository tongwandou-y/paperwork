# DNNV3-0/run_equalization.py
#
# 目的：加载训练好的模型，对 'Data_For_NN_PRBS31_test.mat' 中的测试数据进行均衡，
#       并保存为 'NN_Output_test.mat' 以供 MATLAB (Rx_Part2) 使用。

import torch
import scipy.io as spio
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

import configs as cfg
from util.model_loader import build_model_from_config
from util.utils import load_checkpoint  # 导入加载检查点的函数


def create_block_window_for_test(signal, taps, block_size):
    """
    块级对齐的测试特征构建:
    - 以物理块为步长生成窗口
    输入:
        signal: 失真的信号序列 (numpy array)
        taps:   窗口大小 (均衡器抽头数)
        block_size: 物理块大小 G
    输出:
        X: 特征矩阵 (样本数, taps)
    """
    X = []

    # 确保信号长度足够
    if len(signal) < taps:
        print(f"警告：信号长度 ({len(signal)}) 小于抽头数 ({taps})，无法创建数据集。")
        return np.array(X)

    seq_len = taps // 2
    num_blocks = len(signal) // block_size
    for b in range(num_blocks):
        # 与训练一致：以物理块起点(MSB)作为对齐中心
        center = b * block_size
        start = center - seq_len
        end = center + seq_len + 1
        if start < 0 or end > len(signal):
            continue
        window = signal[start:end]
        X.append(window)

    return np.array(X)


def run_equalization(configs):
    # 使用配置动态显示当前模型类型
    print(f"--- 开始执行 {configs.model_type} 均衡（测试）---")
    device = configs.device

    # --- 1. 加载配置 ---
    # 窗口总长度 (taps)
    taps = 2 * configs.seq_len + 1

    # MATLAB数据文件路径
    mat_file_path = configs.test_data_file
    # 最终输出文件路径
    output_file = configs.test_output_file

    # 训练好的模型权重目录
    ckpt_dir = os.path.join('output', configs.experiment_name, 'checkpoint')

    # 检查并创建输出目录 (D:\paperwork\Experiment_Data\16Gsyms\NN_Output_Data_mat)
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"输出目录不存在，已自动创建: {output_dir}")

    # --- 2. 加载模型和训练好的权重 ---
    model, model_class_name = build_model_from_config(configs)
    model = model.to(device)
    print(f"加载模型: file={configs.model_file}, class={model_class_name}...")

    try:
        # 加载最新的检查点
        # [修改] 使用 load_best=True 自动加载训练过程中 Loss 最低的 'best_model.ckpt'
        ckpt, ckpt_path = load_checkpoint(ckpt_dir, load_best=True)

        model.load_state_dict(ckpt['model'])
        print(f"成功加载最佳模型权重: {ckpt_path}")
        print(f"最佳轮次 (Epoch): {ckpt.get('epoch', 'Unknown')}")
        print(f"最低损失 (Loss): {ckpt.get('loss', 'Unknown')}")

    except Exception as e:
        print(f"错误: 无法加载模型权重。请确保 'train.py' 已成功运行，")
        print(f"并且权重保存在 '{ckpt_dir}' 目录中。")
        print(f"错误详情: {e}")
        return

    # 切换到评估模式 (这会关闭BatchNorm和Dropout等)
    model.eval()

    # --- 3. 加载并预处理测试数据 ---
    print(f"从 '{mat_file_path}' 加载测试数据...")

    if not os.path.exists(mat_file_path):
        print(f"错误: 找不到输入文件 {mat_file_path}")
        print("请检查 MATLAB Part 1 是否已生成对应功率的测试数据。")
        return

    mat_contents = spio.loadmat(mat_file_path)

    try:
        # 加载 MATLAB Rx_Part1_Generate_TEST 保存的测试输入信号
        distorted_signal = mat_contents['dnn_test_input'].flatten()
    except KeyError:
        print(f"错误: 在 '{mat_file_path}' 中未找到 'dnn_test_input'。")
        return

    # ====================== Z-score 归一化 ======================
    # 【关键修改】训练时在 MatDatasetBlock 中对输入做了 Z-score 标准化，
    # 测试时必须做同样的处理，否则数据分布不一致，模型输出无意义。
    feat_mean = np.mean(distorted_signal)
    feat_std = np.std(distorted_signal)
    distorted_signal = (distorted_signal - feat_mean) / (feat_std + 1e-8)
    print(f"已对测试数据进行 Z-score 归一化 (mean={feat_mean:.4f}, std={feat_std:.4f})")

    # 物理块大小 G = quant / Mm (PAM4 -> Mm=2)
    block_size = configs.quant // 2
    print(f"为测试数据创建块级窗口 (Taps: {taps}, G: {block_size})...")
    test_features = create_block_window_for_test(distorted_signal, taps, block_size)

    if test_features.shape[0] == 0:
        print("错误：未能创建测试特征。")
        return

    # 转换为 PyTorch 张量
    test_features_tensor = torch.from_numpy(test_features.astype(np.float32))

    # --- 4. 执行均衡 (模型推理) ---
    print("开始执行均衡...")

    # 将所有测试特征一次性放到GPU (或CPU)
    test_features_tensor = test_features_tensor.to(device)

    equalized_output = None
    # 禁用梯度计算以节省内存和加速
    with torch.no_grad():
        equalized_output = model(test_features_tensor)

    # 将结果移回CPU并转换为numpy数组
    pam_out, pcm_out = equalized_output
    pam_out_np = pam_out.cpu().numpy()
    pcm_out_np = pcm_out.cpu().numpy().flatten()

    # PAM 输出展平为符号序列
    pam_out_seq = pam_out_np.reshape(-1)

    # ====================== Clamp PAM 输出 ======================
    # Head-A 已移除 Tanh，输出可能超出 [-1, 1]
    # MATLAB Rx_Part2 期望接收 [-1, 1] 范围的信号进行门限判决
    pam_out_seq = np.clip(pam_out_seq, -1.0, 1.0)
    print(f"PAM 输出已 clamp 到 [-1, 1] 范围 (实际范围: [{pam_out_np.min():.4f}, {pam_out_np.max():.4f}])")

    # --- 5. 保存结果到 .mat 文件 ---
    # MATLAB Rx_Part2_PostProcessing.m 期望一个名为 'received_eq' 的变量
    output_data = {
        'received_eq': pam_out_seq,
        'pcm_eq': pcm_out_np
    }

    spio.savemat(output_file, output_data)

    print(f"--- 均衡完成 ---")
    print(f"均衡后的信号 (变量 'received_eq') 结果已保存到: {output_file}")
    print(f"PCM 回归输出 (变量 'pcm_eq') 结果已保存到: {output_file}")
    print(f"原始测试信号长度: {len(distorted_signal)} 符号")
    print(f"均衡后 PAM 符号长度: {len(pam_out_seq)} (块级输出)")
    print(f"均衡后 PCM 采样长度: {len(pcm_out_np)}")


if __name__ == '__main__':
    # 使用与 'drof_dnn_train' 相同的配置，因为它包含所有必需的路径
    config = cfg.Configs['drof_dnn_test']  #
    run_equalization(config)