# DNNV3-0/train.py

from torch.utils.data import DataLoader
from util.load_data_mat import MatDatasetBlock
from util.utils import *
from models.model.DNN import DNN
from models.model.CNN import CNN
from torch import nn
import configs as cfg
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import numpy as np


def train(configs):
    device = configs.device

    # --- 1. 使用新的MatlabDataset加载数据 ---
    print("使用MatDatasetBlock加载训练数据(块级对齐)...")

    dataset = MatDatasetBlock(mat_file_path=configs.train_data_file,
                              seq_len=configs.seq_len,
                              is_train=True)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True
    )

    # ====================== 数据质量诊断 ======================
    # 检查输入特征与标签之间的相关性
    # 如果相关系数很低 (<0.1)，说明帧同步可能失败或数据对齐有误
    print("\n--- 数据质量诊断 ---")
    all_features = dataset.features.numpy()
    all_labels_pam = dataset.labels_pam.numpy()
    seq_len = configs.seq_len
    center_feat = all_features[:, seq_len]  # 窗口中心值 = 当前符号
    for j in range(all_labels_pam.shape[1]):
        corr = np.corrcoef(center_feat, all_labels_pam[:, j])[0, 1]
        print(f"  输入中心 vs PAM标签[{j}] 相关系数: {corr:.4f}")
    mean_corr = np.mean([np.corrcoef(center_feat, all_labels_pam[:, j])[0, 1]
                         for j in range(all_labels_pam.shape[1])])
    if abs(mean_corr) < 0.1:
        print("  ⚠️ 警告：输入与标签相关性极低！请检查帧同步和数据对齐是否正确。")
    else:
        print(f"  ✓ 平均相关系数: {mean_corr:.4f}，数据对齐看起来正常。")
    # 打印标签统计
    print(f"  PAM标签统计: mean={all_labels_pam.mean():.4f}, std={all_labels_pam.std():.4f}, "
          f"min={all_labels_pam.min():.4f}, max={all_labels_pam.max():.4f}")
    print("--- 诊断结束 ---\n")

    # --- 2. 模型初始化、损失函数、优化器 ---
    print(f"正在初始化模型: {configs.model_type} ...")

    if configs.model_type == 'DNN':
        model = DNN(configs).to(device)
    elif configs.model_type == 'CNN':
        model = CNN(configs).to(device)
    else:
        raise ValueError(f"未知的模型类型: {configs.model_type}")

    init_weights(model, init_type=configs.init_type)
    criterion = nn.MSELoss()

    # ====================== 自适应损失权重策略 ======================
    # 仅保留 auto_uncertainty 路径，移除旧的手动分段逻辑
    use_auto_uncertainty = (getattr(configs, 'loss_weight_strategy', 'auto_uncertainty') == 'auto_uncertainty')
    if not use_auto_uncertainty:
        raise ValueError("当前版本仅支持 loss_weight_strategy='auto_uncertainty'")
    uncertainty_reg = float(getattr(configs, 'auto_uncertainty_reg', 0.05))

    # 可学习的log-variance参数（对应 L_pam / L_pcm / L_cons）
    # 初始化为0 -> 初始权重约为1
    loss_log_vars = nn.Parameter(torch.zeros(3, device=device))
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [loss_log_vars],
        lr=configs.learn_rate
    )

    # ====================== 学习率调度器 ======================
    # 【关键修改】用 CosineAnnealingLR 替换 ReduceLROnPlateau
    # 原因：ReduceLROnPlateau + 分阶段损失 = 灾难
    #   - 每次引入新损失项时总loss会跳升，触发patience计数
    #   - patience=10 导致epoch 31就开始衰减LR
    #   - 到epoch 64时LR已经只有6.25e-5，模型根本学不动
    #
    # CosineAnnealing 平滑衰减，保证整个训练过程有足够大的学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=configs.epoch, eta_min=1e-6)

    # --- 3. 加载检查点 ---
    ckpt_dir = os.path.join('output', configs.experiment_name, 'checkpoint')
    mkdir(ckpt_dir)
    log_file_path = configs.loss_log_file
    best_loss = float('inf')
    best_model_metric = getattr(configs, 'best_model_metric', 'pam_loss')

    try:
        if configs.force_restart:
            raise FileNotFoundError
        ckpt, ckpt_path = load_checkpoint(ckpt_dir)
        start_ep = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'loss_log_vars' in ckpt:
            with torch.no_grad():
                loss_log_vars.copy_(ckpt['loss_log_vars'].to(device))
        print(f"从 {ckpt_path} 恢复训练...")

        if start_ep == 0 or not os.path.exists(log_file_path):
            log_dir = os.path.dirname(log_file_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(log_file_path, "w") as f:
                f.write("Epoch,Loss,L_pam,L_pcm,L_cons,LearningRate\n")

    except FileNotFoundError:
        print('[*] 没有找到检查点，从头开始训练。')
        start_ep = 0
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(log_file_path, "w") as f:
            f.write("Epoch,Loss,L_pam,L_pcm,L_cons,LearningRate\n")

    # --- 4. 训练循环 ---
    print(f"\n开始训练，共 {configs.epoch} 轮...")
    model.train()

    # Soft-DAC 温度退火参数（可配置）
    alpha_start = float(getattr(configs, 'soft_dac_alpha_start', 3.0))
    alpha_end = float(getattr(configs, 'soft_dac_alpha_end', 12.0))
    alpha_step = (alpha_end - alpha_start) / max(configs.epoch - 1, 1)
    cons_target = getattr(configs, 'consistency_target', 'pcm_head')
    cons_ramp_ratio = float(getattr(configs, 'consistency_ramp_ratio', 0.35))
    cons_ramp_epochs = max(1, int(configs.epoch * cons_ramp_ratio))

    # 物理参数
    K = 2 ** configs.quant
    pam_levels = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], device=device).view(1, 1, 4)

    def soft_dac_reconstruct(pam_out, alpha):
        """
        pam_out: [B, G]  (Head-A的输出，已clamp到[-1.2, 1.2])
        返回: [B, 1] 的连续电压估计 (归一化到[-1, 1])
        """
        # 先 clamp 确保不会太远离 PAM 电平
        pam_clamped = pam_out.clamp(-1.2, 1.2)

        # [B, G, 1] 与 [1,1,4] 广播 -> [B, G, 4]
        dist2 = (pam_clamped.unsqueeze(-1) - pam_levels) ** 2
        probs = torch.softmax(-alpha * dist2, dim=-1)

        # Gray 码顺序: 00, 01, 11, 10 (对应 pam_levels 顺序)
        p00 = probs[:, :, 0]
        p01 = probs[:, :, 1]
        p11 = probs[:, :, 2]
        p10 = probs[:, :, 3]

        b1 = p11 + p10
        b0 = p01 + p11

        # [B, G*2] 按 MSB -> LSB 拼接
        bits = torch.stack([b1, b0], dim=-1).reshape(pam_out.size(0), -1)

        # 软码字期望
        weights = torch.arange(bits.size(1) - 1, -1, -1, device=device, dtype=bits.dtype)
        weights = (2 ** weights).view(1, -1)
        q_hat = torch.sum(bits * weights, dim=1, keepdim=True)

        # 线性映射到 [-1, 1]
        v_hat = 2.0 * (q_hat / (K - 1)) - 1.0
        return v_hat

    for ep in range(start_ep, configs.epoch):
        epoch_loss_sum = 0.0
        epoch_loss_pam = 0.0
        epoch_loss_pcm = 0.0
        epoch_loss_cons = 0.0
        alpha = alpha_start + alpha_step * ep

        # 损失项开关
        use_pam = configs.use_loss_pam
        use_pcm = configs.use_loss_pcm
        use_cons = configs.use_loss_cons

        for i, (features, pam_labels, pcm_labels) in enumerate(train_loader):
            features = features.to(device)
            pam_labels = pam_labels.to(device)
            pcm_labels = pcm_labels.to(device)

            # 前向传播
            pam_out, pcm_out = model(features)

            # 计算损失
            loss = torch.tensor(0.0, device=device)
            loss_pam = None
            loss_pcm = None
            loss_cons = None
            if use_pam:
                loss_pam = criterion(pam_out, pam_labels)
                epoch_loss_pam += loss_pam.item()
            if use_pcm:
                loss_pcm = criterion(pcm_out, pcm_labels)
                epoch_loss_pcm += loss_pcm.item()
            if use_cons:
                # 一致性损失：默认约束 Head-A 与 Head-B 的物理一致性，减少对PAM主任务的硬拉扯
                v_reconstruct = soft_dac_reconstruct(pam_out, alpha)
                if cons_target == 'pcm_label':
                    cons_ref = pcm_labels
                else:
                    cons_ref = pcm_out
                loss_cons = criterion(v_reconstruct, cons_ref)
                epoch_loss_cons += loss_cons.item()

            # 组合总损失（Kendall & Gal 风格多任务不确定性加权）
            #   L = sum(exp(-s_i) * L_i + lambda * s_i)
            # 其中 s_i 为可学习log-variance，lambda控制正则强度
            if use_pam and (loss_pam is not None):
                loss = loss + torch.exp(-loss_log_vars[0]) * loss_pam + uncertainty_reg * loss_log_vars[0]
            if use_pcm and (loss_pcm is not None):
                loss = loss + configs.loss_alpha * (
                    torch.exp(-loss_log_vars[1]) * loss_pcm + uncertainty_reg * loss_log_vars[1]
                )
            if use_cons and (loss_cons is not None):
                # 连续软启动：前若干epoch逐步引入一致性，不使用硬分段
                cons_scale = min(1.0, float(ep + 1) / float(cons_ramp_epochs))
                loss = loss + (configs.loss_beta * cons_scale) * (
                    torch.exp(-loss_log_vars[2]) * loss_cons + uncertainty_reg * loss_log_vars[2]
                )

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss_sum += loss.item()

        # 打印每轮的平均损失
        avg_loss = epoch_loss_sum / len(train_loader)
        avg_loss_pam = epoch_loss_pam / len(train_loader) if use_pam else 0.0
        avg_loss_pcm = epoch_loss_pcm / len(train_loader) if use_pcm else 0.0
        avg_loss_cons = epoch_loss_cons / len(train_loader) if use_cons else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        with torch.no_grad():
            w_pam = float(torch.exp(-loss_log_vars[0]).item())
            w_pcm = float(torch.exp(-loss_log_vars[1]).item())
            w_cons = float(torch.exp(-loss_log_vars[2]).item())
        print(
            f'轮次 [{ep + 1}/{configs.epoch}], 平均损失: {avg_loss:.8f} | '
            f'L_pam: {avg_loss_pam:.6f} | L_pcm: {avg_loss_pcm:.6f} | '
            f'L_cons: {avg_loss_cons:.6f} | w_pam: {w_pam:.3f} | w_pcm: {w_pcm:.3f} | '
            f'w_cons: {w_cons:.3f} | alpha: {alpha:.2f} | LR: {current_lr:.2e}'
        )

        # CosineAnnealing：每个epoch步进一次（不需要传参数）
        scheduler.step()

        # 判断是否是最佳模型
        if best_model_metric == 'pam_loss':
            metric_value = avg_loss_pam
        elif best_model_metric == 'total_loss':
            metric_value = avg_loss
        elif best_model_metric == 'hybrid':
            metric_value = avg_loss_pam + 0.2 * avg_loss_pcm + 0.1 * avg_loss_cons
        else:
            raise ValueError(f"未知 best_model_metric: {best_model_metric}")

        is_best = False
        if metric_value < best_loss:
            best_loss = metric_value
            is_best = True

        # 写入日志
        with open(log_file_path, "a") as f:
            f.write(f"{ep + 1},{avg_loss:.8f},{avg_loss_pam:.8f},{avg_loss_pcm:.8f},{avg_loss_cons:.8f},{current_lr}\n")

        # 保存检查点
        ckpt_payload = {'epoch': ep + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': avg_loss,
                        'model_select_metric': best_model_metric,
                        'model_select_value': metric_value,
                        'learning_rate': current_lr,
                        'config': configs,
                        }
        ckpt_payload['loss_log_vars'] = loss_log_vars.detach().cpu()

        save_checkpoint(ckpt_payload,
                        os.path.join(ckpt_dir, f'Epoch_({ep + 1}).ckpt'),
                        is_best=is_best,
                        max_keep=None)

    print("模型训练完成。")
    print(f"模型权重已保存在目录: {ckpt_dir}")


if __name__ == '__main__':
    config = cfg.Configs['drof_dnn_train']
    train(config)
