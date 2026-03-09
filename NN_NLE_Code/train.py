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
    use_loss_ema_normalization = bool(getattr(configs, 'use_loss_ema_normalization', True))
    loss_ema_momentum = float(getattr(configs, 'loss_ema_momentum', 0.98))
    loss_norm_eps = float(getattr(configs, 'loss_norm_eps', 1e-8))
    pam_priority_factor = float(getattr(configs, 'pam_priority_factor', 1.20))
    aux_to_pam_max_ratio = float(getattr(configs, 'aux_to_pam_max_ratio', 0.80))

    # 可学习的log-variance参数（对应 L_pam / L_pcm）
    # 初始化为0 -> 初始权重约为1
    loss_log_vars = nn.Parameter(torch.zeros(2, device=device))
    # 用于loss尺度归一化的EMA状态，避免不同任务量纲差异导致权重抖动
    loss_ema_state = torch.ones(2, device=device)
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
                ckpt_log_vars = ckpt['loss_log_vars'].to(device).flatten()
                n_copy = min(loss_log_vars.numel(), ckpt_log_vars.numel())
                loss_log_vars[:n_copy].copy_(ckpt_log_vars[:n_copy])
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

    for ep in range(start_ep, configs.epoch):
        epoch_loss_sum = 0.0
        epoch_loss_pam = 0.0
        epoch_loss_pcm = 0.0
        avg_loss_cons = 0.0

        # 损失项开关
        use_pam = configs.use_loss_pam
        use_pcm = configs.use_loss_pcm

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
            if use_pam:
                loss_pam = criterion(pam_out, pam_labels)
                epoch_loss_pam += loss_pam.item()
            if use_pcm:
                loss_pcm = criterion(pcm_out, pcm_labels)
                epoch_loss_pcm += loss_pcm.item()

            # 组合总损失（自适应多任务 + 主任务保护）
            # 1) 可选EMA归一化：消除不同loss量纲差异
            # 2) 不确定性加权：自动学习各任务相对权重
            # 3) PAM优先与辅助限幅：保证SER/BER目标不被辅助任务拖累
            def normalize_loss_with_ema(raw_loss, idx):
                if (raw_loss is None) or (not use_loss_ema_normalization):
                    return raw_loss
                with torch.no_grad():
                    loss_ema_state[idx] = (
                        loss_ema_momentum * loss_ema_state[idx]
                        + (1.0 - loss_ema_momentum) * raw_loss.detach()
                    )
                denom = torch.clamp(loss_ema_state[idx].detach(), min=loss_norm_eps)
                return raw_loss / denom

            norm_loss_pam = normalize_loss_with_ema(loss_pam, 0)
            norm_loss_pcm = normalize_loss_with_ema(loss_pcm, 1)

            pam_term = None
            pcm_term = None

            if use_pam and (norm_loss_pam is not None):
                pam_term = torch.exp(-loss_log_vars[0]) * norm_loss_pam + uncertainty_reg * loss_log_vars[0]
            if use_pcm and (norm_loss_pcm is not None):
                pcm_term = configs.loss_alpha * (
                    torch.exp(-loss_log_vars[1]) * norm_loss_pcm + uncertainty_reg * loss_log_vars[1]
                )

            aux_terms = []
            if pcm_term is not None:
                aux_terms.append(pcm_term)

            if pam_term is not None:
                loss = pam_priority_factor * pam_term
                if len(aux_terms) > 0:
                    aux_total = aux_terms[0]
                    for aux_t in aux_terms[1:]:
                        aux_total = aux_total + aux_t

                    aux_limit = aux_to_pam_max_ratio * torch.clamp(torch.abs(pam_term.detach()), min=1e-8)
                    aux_total_detached = torch.abs(aux_total.detach())
                    if aux_total_detached.item() > aux_limit.item():
                        scale = aux_limit / torch.clamp(aux_total_detached, min=1e-8)
                        aux_total = aux_total * scale
                    loss = loss + aux_total
            else:
                if len(aux_terms) > 0:
                    loss = aux_terms[0]
                    for aux_t in aux_terms[1:]:
                        loss = loss + aux_t

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
        current_lr = optimizer.param_groups[0]['lr']
        with torch.no_grad():
            w_pam = float(torch.exp(-loss_log_vars[0]).item())
            w_pcm = float(torch.exp(-loss_log_vars[1]).item())
        print(
            f'轮次 [{ep + 1}/{configs.epoch}], 平均损失: {avg_loss:.8f} | '
            f'L_pam: {avg_loss_pam:.6f} | L_pcm: {avg_loss_pcm:.6f} | '
            f'L_cons: {avg_loss_cons:.6f} | w_pam: {w_pam:.3f} | w_pcm: {w_pcm:.3f} | '
            f'LR: {current_lr:.2e}'
        )

        # CosineAnnealing：每个epoch步进一次（不需要传参数）
        scheduler.step()

        # 判断是否是最佳模型
        if best_model_metric == 'pam_loss':
            metric_value = avg_loss_pam
        elif best_model_metric == 'total_loss':
            metric_value = avg_loss
        elif best_model_metric == 'hybrid':
            metric_value = avg_loss_pam + 0.2 * avg_loss_pcm
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
