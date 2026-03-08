# DNNV3-0/train.py

from torch.utils.data import DataLoader
from util.load_data_mat import MatDataset  # 导入我们新添加的MatlabDataset
from util.utils import *
from models.model.DNN import DNN
from models.model.CNN import CNN
from torch import nn
import configs as cfg
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os  # 确保导入 os


def train(configs):
    device = configs.device

    # --- 1. 使用新的MatlabDataset加载数据 ---
    print("使用MatlabDataset加载训练数据...")

    dataset = MatDataset(mat_file_path=configs.train_data_file,
                         seq_len=configs.seq_len,
                         is_train=True)  # is_train=True 表示加载训练部分

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=configs.batch_size,
        shuffle=True,  # 在训练时打乱数据
        pin_memory=True,
        num_workers=0,  # 在Windows上建议设为0以避免多进程问题
        drop_last=True
    )

    # --- 2. 模型初始化 (根据配置选择)、损失函数、优化器 ---
    print(f"正在初始化模型: {configs.model_type} ...")

    if configs.model_type == 'DNN':
        model = DNN(configs).to(device)
    elif configs.model_type == 'CNN':
        model = CNN(configs).to(device)
    else:
        raise ValueError(f"未知的模型类型: {configs.model_type}")

    init_weights(model, init_type=configs.init_type)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learn_rate)

    # ！！！！！！定义调度器 ！！！！！！
    # 当 avg_loss 停止下降时，自动降低学习率
    # 'min': 监控的指标越小越好
    # patience=10: 10轮没下降，就触发
    # factor=0.5: 触发时，学习率 x 0.5
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
    # ！！！！！！！！！！！！！！！！！！ 修改点 结束 ！！！！！！！！！！！！！！！！！！！！

    # --- 3. 加载检查点 (如果存在) ---
    ckpt_dir = os.path.join('output', configs.experiment_name, 'checkpoint')
    mkdir(ckpt_dir)

    # [新增] 获取日志文件路径 (在 configs.py 中定义)
    log_file_path = configs.loss_log_file
    # [新增] 初始化最佳 Loss
    best_loss = float('inf')

    try:
        ckpt, ckpt_path = load_checkpoint(ckpt_dir)
        start_ep = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

        # 如果ckpt里有调度器状态，就恢复它；没有就不恢复（兼容旧模型）
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])

        print(f"从 {ckpt_path} 恢复训练...")

        # [新增] 如果是从 0 开始 (或者文件不存在)，写入表头；否则追加
        if start_ep == 0 or not os.path.exists(log_file_path):
            # 确保存放日志的目录存在
            log_dir = os.path.dirname(log_file_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(log_file_path, "w") as f:
                f.write("Epoch,Loss,LearningRate\n")

    except FileNotFoundError:
        print('[*] 没有找到检查点，从头开始训练。')
        start_ep = 0

        # [新增] 从头开始，初始化日志文件
        # 确保存放日志的目录存在
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(log_file_path, "w") as f:
            f.write("Epoch,Loss,LearningRate\n")

    # --- 4. 训练循环 ---
    print(f"\n开始训练，共 {configs.epoch} 轮...")
    model.train()  # 将模型设置为训练模式

    for ep in range(start_ep, configs.epoch):
        epoch_loss_sum = 0.0
        for i, (features, _, labels) in enumerate(train_loader):  # 我们只需要 features 和 labels
            # 将数据移动到GPU或CPU
            features = features.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(features)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重

            epoch_loss_sum += loss.item()

        # 打印每轮的平均损失
        avg_loss = epoch_loss_sum / len(train_loader)
        print(f'轮次 [{ep + 1}/{configs.epoch}], 平均损失: {avg_loss:.8f}')

        # ！！！！！！执行调度器 ！！！！！！
        scheduler.step(avg_loss)

        # 获取当前的学习率
        current_lr = optimizer.param_groups[0]['lr']

        # [新增] 判断是否是最佳模型
        is_best = False
        if avg_loss < best_loss:
            best_loss = avg_loss
            is_best = True
            #print(f"   --> 发现新低 Loss! (Epoch {ep + 1})")

        # [新增] 将 Loss 和 LR 写入日志文件 (追加模式)
        with open(log_file_path, "a") as f:
            f.write(f"{ep + 1},{avg_loss:.8f},{current_lr}\n")

        # 保存模型检查点
        save_checkpoint({'epoch': ep + 1,
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'loss': avg_loss,  # 当前轮次的 Loss
                         'learning_rate': current_lr,  # 当前的学习率
                         'config': configs,  # 实验配置快照 (新增)
                         },
                        os.path.join(ckpt_dir, f'Epoch_({ep + 1}).ckpt'),
                        is_best=is_best,  # [新增] 传入 is_best 标记，utils会处理保存 best_model.ckpt
                        max_keep=None)

    print("模型训练完成。")
    print(f"模型权重已保存在目录: {ckpt_dir}")


if __name__ == '__main__':
    # 从configs.py加载配置
    config = cfg.Configs['drof_dnn_train']
    train(config)
