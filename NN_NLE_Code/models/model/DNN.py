# DNNV3-0/models/model/DNN.py

from torch import nn
import torch

class DNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()
        self.device = config.device

        # 完整的滑动窗口长度
        input_dim = 2 * config.seq_len + 1

        # 定义子网络块：线性层 -> 批归一化 -> 激活函数
        def fullnet(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # BatchNorm1d 适用于批处理数据, 每个通道独立归一化
                nn.BatchNorm1d(out_dim),
                # 【修改点】使用 GELU 替代 ReLU
                # GELU (Gaussian Error Linear Unit) 提供平滑的非线性，更适合模拟信号回归
                nn.GELU()
            )

        self.ls = nn.Sequential(
            fullnet(in_dim=input_dim, out_dim=256),
            fullnet(256, 128),
            fullnet(128, 64)
        )

        # 输出层：线性映射到 1 维
        self.ln = nn.Linear(64, 1)
        pam_act_name = str(getattr(config, 'pam_output_activation', 'tanh')).lower()
        self.pam_act = nn.Tanh() if pam_act_name == 'tanh' else nn.Identity()

    def forward(self, x):
        # x的输入形状是 [batch_size, taps]，需要先展平
        # 注意：在 train.py 中，我们会确保输入是正确的二维形状
        x = self.ls(x)
        x = self.ln(x)
        x = self.pam_act(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用 Kaiming 初始化 (适配 GELU/ReLU)
                # mode='fan_out' 保持前向传播中方差的一致性
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)