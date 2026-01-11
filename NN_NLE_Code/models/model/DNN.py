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

        # ！！！！！！！！！！！！！！！！！！ 修改点 开始 ！！！！！！！！！！！！！！！！！！！！
        # 原来的模型非常庞大 (19 -> 512 -> 256 -> 128 -> 64 -> 32 -> 1)
        # 拥有约 185,000 个参数，对于 36,000 个训练样本来说太大了，容易过拟合。

        # 我们上次使用的 19 -> 64 -> 32 -> 1 模型欠拟合了。
        # 现在我们尝试一个中等大小的模型：
        # (19 -> 128 -> 64 -> 32 -> 1)
        # 这个模型大约有 22,000 个参数。

        self.ls = nn.Sequential(
            fullnet(in_dim=input_dim, out_dim=128),
            fullnet(128, 64),
            fullnet(64, 32)
        )

        # 输出层：线性映射到 1 维
        self.ln = nn.Linear(32, 1)

        # Tanh 激活函数：确保输出严格限制在 [-1, 1] 之间
        # 适配归一化后的标签范围
        self.tanh = nn.Tanh()

        # ！！！！！！！！！！！！！！！！！！ 修改点 结束 ！！！！！！！！！！！！！！！！！！！！

        # self.ls = nn.Sequential(
        #     # 输入层: 输入维度为滑动窗口的总长度
        #     fullnet(in_dim=input_dim, out_dim=512),
        #     fullnet(512, 256),
        #     fullnet(256, 128),
        #     fullnet(128, 64),
        #     fullnet(64, 32)
        # )
        # 输出层: 输出一个均衡后的值

    def forward(self, x):
        # x的输入形状是 [batch_size, taps]，需要先展平
        # 注意：在 train.py 中，我们会确保输入是正确的二维形状
        x = self.ls(x)
        x = self.ln(x)
        x = self.tanh(x)  # Tanh激活函数确保输出值在-1到1之间，与我们归一化的标签匹配
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



from torch import nn

class DNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()
        self.device = config.device

        # 完整的滑动窗口长度
        input_dim = 2 * config.seq_len + 1

        def fullnet(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # BatchNorm1d 适用于批处理数据, 每个通道独立归一化
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            )

        # ！！！！！！！！！！！！！！！！！！ 修改点 开始 ！！！！！！！！！！！！！！！！！！！！
        # 原来的模型非常庞大 (19 -> 512 -> 256 -> 128 -> 64 -> 32 -> 1)
        # 拥有约 185,000 个参数，对于 36,000 个训练样本来说太大了，容易过拟合。

        # 我们上次使用的 19 -> 64 -> 32 -> 1 模型欠拟合了。
        # 现在我们尝试一个中等大小的模型：
        # (19 -> 128 -> 64 -> 32 -> 1)
        # 这个模型大约有 22,000 个参数。

        self.ls = nn.Sequential(
            fullnet(in_dim=input_dim, out_dim=128),
            fullnet(128, 64),
            fullnet(64, 32)
        )
        self.ln = nn.Linear(32, 1)
        self.tanh = nn.Tanh()

        # ！！！！！！！！！！！！！！！！！！ 修改点 结束 ！！！！！！！！！！！！！！！！！！！！

        # self.ls = nn.Sequential(
        #     # 输入层: 输入维度为滑动窗口的总长度
        #     fullnet(in_dim=input_dim, out_dim=512),
        #     fullnet(512, 256),
        #     fullnet(256, 128),
        #     fullnet(128, 64),
        #     fullnet(64, 32)
        # )
        # 输出层: 输出一个均衡后的值


    def forward(self, x):
        # x的输入形状是 [batch_size, taps]，需要先展平
        # 注意：在 train.py 中，我们会确保输入是正确的二维形状
        x = self.ls(x)
        x = self.ln(x)
        x = self.tanh(x) # Tanh激活函数确保输出值在-1到1之间，与我们归一化的标签匹配
        return x