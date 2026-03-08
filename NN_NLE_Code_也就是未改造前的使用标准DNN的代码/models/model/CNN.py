# # DNNV3-0/models/model/CNN.py
#
# from torch import nn
# import torch
# import math
#
#
# class SEBlock(nn.Module):
#     """注意力机制：在低算力下强化关键特征"""
#
#     def __init__(self, channel, reduction=8):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1)
#         return x * y.expand_as(x)
#
#
# class GhostModule(nn.Module):
#     """
#     Ghost Module: 用极低的代价实现高维特征
#     替代原本昂贵的 Pointwise 卷积
#     """
#
#     def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
#         super(GhostModule, self).__init__()
#         self.oup = oup
#         init_channels = math.ceil(oup / ratio)
#         new_channels = init_channels * (ratio - 1)
#
#         # 1. Primary Conv: 生成一部分“本征”特征 (计算量小)
#         self.primary_conv = nn.Sequential(
#             nn.Conv1d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
#             nn.BatchNorm1d(init_channels),
#             nn.GELU() if relu else nn.Sequential(),
#         )
#
#         # 2. Cheap Operation: 通过简单的线性变换生成“幻影”特征
#         self.cheap_operation = nn.Sequential(
#             nn.Conv1d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
#             nn.BatchNorm1d(new_channels),
#             nn.GELU() if relu else nn.Sequential(),
#         )
#
#     def forward(self, x):
#         x1 = self.primary_conv(x)
#         x2 = self.cheap_operation(x1)
#         # 将本征特征和幻影特征拼接，还原完整维度
#         out = torch.cat([x1, x2], dim=1)
#         return out[:, :self.oup, :]
#
#
# class CNN(nn.Module):
#     def __init__(self, config):
#         super(CNN, self).__init__()
#         self.device = config.device
#
#         # 输入序列长度 (例如: 8*2 + 1 = 17)
#         self.input_len = 2 * config.seq_len + 1
#
#         # --- Ghost-Wider-CNN ---
#         # 策略:
#         # 1. 继承 Sep-Wider 的成功架构 (3层, L=5, C=64)。
#         # 2. 用 Ghost Module 替换 Pointwise 卷积，大幅降低算力。
#         # 3. 重新引入 SE Block，在不牺牲 Regressor 宽度的前提下提升精度。
#
#         def ghost_bottleneck(in_c, out_c, stride):
#             return nn.Sequential(
#                 # 1. Depthwise Conv (负责降采样和空间特征)
#                 nn.Conv1d(in_c, in_c, kernel_size=3, stride=stride,
#                           padding=1, groups=in_c, bias=False),
#                 nn.BatchNorm1d(in_c),
#                 nn.GELU(),
#
#                 # 2. Ghost Module (负责通道扩张，替代昂贵的 1x1 Conv)
#                 # ratio=2 意味着一半的特征是“幻影”，节省一半算力
#                 GhostModule(in_c, out_c, relu=True),
#
#                 # 3. SE Attention (画龙点睛)
#                 SEBlock(out_c)
#             )
#
#         self.feature_extractor = nn.Sequential(
#             # Layer 1: 1 -> 16 (标准卷积)
#             nn.Conv1d(1, 16, kernel_size=5, padding=2),
#             nn.BatchNorm1d(16),
#             nn.GELU(),
#
#             # Layer 2: 16 -> 32 (Ghost Bottleneck)
#             # 算力比 SepConv 还要低
#             ghost_bottleneck(16, 32, stride=2),
#
#             # Layer 3: 32 -> 64 (Ghost Bottleneck)
#             # 这里的 64 通道是高质量的，且由 SE 加持
#             ghost_bottleneck(32, 64, stride=2)
#         )
#
#         # --- 自动计算展平后的维度 ---
#         # 17 -> 9 -> 5
#         # Flatten维度: 64通道 * 5长度 = 320 (保持这个维度是高精度的关键！)
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, 1, self.input_len)
#             dummy_output = self.feature_extractor(dummy_input)
#             flatten_dim = dummy_output.view(1, -1).size(1)
#
#         self.regressor = nn.Sequential(
#             # 坚决保持 64 个神经元，吸取 SE-Sep 失败的教训
#             nn.Linear(flatten_dim, 64),
#             nn.GELU(),
#
#             # 64 -> 1
#             nn.Linear(64, 1),
#             nn.Tanh()
#         )
#
#         self._initialize_weights()
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.feature_extractor(x)
#         x = x.view(x.size(0), -1)
#         x = self.regressor(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv1d, nn.Linear)):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

##############################################

# DNNV3-0/models/model/CNN.py
# 已经很牛的一个版本了  计算三倍  但是比DNN能提高快2dB

from torch import nn
import torch


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.device = config.device

        # 输入序列长度 (例如: 8*2 + 1 = 17)
        self.input_len = 2 * config.seq_len + 1

        # --- Sep-Wider-CNN (深度可分离宽体网络) ---
        # 策略: 利用 MobileNet 的核心技术(分离卷积)大幅降低计算量，
        #       从而在有限算力下实现 64 通道的宽体特征提取。

        def sep_conv_block(in_c, out_c, kernel_size, stride, padding):
            """构建深度可分离卷积块"""
            return nn.Sequential(
                # 1. Depthwise Conv: 只负责提取空间特征 (计算量极低)
                # groups=in_c 是实现深度卷积的关键
                nn.Conv1d(in_c, in_c, kernel_size=kernel_size, stride=stride,
                          padding=padding, groups=in_c),

                # 2. Pointwise Conv: 只负责组合通道特征 (1x1 卷积)
                nn.Conv1d(in_c, out_c, kernel_size=1),

                # 3. 归一化和激活
                nn.BatchNorm1d(out_c),
                nn.GELU()
            )

        self.feature_extractor = nn.Sequential(
            # Layer 1: 1 -> 16 (标准卷积)
            # 第一层输入通道少，直接用普通卷积提取基础特征最稳
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),

            # Layer 2: 16 -> 32 (分离卷积 & 降采样)
            # 长度 17 -> 9
            sep_conv_block(16, 32, kernel_size=3, stride=2, padding=1),

            # Layer 3: 32 -> 64 (分离卷积 & 降采样)
            # 长度 9 -> 5
            # 在这里，我们做到了 64 通道，但计算量非常低
            sep_conv_block(32, 64, kernel_size=3, stride=2, padding=1)
        )

        # --- 自动计算展平后的维度 ---
        # 17 -> 9 -> 5
        # Flatten维度: 64通道 * 5长度 = 320
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.input_len)
            dummy_output = self.feature_extractor(dummy_input)
            flatten_dim = dummy_output.view(1, -1).size(1)

        self.regressor = nn.Sequential(
            # 320 -> 64: 全连接层占据了模型一半的算力，用于非线性拟合
            nn.Linear(flatten_dim, 64),
            nn.GELU(),

            # 64 -> 1: 输出
            nn.Linear(64, 1),
            nn.Tanh()
        )

        self._initialize_weights()

    def forward(self, x):
        # [Batch, Taps] -> [Batch, 1, Taps]
        x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



########################################################



# # DNNV3-0/models/model/CNN.py
#
# from torch import nn
# import torch
#
#
# class CNN(nn.Module):
#     def __init__(self, config):
#         super(CNN, self).__init__()
#         self.device = config.device
#
#         # 输入序列长度 (例如: 8*2 + 1 = 17)
#         self.input_len = 2 * config.seq_len + 1
#
#         # --- Pyramid-CNN (金字塔卷积网络) ---
#         # 策略: 3层深度 + 逐层通道加倍 + 逐层时间减半
#         # 目标: 显著提升非线性拟合能力，同时控制计算量在合理范围 (DNN的3倍左右)
#
#         self.feature_extractor = nn.Sequential(
#             # Layer 1: 1 -> 12 (保持长度 17)
#             # 提取基础时域特征
#             nn.Conv1d(1, 12, kernel_size=5, padding=2),
#             nn.BatchNorm1d(12),
#             nn.GELU(),
#
#             # Layer 2: 12 -> 24 (长度 17 -> 9)
#             # 第一次降采样，特征维度加倍
#             nn.Conv1d(12, 24, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(24),
#             nn.GELU(),
#
#             # Layer 3: 24 -> 48 (长度 9 -> 5)
#             # 第二次降采样，特征维度再加倍，提取深层抽象特征
#             nn.Conv1d(24, 48, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(48),
#             nn.GELU()
#         )
#
#         # --- 自动计算展平后的维度 ---
#         # 17 ->(stride2)-> 9 ->(stride2)-> 5
#         # 最终特征图大小: [Batch, 48, 5]
#         # Flatten维度: 48 * 5 = 240
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, 1, self.input_len)
#             dummy_output = self.feature_extractor(dummy_input)
#             flatten_dim = dummy_output.view(1, -1).size(1)
#
#         self.regressor = nn.Sequential(
#             # 240 -> 64: 融合多尺度特征
#             # 参数量: 240*64 ≈ 1.5w (可接受)
#             nn.Linear(flatten_dim, 64),
#             nn.GELU(),
#
#             # 64 -> 1: 输出
#             nn.Linear(64, 1),
#             nn.Tanh()
#         )
#
#         self._initialize_weights()
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.feature_extractor(x)
#         x = x.view(x.size(0), -1)
#         x = self.regressor(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv1d, nn.Linear)):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


#######################################################################

#
# # DNNV3-0/models/model/CNN.py
#
# from torch import nn
# import torch
#
#
# class CNN(nn.Module):
#     def __init__(self, config):
#         super(CNN, self).__init__()
#         self.device = config.device
#
#         # 输入序列长度 (例如: 12*2 + 1 = 25)
#         self.input_len = 2 * config.seq_len + 1
#
#         # --- 卷积特征提取层 ---
#         # Conv1d 输入要求: [batch, channels, length]
#         # 我们的输入是单通道 (channel=1)
#
#         self.feature_extractor = nn.Sequential(
#             # 第一层卷积: 提取基础特征
#             # 输入: [batch, 1, 25] -> 输出: [batch, 32, 25] (padding=1 保持长度)
#             nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
#             nn.BatchNorm1d(32),
#             nn.GELU(),
#
#             # 第二层卷积: 加深特征
#             # 输入: [batch, 32, 25] -> 输出: [batch, 64, 25]
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3),
#             nn.BatchNorm1d(64),
#             nn.GELU(),
#
#             # 第三层卷积: 进一步提取
#             # 输入: [batch, 64, 25] -> 输出: [batch, 64, 25]
#             nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3),
#             nn.BatchNorm1d(64),
#             nn.GELU()
#         )
#
#         # --- 全连接回归层 ---
#         # 卷积后的数据需要展平 (Flatten) 才能进入全连接层
#         # 展平后的维度 = 通道数(64) * 序列长度(input_len)
#         flatten_dim = 64 * self.input_len
#
#         self.regressor = nn.Sequential(
#             nn.Linear(flatten_dim, 128),
#             nn.GELU(),
#             # nn.Dropout(0.2),  # 移除 Dropout，防止欠拟合
#
#             nn.Linear(128, 1),  # 输出层: 1个值
#             nn.Tanh()  # 激活: 映射到 [-1, 1] 以匹配标签
#         )
#
#         self._initialize_weights()
#
#     def forward(self, x):
#         # --- 关键适配点 ---
#         # 原始输入 x 的形状是 [batch_size, taps] (例如 [64, 25])
#         # Conv1d 需要输入形状为 [batch_size, channels, length]
#         # 所以我们需要在这里增加一个维度
#         x = x.unsqueeze(1)  # 变为 [batch_size, 1, taps]
#
#         # 特征提取
#         x = self.feature_extractor(x)
#
#         # 展平: [batch, 64, 25] -> [batch, 64*25]
#         x = x.view(x.size(0), -1)
#
#         # 回归预测
#         x = self.regressor(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 # 卷积层初始化 (Kaiming)
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 # [核心修改]: 全连接层也改用 Kaiming 初始化，与 DNN 保持一致！
#                 # 原来的 nn.init.normal_(m.weight, 0, 0.01) 方差太小，导致收敛慢
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

