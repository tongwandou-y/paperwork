# DNNV3-0/models/model/DNN.py

from torch import nn
import torch

class DNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()
        self.device = config.device

        # 完整的滑动窗口长度
        input_dim = 2 * config.seq_len + 1
        # 物理块大小 G = quant / Mm (PAM4 => Mm=2)
        self.block_size = config.quant // 2

        # 定义子网络块：线性层 -> 批归一化 -> 激活函数
        def fullnet(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU()
            )

        # ====================== 骨干网络 ======================
        # 增大容量：17 → 256 → 128 → 64
        # 为双头4+1输出提供更丰富的共享特征表示
        self.ls = nn.Sequential(
            fullnet(in_dim=input_dim, out_dim=256),
            fullnet(256, 128),
            fullnet(128, 64)
        )

        # ====================== 双头输出层 ======================
        # Head-A: 输出 G 个 PAM 符号电平 (无 Tanh，线性输出)
        #   - 关键修改：移除 Tanh 激活！
        #   - 原因：PAM4 标签 {-1, +1} 恰好在 Tanh 的饱和区，
        #     导致约50%的样本梯度接近0，严重阻碍学习。
        #   - 线性输出 + MSE Loss 自然把输出推向正确电平。
        self.ln_pam = nn.Linear(64, self.block_size)

        # Head-B: 输出 1 个 PCM 电压 (保留 Tanh，因为 PCM 标签在 [-1,1] 内部)
        self.ln_pcm = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.ls(x)
        # Head-A: 线性输出，不经过 Tanh
        pam_out = self.ln_pam(x)
        # Head-B: Tanh 限制在 [-1, 1]
        pcm_out = self.tanh(self.ln_pcm(x))
        return pam_out, pcm_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)