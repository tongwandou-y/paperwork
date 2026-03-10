# DNNV3-0/models/model/PP-CDNN.py

from torch import nn
import torch

class PP_CDNN(nn.Module):
    def __init__(self, config):
        super(PP_CDNN, self).__init__()
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

        # ====================== 级联双头输出层 ======================
        # Head-B (PCM) 先预测连续电压，再作为先验回注到 Head-A (PAM)。
        # 这样 PCM 任务不再只是旁路监督，而是直接参与 PAM 判决特征构建。
        self.ln_pcm = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

        # PAM 头输入维度 = 共享特征(64) + PCM回注特征(1) + 门控特征(64)
        # 门控特征采用逐通道乘法 x * pcm_out，增强“PCM先验 -> PAM判决”耦合强度。
        self.pam_fusion = nn.Sequential(
            nn.Linear(129, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Linear(96, self.block_size)
        )

    def forward(self, x):
        x = self.ls(x)
        # Head-B: 先预测PCM，Tanh限制在[-1, 1]
        pcm_out = self.tanh(self.ln_pcm(x))
        # Head-A: 级联 + 门控回注
        gated_feat = x * pcm_out
        pam_in = torch.cat([x, pcm_out, gated_feat], dim=1)
        pam_out = self.pam_fusion(pam_in)
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