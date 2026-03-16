from torch import nn


class SH_DNN(nn.Module):
    """
    Single-Head block DNN baseline:
    - 输入: 滑动窗口 [B, taps]
    - 输出: 块级 PAM 估计 [B, block_size]
    不包含 PCM 分支与先验回注。
    """
    def __init__(self, config):
        super(SH_DNN, self).__init__()
        input_dim = 2 * config.seq_len + 1
        self.block_size = config.quant // 2

        def fullnet(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU()
            )

        self.ls = nn.Sequential(
            fullnet(input_dim, 256),
            fullnet(256, 128),
            fullnet(128, 64),
        )
        self.pam_head = nn.Sequential(
            nn.Linear(64, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Linear(96, self.block_size),
        )
        pam_act_name = str(getattr(config, 'pam_output_activation', 'tanh')).lower()
        self.pam_act = nn.Tanh() if pam_act_name == 'tanh' else nn.Identity()

    def forward(self, x):
        x = self.ls(x)
        pam_out = self.pam_head(x)
        pam_out = self.pam_act(pam_out)
        return pam_out

