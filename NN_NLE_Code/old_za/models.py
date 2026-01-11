
from torch import nn



class DNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()
        self.device = config.device
        def fullnet(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            )

        self.ls = nn.Sequential(
            fullnet(in_dim=config.d_model * (config.seq_len*2+1),out_dim = 512),
            fullnet(512,256),
            fullnet(256, 128),
            fullnet(128, 64),
            fullnet(64, 32)
        )
        self.ln = nn.Linear(32,20)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.ls(x)
        x = self.ln(x)
        x = self.tanh(x)
        return x