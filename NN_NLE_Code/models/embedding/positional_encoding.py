"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class PostionalEncoding(nn.Module):
    """
    compute sinusoid encoding. 计算正弦编码
    """

    def __init__(self, d_model, seq_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param seq_len: max sequence length
        :param device: hardware Devices setting
        """
        super(PostionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(seq_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, seq_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len_,_ = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len_, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ps_en = PostionalEncoding(10,5,device)
    a_shape = [8,5]
    aka = torch.randn(a_shape)
    aka_pe = ps_en(aka)
    b_shape = [8, 5, 10]
    akb = torch.randn(b_shape)
    emb_ak = aka_pe + akb
    print(emb_ak)


if __name__ == '__main__':
    main()