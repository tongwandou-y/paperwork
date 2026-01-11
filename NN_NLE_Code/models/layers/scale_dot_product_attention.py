"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        d_tensor = q.size(-1) # [batch_size, head, length, d_tensor]

        # 1. dot product Query with Key^T to compute similarity

        k_ = k.transpose(-2,-1)  #(batch_size, head, d_tensor, length)
        score = torch.matmul(q,k_) / math.sqrt(d_tensor) # scaled dot product

        # 2. apply masking (opt)

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        p_attn = self.softmax(score)

        # 4. multiply with Value
        v_ = torch.matmul(p_attn, v)

        return v_, p_attn
