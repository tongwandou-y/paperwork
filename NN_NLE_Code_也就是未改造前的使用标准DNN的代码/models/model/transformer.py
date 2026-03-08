"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
import numpy as np
from models.model.decoder import Decoder
from models.model.encoder import Encoder
import configs as cfg

class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(d_model=config.d_model,
                               n_head=config.transformer.n_heads,
                               seq_len=config.seq_len * 2 +1,
                               ffn_hidden=config.transformer.ffn_hidden,
                               drop_prob=config.transformer.drop_prob,
                               n_layers=config.transformer.encoder_num_layers,
                               device=config.device)

        self.decoder = Decoder(d_model=config.d_model,
                               n_head=config.transformer.n_heads,
                               seq_len=1,
                               ffn_hidden=config.transformer.ffn_hidden,
                               drop_prob=config.transformer.drop_prob,
                               n_layers=config.transformer.decoder_num_layers,
                               device=config.device)
        self.device = config.device
        self.tanh = nn.Tanh()

    def forward(self, src, trg):
        trg_mask = self.get_attn_subsequence_mask(trg)
        enc_src = self.encoder(src)
        output = self.decoder(trg, enc_src, trg_mask)
        output = self.tanh(output)

        return output

    def get_attn_subsequence_mask(self,seq):

        'seq:[batch_size, tat_len, d_model]'
        # batch_size,
        if seq is not None:
            attn_shape = [seq.size(0), 1, seq.size(1), seq.size(1)]
            subsequence_mask = np.triu(np.ones(attn_shape), k=1)
            subsequence_mask = torch.from_numpy(subsequence_mask).byte().to(self.device)
        else:
            subsequence_mask = None
        # print(subsequence_mask.size())
        return subsequence_mask  # [batch_size, 1, tgt_len, tgt_len]




