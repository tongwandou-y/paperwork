"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, seq_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        seq_len=seq_len,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, d_model*4)
        self.linear1 = nn.Linear(d_model*4, d_model)

    def forward(self, trg, enc_src, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src,  src_mask)

        # pass to LM head
        output = self.linear(trg)
        output = self.linear1(output)
        return output
