# -*- coding: utf-8 -*-
import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SOS_token = 0
    EOS_token = 1
    n_layers = 3
    n_iters = 100
    hidden_size = 256
    dropout_p = 0.1
    attn_model = 'concat'


opt = Config()
