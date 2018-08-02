# -*- coding: utf-8 -*-
import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_percent = 0.1
    batch_size = 100
    max_length= 10
    SOS_token = 0
    EOS_token = 1
    PAD_token = 2
    n_layers = 3
    n_iters = 1
    hidden_size = 256
    dropout_p = 0.1
    attn_model = 'general'


opt = Config()
