# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from config import opt


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        attn_energies = Variable(torch.zeros(seq_len)).to(opt.device)

        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        return F.softmax(attn_energies, dim=0).unsqueeze(0)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p,
                          batch_first=True)
        self.out = nn.Linear(hidden_size * 2, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        word_embedded = self.embedding(word_input).view(1, 1, -1)

        # use (last target word or last predicated word) and last context as current input
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        attn_weights = self.attn(rnn_output.view(-1), encoder_outputs.squeeze(0))

        context = torch.mm(attn_weights, encoder_outputs.squeeze(0))

        rnn_output = rnn_output.squeeze(0)

        # 1 x hidden_size -> hidden_size
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), dim=1)

        return output, context, hidden, attn_weights
