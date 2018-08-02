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
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        attn_energies = torch.zeros(batch_size, seq_len).to(opt.device)

        for i in range(seq_len):
            attn_energies[:, i] = self.score(hidden[:, 0], encoder_outputs[:, i])

        return F.softmax(attn_energies, dim=1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.bmm(hidden.unsqueeze(1), energy.unsqueeze(2)).view(-1)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.mm(energy, self.other.unsqueeze(1))
            return energy.squeeze(1)


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
        word_embedded = self.embedding(word_input)

        # use (last target word or last predicated word) and last context as current input
        rnn_input = torch.cat((word_embedded, last_context), 1).unsqueeze(1)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        attn_weights = self.attn(rnn_output, encoder_outputs)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 2)), dim=2)

        output = output.squeeze(1)
        context = context.squeeze(1)
        return output, context, hidden, attn_weights
