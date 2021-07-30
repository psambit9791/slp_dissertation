# adapted from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
# MIT License
#
# Copyright (c) 2018 MagicGirl Sakura
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, ref_enc_input_dim, gru_dim=128, filters=[32, 32, 64, 64, 128, 128]):
        super().__init__()

        self.ref_enc_filters = filters
        self.ref_enc_strides = [2, 2]
        self.ref_enc_size = [3, 3]
        self.ref_enc_pad = [1, 1]

        self.ref_enc_gru_size = gru_dim

        K = len(self.ref_enc_filters)
        filters = [1] + self.ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=self.ref_enc_filters[i])
             for i in range(K)])

        self.n_mel_channels = ref_enc_input_dim

        out_channels = self.calculate_channels(self.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=self.ref_enc_filters[-1] * out_channels,
                          hidden_size=self.ref_enc_gru_size,
                          batch_first=True)
        self.ref_enc_gru_size = self.ref_enc_gru_size

    def forward(self, inputs):

        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)

        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        _, out = self.gru(out)

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, token_embedding_size//2]
    '''
    def __init__(self, tokens=10, reference_dim=128):
        super().__init__()
        tokens = 10
        reference_dim = 128
        print('GST settings: Nº tokens =', tokens, 'Reference dim =', reference_dim)
        self.token_embedding_size = 384 # hidden dimension model
        self.token_num = tokens
        self.num_heads = 8

        self.embed = nn.Parameter(torch.FloatTensor(self.token_num, self.token_embedding_size // self.num_heads))
        d_q = reference_dim
        d_k = self.token_embedding_size // self.num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=self.token_embedding_size,
            num_heads=self.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, token_embedding_size // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class GST(nn.Module):
    def __init__(self, tokens=10, reference_dim=384):
        super().__init__()

        #self.encoder = ReferenceEncoder(ref_enc_input_dim)
        self.tokens = tokens
        self.reference_dim = reference_dim
        self.stl = STL(self.tokens, self.reference_dim)

    def forward(self, inputs):

        #enc_out = self.encoder(inputs)
        #if style_code is not None:
        #    enc_out = torch.cat((enc_out, style_code), dim=1)
        style_embed = self.stl(inputs)

        return style_embed.squeeze(1)


class lmPredictor(nn.Module):
    def __init__(self):
        super(lmPredictor, self).__init__()

        self.layers = nn.Sequential(
                      nn.Linear(256, 128),
                      nn.BatchNorm1d(128),
                      nn.Linear(128, 64),
                      nn.BatchNorm1d(64),
                      nn.Linear(64, 64),
                      nn.Dropout(p=0.1),
                      nn.Linear(64, 128),
                      nn.BatchNorm1d(128),
                      nn.Linear(128, 256),
                      nn.ReLU()
                      )


    def forward(self, inputs):

        x = self.layers(inputs)

        return x

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn

class Attention(torch.nn.Module):

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

    def forward(self,
        query: torch.Tensor,  # [decoder_dim]
        values: torch.Tensor, # [seq_length, encoder_dim]
        ):
        weights = self._get_weights(query, values) # [seq_length]
        weights = torch.nn.functional.softmax(weights, dim=0)
        return weights @ values  # [encoder_dim]

class AdditiveAttention(Attention):

    def __init__(self, encoder_dim, decoder_dim):
        super().__init__(encoder_dim, decoder_dim)
        self.v = torch.nn.Parameter(
            torch.FloatTensor(self.decoder_dim).uniform_(-0.1, 0.1))
        self.W_1 = torch.nn.Linear(self.decoder_dim, self.decoder_dim)
        self.W_2 = torch.nn.Linear(self.encoder_dim, self.decoder_dim)

    def _get_weights(self,
        query: torch.Tensor,  # [decoder_dim]
        values: torch.Tensor,  # [seq_length, encoder_dim]
    ):
        query = query.repeat(values.size(0), 1)  # [seq_length, decoder_dim]
        weights = self.W_1(query) + self.W_2(values)  # [seq_length, decoder_dim]
        return torch.tanh(weights) @ self.v  # [seq_length]

'''
encoder_dim, decoder_dim, encoder_seq_length, decoder_seq_length = 4, 15, 3, 10

attention = AdditiveAttention(encoder_dim, decoder_dim)
encoder_hidden_states = torch.rand(encoder_seq_length, encoder_dim)
decoder_hidden_states = torch.rand(decoder_seq_length, decoder_dim)
weights = torch.FloatTensor(decoder_seq_length, encoder_seq_length)
for step in range(decoder_seq_length):
    context_vector = attention(decoder_hidden_states[step], encoder_hidden_states)
    weights[step] = attention._get_weights(decoder_hidden_states[step], encoder_hidden_states)
    print(weights[step])
    print(torch.argmax(weights[step]))

fig = seaborn.heatmap(weights.detach().numpy())
fig.figure.savefig("output3.png")
'''
