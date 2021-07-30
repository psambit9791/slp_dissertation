# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.layers import LinearNorm
from common.utils import mask_from_lens
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def regulate_len(durations, enc_out, pace=1.0, mel_max_len=None, model_conditions=None):
    """If target=None, then predicted durations are applied"""
    reps = torch.round(durations.float() / pace).long()
    dec_lens = reps.sum(dim=1)
    enc_rep = pad_sequence([torch.repeat_interleave(o, r, dim=0)
                            for o, r in zip(enc_out, reps)],
                           batch_first=True)
    if mel_max_len:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
    return enc_rep, dec_lens, reps

class TacotronAttention(nn.Module):
    def __init__(self, attention_rnn_dim, context_dim, attention_dim=192):
        super(TacotronAttention, self).__init__()

        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(context_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)

    def get_alignment_energies(self, enc_word, processed_context):
        query = enc_word
        key = processed_context
        processed_query = self.query_layer(query.unsqueeze(1)).repeat(1, key.size(1), 1)
        #print('processed_query', processed_query.shape)
        #print('key', key.shape)
        energies = self.v(torch.tanh(processed_query + key))
        energies = energies.squeeze(-1)
        #print(energies.shape, 'energies')

        return energies

    def forward(self, enc_word, context):
        alignment = self.get_alignment_energies(enc_word, self.memory_layer(context))
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), context)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class contextFFblock(nn.Module):
    def __init__(self, d_model, d_input, d_inner, kernel_size, dropout, pre_lnorm=False):
        super(contextFFblock, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.linear = nn.Linear(d_input, self.d_model)
        self.pos_emb = PositionalEmbedding(self.d_model)

        self.CoreNet = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size, 1, (kernel_size // 2)),
            nn.ReLU(),
            # nn.Dropout(dropout),  # worse convergence
            nn.Conv1d(d_inner, d_model, kernel_size, 1, (kernel_size // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp, mask):
        return self._forward(inp, mask)

    def _forward(self, inp, seq_lens):

        inp = self.linear(inp)
        mask = mask_from_lens(seq_lens).unsqueeze(2)
        pos_seq = torch.arange(inp.size(1), device=inp.device, dtype=inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask
        inp = inp + pos_emb

        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(self.layer_norm(core_out))
            core_out = core_out.transpose(1, 2)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout, pre_lnorm=False):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size, 1, (kernel_size // 2)),
            nn.ReLU(),
            # nn.Dropout(dropout),  # worse convergence
            nn.Conv1d(d_inner, d_model, kernel_size, 1, (kernel_size // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        return self._forward(inp)

    def _forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(self.layer_norm(core_out))
            core_out = core_out.transpose(1, 2)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.1,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head # = 64
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp, attn_mask=None):
        return self._forward(inp, attn_mask)

    def _forward(self, inp, attn_mask=None):
        residual = inp

        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp)

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=-1)

        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)

        q = head_q.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)

        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)
            attn_score.masked_fill_(attn_mask, -float('inf'))

        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(
            inp.size(0), inp.size(1), n_head * d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out)

        return output

    # disabled; slower
    def forward_einsum(self, h, attn_mask=None):
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [bsz x n_head x qlen x klen]
        # attn_score = torch.einsum('ibnd,jbnd->bnij', (head_q, head_k))
        attn_score = torch.einsum('bind,bjnd->bnij', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            attn_score.masked_fill_(attn_mask[:, None, None, :], -float('inf'))

        # [bsz x qlen x klen x n_head]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # [bsz x n_head x qlen x klen] * [klen x bsz x n_head x d_head]
        #     -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('bnij,bjnd->bind', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, kernel_size, dropout,
                 **kwargs):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseConvFF(d_model, d_inner, kernel_size, dropout,
                                         pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, mask=None):
        output = self.dec_attn(dec_inp, attn_mask=~mask.squeeze(2))
        output *= mask
        output = self.pos_ff(output)
        output *= mask
        return output


class FFTransformer(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, kernel_size,
                 dropout, dropatt, dropemb=0.0, embed_input=True,
                 n_embed=None, d_embed=None, padding_idx=0, pre_lnorm=False, add_ctxt_att=False):
        super(FFTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.padding_idx = padding_idx

        if add_ctxt_att:
            self.additive_attention = TacotronAttention(384, 384)

        if embed_input:
            self.word_emb = nn.Embedding(n_embed, d_embed or d_model,
                                         padding_idx=self.padding_idx)
        else:
            self.word_emb = None

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layer): # n_layer = 6
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, kernel_size, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm)
            )

    def average_feats(self, feature, feature_lens, feature_word_lens):

        max_word = max([j.shape[0] for j in feature_word_lens])
        # output will have the resolution on words rather than phonemes or frames
        feature_words = torch.zeros((feature.size(0), max_word, feature.size(2)))

        # loop over each sample
        for i in range(0, feature.size(0)):
            # split given the word durations, without padding
            split_feature = torch.split(feature[i][:feature_lens[i]], list(feature_word_lens[i]))
            # average each segment that splitted
            averages = [torch.mean(n.unsqueeze(0), dim=1) for n in split_feature]
            # fix nan problem for torch.mean with apex
            av_words = [torch.from_numpy(np.nan_to_num(np.array(av.detach().cpu().numpy()))).cuda() for av in averages]
            # pad to maximum length
            [av_words.append(torch.zeros((1, feature.size(2))).cuda()) for _ in range(len(av_words), max_word)]
            # stack
            av_words = torch.vstack(av_words)
            # assign to output matrix
            feature_words[i, :av_words.size(0), :] = av_words
        return feature_words.cuda(), max_word

    def alignContext(self, context, enc_out):

        enc_out_dim = enc_out.shape[1]
        max_weigths = torch.LongTensor(enc_out.size(0), enc_out_dim)

        for step in range(enc_out_dim):

            enc_word = enc_out[:,step]
            _, weights = self.additive_attention(enc_word, context)
            max_weigths[:,step] = torch.argmax(weights, dim=1)

        return max_weigths # index of the context word in order for each encoder step


    def word_level_context(self, enc_out, inputs):
        prev_ctxt, input_lengths, word_lengths = inputs

        # get word level context
        context_words = prev_ctxt

        # average at word level the encoder outputs (phoneme transcription of the current sentence)
        enc_out_words, max_word = self.average_feats(enc_out, input_lengths, word_lengths)

        # align and obtain weights
        max_weights = self.alignContext(context_words, enc_out_words)
        #print(max_weights) # inspect that weights are different

        # repeat weights to gather later
        max_weights = max_weights.unsqueeze(2).repeat(1, 1, context_words.size(2)).cuda()

        ## gather the word level feats in the context that were relevant given the attention weights
        prev_ctxt = torch.gather(context_words, 1, max_weights)

        ## then we need to upsample those back to the encoder output resolution again, given the lengths of the encoder outputs
        padded_word_lengths = []
        for b in range(enc_out.size(0)):
            nwl = word_lengths[b]
            for _ in range(len(word_lengths[b]), max_word):
                nwl = np.append(nwl, 0)
            padded_word_lengths.append(nwl)

        padded_word_lengths = np.array(padded_word_lengths)
        prev_ctxt, _, _ = regulate_len(torch.from_numpy(padded_word_lengths).long().cuda(), prev_ctxt)

        return prev_ctxt, max_weights

    def forward(self, dec_inp, seq_lens=None, conditioning=None):
        if self.word_emb is None:
            inp = dec_inp
            mask = mask_from_lens(seq_lens).unsqueeze(2)
        else:
            inp = self.word_emb(dec_inp)
            # [bsz x L x 1]
            mask = (dec_inp != self.padding_idx).unsqueeze(2)

        pos_seq = torch.arange(inp.size(1), device=inp.device, dtype=inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask

        out = self.drop(inp + pos_emb)

        if conditioning is not None:
            if 'utt_ctxt' in conditioning:
                out += conditioning['utt_ctxt']

            if 'word_ctxt' in conditioning:
                word_ctxt, max_weights = self.word_level_context(out, [conditioning['word_ctxt'], conditioning['input_lengths'], conditioning['word_lengths']])
                out += word_ctxt

        for layer in self.layers:
            out = layer(out, mask=mask)

        # out = self.drop(out)
        return out, mask