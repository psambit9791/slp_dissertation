# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import torch
import json
from torch import nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
from common.utils import mask_from_lens, to_gpu
from common.layers import ConvReLUNorm
from fastpitch.transformer import FFTransformer, contextFFblock
from modules.modules import GST, ReferenceEncoder


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


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout,
                 n_layers=2, out_dim=1):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(*[
            ConvReLUNorm(input_size if i == 0 else filter_size, filter_size,
                         kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)]
        )
        self.fc = nn.Linear(filter_size, out_dim, bias=True)

    def forward(self, enc_out, enc_out_mask=None):
        if enc_out_mask is not None:
            out = enc_out * enc_out_mask
        else:
            out = enc_out
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        if enc_out_mask is not None:
            out = self.fc(out) * enc_out_mask
        else:
            out = self.fc(out)
        return out.squeeze(-1)


class FastPitch(nn.Module):
    def __init__(self, n_mel_channels, max_seq_len, n_symbols, padding_idx,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size,
                 p_in_fft_dropout, p_in_fft_dropatt, p_in_fft_dropemb,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 out_fft_output_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 dur_predictor_kernel_size, dur_predictor_filter_size,
                 p_dur_predictor_dropout, dur_predictor_n_layers,
                 pitch_predictor_kernel_size, pitch_predictor_filter_size,
                 p_pitch_predictor_dropout, pitch_predictor_n_layers,
                 pitch_embedding_kernel_size, n_speakers, speaker_emb_weight,
                 model_conditions, data_inputs_file, context_idx):
        super(FastPitch, self).__init__()
        del max_seq_len  # unused

        self.model_conditions = model_conditions
        self.data_inputs = json.load(open(data_inputs_file, 'r'))

        if 'Phon' in self.model_conditions:
            n_symbols = 47 # + pad
        else:
            # characters
            n_symbols = 148

        # If word level context, add attention module for it to encoder instance
        if 'ctxt' in self.model_conditions and 'word' in self.model_conditions:
            add_ctxt_att=True
        else:
            add_ctxt_att=False

        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers, n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=True,
            d_embed=symbols_embedding_dim,
            n_embed=n_symbols,
            padding_idx=padding_idx,
            add_ctxt_att=add_ctxt_att)

        if n_speakers > 1:
            self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
        else:
            self.speaker_emb = None
        self.speaker_emb_weight = speaker_emb_weight

        if 'ctxt' in self.model_conditions:
            # Utt level context
            if 'utt' in self.model_conditions:
                self.utt_ctxt_prenet = nn.Linear(self.data_inputs['utt_level_ctxt']['dim'], 128, bias=True)
                self.gst = GST()

            # Word level context
            if 'word' in self.model_conditions :
                self.word_ctxt_prenet = nn.Linear(self.data_inputs['word_level_ctxt']['dim'], 384, bias=True) # not sure about this bias

        self.duration_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=dur_predictor_filter_size,
            kernel_size=dur_predictor_kernel_size,
            dropout=p_dur_predictor_dropout, n_layers=dur_predictor_n_layers
        )

        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )

        self.pitch_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=pitch_predictor_filter_size,
            kernel_size=pitch_predictor_kernel_size,
            dropout=p_pitch_predictor_dropout, n_layers=pitch_predictor_n_layers,
            out_dim=self.data_inputs['pitch_phone']['dim']
        )

        self.pitch_emb = nn.Conv1d(
            self.data_inputs['pitch_phone']['dim'], symbols_embedding_dim,
            kernel_size=pitch_embedding_kernel_size,
            padding=int((pitch_embedding_kernel_size - 1) / 2))

        # Store values precomputed for training data within the model
        self.register_buffer('pitch_mean', torch.zeros(1))
        self.register_buffer('pitch_std', torch.zeros(1))

        self.proj = nn.Linear(out_fft_output_size, n_mel_channels, bias=True)


    def forward(self, inputs, use_gt_durations=True, use_gt_pitch=True,
                pace=1.0, max_duration=75):

        inputs, input_lengths, mel_tgt, _, dur_tgt, _, pitch_tgt, speaker, utt_ctxt, word_ctxt, ctxt_lengths, word_lengths = inputs

        mel_max_len = mel_tgt.size(2)

        conditioning = {}

        # Calculate speaker embedding
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)
            conditioning['speaker'] = spk_emb

        # Context conditioning
        if 'ctxt' in self.model_conditions:

            # Utt level context
            if 'utt' in self.model_conditions:
                utt_ctxt = self.utt_ctxt_prenet(utt_ctxt)
                utt_ctxt = self.gst(utt_ctxt).unsqueeze(1)
            else:
                utt_ctxt = None

            conditioning['utt_ctxt'] = utt_ctxt

            # Word level context
            if 'word' in self.model_conditions:
                word_ctxt = self.word_ctxt_prenet(word_ctxt.permute(0, 2, 1))
                conditioning['word_ctxt'] = word_ctxt
                conditioning['ctxt_lengths'] = ctxt_lengths
                conditioning['word_lengths'] = word_lengths
            else:
                word_ctxt = None
                ctxt_lengths = None
                word_lengths = None

            
            conditioning['input_lengths'] = input_lengths

            enc_out, enc_mask = self.encoder(inputs, conditioning=conditioning)

        # Baseline condition
        else:
            enc_out, enc_mask = self.encoder(inputs, conditioning=conditioning)

        # Duration and Pitch model - input is encoder outputs
        pred_enc_out, pred_enc_mask = enc_out, enc_mask

        log_dur_pred = self.duration_predictor(pred_enc_out, pred_enc_mask)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        pitch_pred = self.pitch_predictor(pred_enc_out, enc_mask)

        # If replacing pitch with labels
        if 'control' in self.model_conditions:
            pitch_pred = pitch_pred.permute(0, 2, 1)
            pitch_emb = self.pitch_emb(label_tgt)

        if use_gt_pitch and pitch_tgt is not None:
            pitch_emb = self.pitch_emb(pitch_tgt.unsqueeze(1))
        else:
            pitch_emb = self.pitch_emb(pitch_pred.unsqueeze(1))

        # FORWARD
        # Add pitch
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        # Encoder output upsampling
        len_regulated, dec_lens, _ = regulate_len(
            dur_tgt if use_gt_durations else dur_pred,
            enc_out, pace, mel_max_len, model_conditions=self.model_conditions)

        # Output Tranformer - decoder
        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)

        # Project to mel dimensionality
        mel_out = self.proj(dec_out)

        return mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred

    def infer(self, inputs, context=None, pace=1.0, dur_tgt=None, pitch_tgt=None,
              pitch_transform=None, max_duration=75, speaker=0):

        # speaker embedding
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            speaker = torch.ones(inputs.size(0)).long().to(inputs.device) * speaker
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        # zero context
        if 'no_ctxt' in self.model_conditions:
            enc_out, enc_mask = self.encoder(inputs, conditioning=context)

        # context conditioned synthesis
        elif 'ctxt' in self.model_conditions:

            if 'utt' in self.model_conditions:
                context['utt_ctxt'] = self.utt_ctxt_prenet(context['utt_ctxt'])
                context['utt_ctxt'] = self.gst(context['utt_ctxt']).unsqueeze(1)

            if 'word' in self.model_conditions:
                    context['word_ctxt'] = self.word_ctxt_prenet(context['word_ctxt'])

            enc_out, enc_mask = self.encoder(inputs, conditioning=context)

        # baseline inference
        else:
            enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        # Embedded for predictors
        pred_enc_out, pred_enc_mask = enc_out, enc_mask

        log_dur_pred = self.duration_predictor(pred_enc_out, pred_enc_mask)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        pitch_pred = self.pitch_predictor(pred_enc_out, enc_mask)
        pitch_emb = self.pitch_emb(pitch_pred.unsqueeze(1))

        #if 'control' in self.model_conditions:
        #    pitch_pred = pitch_pred.permute(0, 2, 1)

        #    if 'force' in self.model_conditions:
        #        pitch_pred = context
        #        print('force prediction', pitch_pred.shape)

        #    pitch_emb = self.pitch_emb(pitch_pred)
        #else:

        enc_out = enc_out + pitch_emb.transpose(1, 2)

        len_regulated, dec_lens, reps = regulate_len(
            dur_pred if dur_tgt is None else dur_tgt,
            enc_out, pace, mel_max_len=None, model_conditions=self.model_conditions)

        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)

        mel_out = self.proj(dec_out)
        # mel_lens = dec_mask.squeeze(2).sum(axis=1).long()
        mel_out = mel_out.permute(0, 2, 1)  # For inference.py

        return mel_out, dec_lens, dur_pred, pitch_pred, reps, dec_lens
