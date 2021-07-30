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

import numpy as np

import torch
import re
import json

from common.utils import to_gpu
from tacotron2.data_function import TextMelLoader
from common.text.text_processing import TextProcessing, PhonemeProcessing


class TextMelAliLoader(TextMelLoader):
    """
    """
    def __init__(self, *args):
        super(TextMelAliLoader, self).__init__(*args)

        self.n_speakers = args[-1].n_speakers
        self.model_conditions = args[-1].model_conditions
        self.dataset_path = args[-1].dataset_path
        self.data_inputs = json.load(open(args[-1].data_inputs_file, 'r'))

        # Select correct text processing function
        if 'Char' in self.model_conditions:
            self.tp = TextProcessing(args[-1].symbol_set, args[-1].text_cleaners)
        elif 'Phon' in self.model_conditions:
            self.tp = PhonemeProcessing()

        # Check inputs are minimum
        if len(self.audiopaths_and_text[0]) < 2:
            raise ValueError('Filelist needs to be at least <fileID>|<phones>')
        if 'mels' not in self.data_inputs or 'phone_durs' not in self.data_inputs or 'pitch_phone' not in self.data_inputs:
            raise ValueError('Inputs needs to be at least mels, phone_durs, pitch_phone')

    def __getitem__(self, index):

        # separate filename and text
        if self.n_speakers > 1:
            filename, text, speaker = self.audiopaths_and_text[index]
        else:
            filename, text = self.audiopaths_and_text[index]
            speaker = None

        dataset_path = '/'.join(filename.split('/')[:-1])+'/'
        filename = filename.split('/')[-1]

        outputs = {}

        # Bring in first the basic inputs: mels, durs, pitch
        len_text = len(text)
        text = self.get_text(text)
        mel = self.get_mel(dataset_path+self.data_inputs['mels']['path']+filename+self.data_inputs['mels']['ext'])
        dur = torch.load(dataset_path+self.data_inputs['phone_durs']['path']+filename+self.data_inputs['phone_durs']['ext'])
        pitch = torch.load(dataset_path+self.data_inputs['pitch_phone']['path']+filename+self.data_inputs['pitch_phone']['ext'])

        outputs['text'] = text
        outputs['mels'] = mel
        outputs['len_text'] = len_text
        outputs['dur'] = dur
        outputs['pitch'] = pitch
        outputs['speaker'] = speaker

        # Bring in inputs for context-aware model

        # First, find the corresponding "previous sentence"
        if 'ctxt' in self.model_conditions:

            # Find previous sentence through regex or filename format
            if 'ibm' in self.model_conditions:
                # pos = int(filename.split('_')[-1])
                # if pos != 0: p_pos = pos - 1
                # else: p_pos = pos
                # prev_basename = '_'.join(filename.split('_')[:-1]+[str(p_pos)])
                prev_basename = filename

            elif 'lj' in self.model_conditions:
                # chapter = re.findall(r'LJ([0-9]+)\-', filename)[0]
                # pos = int(re.findall(r'LJ[0-9]+\-([0-9]+)', filename)[0])
                # if pos != 1: p_pos = pos - 1
                # else: p_pos = pos
                # prev_basename = 'LJ'+chapter+'-'+str(p_pos).zfill(4)
                prev_basename = filename

            elif 'blizzard' in self.model_conditions:
                # chapter = re.findall(r'LJ([0-9]+)\-', filename)[0]
                # pos = int(re.findall(r'LJ[0-9]+\-([0-9]+)', filename)[0])
                # if pos != 1: p_pos = pos - 1
                # else: p_pos = pos
                # prev_basename = 'LJ'+chapter+'-'+str(p_pos).zfill(4)
                prev_basename = filename

            else:
                raise ValueError('You need to add a condition to find utterance sentiment embedding for your dataset')

            # Load utterance level context if present
            if 'utt' in self.model_conditions and 'utt_level_ctxt' not in self.data_inputs:
                raise ValueError('Model condition utt context, but no utt level inputs in data json provided')

            if 'utt_level_ctxt' in self.data_inputs:
                prev_sample_path = dataset_path+self.data_inputs['utt_level_ctxt']['path']+prev_basename+self.data_inputs['utt_level_ctxt']['ext']
                if self.data_inputs['utt_level_ctxt']['ext'] == '.npy':
                    utt_ctxt = torch.from_numpy(np.load(prev_sample_path))
                elif self.data_inputs['utt_level_ctxt']['ext'] == '.pt':
                    utt_ctxt = torch.load(prev_sample_path)
                else:
                    raise ValueError('Utt level ctxt format not supported, use either .npy or .pt')
                # Confirm dim of utt level features
                if len(utt_ctxt.shape) != 1:
                    raise ValueError('Utt level features should be one-dimensional')

                outputs['utt_level_ctxt'] = utt_ctxt

            # Load word level context if present
            if 'word' in self.model_conditions and 'word_level_ctxt' not in self.data_inputs:
                raise ValueError('Model condition word context, but no word level inputs in data json provided')

            if 'word_level_ctxt' in self.data_inputs:
                prev_sample_path = dataset_path+self.data_inputs['word_level_ctxt']['path']+prev_basename+self.data_inputs['word_level_ctxt']['ext']
                if self.data_inputs['word_level_ctxt']['ext'] == '.npy':
                    word_ctxt = torch.from_numpy(np.load(prev_sample_path))
                elif self.data_inputs['word_level_ctxt']['ext'] == '.pt':
                    word_ctxt = torch.load(prev_sample_path)
                else:
                    raise ValueError('Utt level ctxt format not supported, use either .npy or .pt')
                outputs['word_level_ctxt'] = word_ctxt

                # Confirm dim of word level features
                if len(word_ctxt.shape) != 2:
                    raise ValueError('Word level features should be two-dimensional')

                # Confirm we have word lengths
                if 'word_lengths' not in self.data_inputs:
                    raise ValueError('If you use word level features, you need to provide word length to average phones at the word level')
                sample_path = dataset_path+self.data_inputs['word_lengths']['path']+filename+self.data_inputs['word_lengths']['ext']
                word_lengths = np.load(sample_path)
                outputs['word_lengths'] = word_lengths

        return outputs


class TextMelAliCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, *args):
        self.n_frames_per_step = 1  # Taco2 bckwd compat
        self.model_conditions = args[-1].model_conditions
        self.data_inputs = json.load(open(args[-1].data_inputs_file, 'r'))

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [ { } ], every item in list is a sample, and every sample is a dict
        """
        # Find max input length
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(n['text']) for n in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        # Prepare text tensor
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        input_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]]['text']
            text_padded[i, :text.size(0)] = text
            input_lengths[i] = text.size(0)

        # Prepare duration tensor
        dur_padded = torch.zeros_like(text_padded, dtype=batch[0]['dur'].dtype)
        dur_lens = torch.zeros(dur_padded.size(0), dtype=torch.int32)
        for i in range(len(ids_sorted_decreasing)):
            dur = batch[ids_sorted_decreasing[i]]['dur']
            dur_padded[i, :dur.shape[0]] = dur
            dur_lens[i] = dur.shape[0]
            assert dur_lens[i] == input_lengths[i]

        # Prepare mel tensor
        num_mels = batch[0]['mels'].size(0)
        max_target_len = max([n['mels'].size(1) for n in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (self.n_frames_per_step - max_target_len
                               % self.n_frames_per_step)
            assert max_target_len % self.n_frames_per_step == 0

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]]['mels']
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        # Prepare pitch tensor
        pitch_padded = torch.zeros(dur_padded.size(0), dur_padded.size(1),
                                   dtype=batch[0]['pitch'].dtype)
        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]]['pitch']
            pitch_padded[i, :pitch.shape[0]] = pitch

        # Prepare speaker tensor
        if batch[0]['speaker'] is not None:
            speaker = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                speaker[i] = batch[ids_sorted_decreasing[i]]['speaker']
        else:
            speaker = None

        # count number of items - characters in text
        len_x = [x['len_text'] for x in batch]
        len_x = torch.Tensor(len_x)

        outputs = {'text_padded':text_padded,
                   'input_lengths':input_lengths,
                   'mel_padded':mel_padded,
                   'output_lengths':output_lengths,
                   'len_x':len_x,
                   'dur_padded':dur_padded,
                   'dur_lens':dur_lens,
                   'pitch_padded':pitch_padded,
                   'speaker':speaker,
                  }

        # Prepare context tensors
        if 'ctxt' in self.model_conditions:

            # Prepare utt level tensor
            if 'utt' in self.model_conditions:
                utt_ctxt = torch.zeros(len(batch), self.data_inputs['utt_level_ctxt']['dim'])
                for i in range(len(ids_sorted_decreasing)):
                    utt_ctxt[i] = batch[ids_sorted_decreasing[i]]['utt_level_ctxt']

                outputs['utt_level_ctxt'] = utt_ctxt

            # Prepare word level tensor
            if 'word' in self.model_conditions:
                prev_max_ctxt_len = max([x['word_level_ctxt'].size(1) for x in batch])
                word_ctxt_padded = torch.FloatTensor(len(batch), prev_max_ctxt_len, self.data_inputs['word_level_ctxt']['dim'])
                word_ctxt_padded.zero_()
                ctxt_lengths = torch.LongTensor(len(batch))
                for i in range(len(ids_sorted_decreasing)):
                    word_ctxt = batch[ids_sorted_decreasing[i]]['word_level_ctxt']
                    word_ctxt_padded[i, :word_ctxt.size(0), :] = word_ctxt
                    ctxt_lengths[i] = word_ctxt.size(0)
                word_ctxt = word_ctxt_padded

                word_lengths = []
                for i in range(len(ids_sorted_decreasing)):
                    word_lengths.append(batch[ids_sorted_decreasing[i]]['word_lengths'])

                outputs['word_level_ctxt'] = word_ctxt
                outputs['word_lengths'] = word_lengths
                outputs['ctxt_lengths'] = ctxt_lengths

        return outputs


def batch_to_gpu(batch, model_conditions):

    # start with baseline inputs
    text_padded, input_lengths, mel_padded, output_lengths = batch['text_padded'], batch['input_lengths'], batch['mel_padded'], batch['output_lengths']
    len_x, dur_padded, dur_lens, pitch_padded, speaker = batch['len_x'], batch['dur_padded'], batch['dur_lens'], batch['pitch_padded'], batch['speaker']

    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    dur_padded = to_gpu(dur_padded).long()
    dur_lens = to_gpu(dur_lens).long()
    pitch_padded = to_gpu(pitch_padded).float()

    if speaker is not None:
        speaker = to_gpu(speaker).long()

    x = [text_padded, input_lengths, mel_padded, output_lengths,
         dur_padded, dur_lens, pitch_padded, speaker]

    # process context related features
    if 'ctxt' in model_conditions and 'utt' in model_conditions:
        utt_level_ctxt = batch['utt_level_ctxt']
        utt_level_ctxt = to_gpu(utt_level_ctxt).float()
    else:
        utt_level_ctxt = None

    if 'ctxt' in model_conditions and 'word' in model_conditions:
        word_level_ctxt = batch['word_level_ctxt']
        ctxt_lengths = batch['ctxt_lengths']
        word_lengths = batch['word_lengths']

        word_level_ctxt = to_gpu(word_level_ctxt).float()
        ctxt_lengths = to_gpu(ctxt_lengths).long()

    else:
        word_level_ctxt = None
        ctxt_lengths = None
        word_lengths = None

    x.append(utt_level_ctxt)
    x.append(word_level_ctxt)
    x.append(ctxt_lengths)
    x.append(word_lengths)

    y = [mel_padded, dur_padded, dur_lens, pitch_padded]
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)
