import os
import re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import soundfile as sf
import codecs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wavsf", help="path to wav folder",
                    type=str)
parser.add_argument("--labsf", help="path to textgrid folder",
                    type=str)
parser.add_argument("--durf", help="path to folder where to store durations",
                    type=str)
parser.add_argument("--outf", help="name of file where to write file|transcriptions",
                    type=str)
parser.add_argument("--outwb", help="folder where to store word boundaries",
                    type=str)
args = parser.parse_args()

# example:
# inputs
#wavsf = '/disk/scratch/pilaroplustil/icassp_2021/RG_speaker/wavs/'
#labsf = '/disk/scratch/pilaroplustil/icassp_2021/RG_speaker/clean_textgrids/' # textgrids

# outputs
#durf = '/disk/scratch/pilaroplustil/icassp_2021/RG_speaker/phone_durs/'
#outf = '/disk/scratch/pilaroplustil/icassp_2021/RG_speaker/rg_phoneme.txt'
#outwb = '/disk/scratch/pilaroplustil/icassp_2021/RG_speaker/word_boundaries/'

wavsf = args.wavsf
labsf = args.labsf
durf = args.durf
outf = args.outf
outwb = args.outwb

outf = open(outf, 'w')
all_diff = []

frame = 0.0116
for file in sorted(os.listdir(labsf)):

    if file.endswith('.TextGrid') and file.replace('.TextGrid', '.wav') in os.listdir(wavsf):
        print(file)
        transcription = []
        duration = []
        word_boundaries = []
        #if file.replace('.wav', '.lab') in os.listdir(labsf):

        lab = open(labsf+file).read().replace('\n', '<nl>').replace('\t', '')
        wtier = re.findall(r'item\s\[1\](.*)item', lab)[0]
        silences = [re.findall(r'text\s=\s(.*?)\<', n)[0][1:-1] for n in wtier.split('intervals [')[1:] if re.findall(r'text\s=\s(.*?)\<', n)[0][1:-1] == '' or re.findall(r'text\s=\s(.*?)\<', n)[0][1:-1] == 'xxcommaxx' or re.findall(r'text\s=\s(.*?)\<', n)[0][1:-1] == 'xxperiodxx' or re.findall(r'text\s=\s(.*?)\<', n)[0][1:-1] == 'xxexclamationxx' or re.findall(r'text\s=\s(.*?)\<', n)[0][1:-1] == 'xxquestionxx' or re.findall(r'text\s=\s(.*?)\<', n)[0][1:-1] == 'xxhesitationxx']
        tier = re.findall(r'item\s\[2\](.*)', lab)[0]


        word_ends = [float(re.findall(r'xmax\s=\s(.*?)\<', n)[0]) for n in wtier.split('intervals [')[1:]]

        #print(word_ends)

        s_idx = 0
        word_idx = 0
        word_boundaries = []
        phone_tier = tier.split('intervals [')[1:]
        #print(phone_tier)
        for i, phone in enumerate(phone_tier):
            # remove stress marks
            text = re.findall(r'text\s=\s(.*?)\<', phone)[0][1:-1].replace('1', '').replace('0', '').replace('2', '')
            if text == 'sp' or text == '' or text == 'sil':
                if s_idx < len(silences):
                    # at the moment both commas and full stops are matched to sp symbol
                    if silences[s_idx] == 'xxcommaxx':
                        text = '<,>'
                    elif silences[s_idx] == 'xxperiodxx':
                        text = '<.>'
                    elif silences[s_idx] == 'xxquestionxx':
                        text = '<?>'
                    elif silences[s_idx] == 'xxexclamationxx':
                        text = '<!>'
                    elif silences[s_idx] == 'xxhesitationxx':
                        text = '<uh>'
                    else:
                        text = '<sp>'
                    s_idx += 1
                else:
                    if i != len(phone_tier) - 1:
                        print('ERROR!', file)
                    else:
                        text = '<sp>'

            start = float(re.findall(r'xmin\s=\s(.*?)\<', phone)[0])
            end = float(re.findall(r'xmax\s=\s(.*?)\<', phone)[0])
            frame_dur = (end - start) / frame
            if (frame_dur % 1) >= 0.45: # 0.45 gives an average difference of 0.35 at the moment
                frame_dur = int(math.ceil(frame_dur))
            else:
                frame_dur = int(math.floor(frame_dur))

            transcription.append(text)
            duration.append(frame_dur)


            # word bondauries
            if end <= word_ends[word_idx]:
                word_boundaries.append(0)
            else:
                word_idx += 1
                word_boundaries.append(1)


        #print (silences)
        #if len(transcription) < 3:
        #    print ('ALERT!', file)
        #    print (transcription)

        #print(transcription)
        #print(duration)

        silences = ['<,>', '<.>', '<sp>']
        final_transcription = []
        final_duration = []
        final_wb = []
        for i in range(len(transcription)):
            # get rid of sp, give duration to next symbol
            if final_transcription and final_transcription[-1] == '<sp>':
                final_duration[-1] += duration[i]
                final_transcription[-1] = transcription[i]
                final_wb[-1] = word_boundaries[i]
            else:
                final_transcription.append(transcription[i])
                final_duration.append(duration[i])
                final_wb.append(word_boundaries[i])

        if final_transcription[-1] == '<sp>':
            final_duration[-2] += final_duration[-1]
            final_duration = final_duration[:-1]
            final_transcription = final_transcription[:-1]
            final_wb = final_wb[:-1]
            final_wb[-1] = 1


        #print(final_transcription)
        #print(final_duration)

        assert len(final_transcription) == len(final_duration) == len(final_wb)
        # check that overall extracted duration kind of matches wav length - to prevent sentences cut at the end in synthesis
        wav = sf.SoundFile(wavsf+file.replace('.TextGrid', '.wav'))
        wav_frames = (len(wav) / wav.samplerate) / frame
        diff = sum(final_duration) - wav_frames
        all_diff.append(diff)

        # save word boundaries
        torch.save(torch.from_numpy(np.array(final_wb)), outwb+file.replace('.TextGrid', '.pt'))

        # save dur file and write to file list
        outf.writelines(wavsf+file.replace('.TextGrid', '.wav')+'|'+' '.join(final_transcription)+'\n')
        torch.save(torch.from_numpy(np.array(final_duration)), durf+file.replace('.TextGrid', '.pt'))


#print (np.mean(all_diff))
