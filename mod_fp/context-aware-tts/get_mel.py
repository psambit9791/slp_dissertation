from common.utils import load_wav_to_torch
import common.layers as layers
import torch

def get_mel(filename, outf):

    stft = layers.TacotronSTFT()
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != 22050:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, 22050))
    audio_norm = audio / 32768.0
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    print(melspec.shape)
    torch.save(melspec, outf+filename.replace('.wav', '.pt'))

get_mel('./RG_994_hydroelectric-dams_pro_release_5_larger_chunks_chunk_29.wav', '')
