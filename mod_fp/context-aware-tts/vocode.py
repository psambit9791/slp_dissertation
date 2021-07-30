from waveglow import model as glow
from waveglow.denoiser import Denoiser
import torch
import argparse
import models
import warnings
from scipy.io.wavfile import write

def load_and_setup_model(model_name, parser, checkpoint, amp, device,
                         unk_args=[], forward_is_infer=False, ema=True,
                         jitable=False):
    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(model_unk_args))

    model_config = models.get_model_config(model_name, model_args)

    model = models.get_model(model_name, model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)

    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint)
        status = ''

        if 'state_dict' in checkpoint_data:
            sd = checkpoint_data['state_dict']
            if ema and 'ema_state_dict' in checkpoint_data:
                sd = checkpoint_data['ema_state_dict']
                status += ' (EMA)'
            elif ema and not 'ema_state_dict' in checkpoint_data:
                print(f'WARNING: EMA weights missing for {model_name}')

            if any(key.startswith('module.') for key in sd):
                sd = {k.replace('module.', ''): v for k,v in sd.items()}
            status += ' ' + str(model.load_state_dict(sd, strict=True))
        else:
            model = checkpoint_data['model']
        print(f'Loaded {model_name}{status}')

    if model_name == "WaveGlow":
        model = model.remove_weightnorm(model)
    if amp:
        model.half()
    model.eval()
    return model.to(device)

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder to save audio (file per phrase)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--cuda', action='store_true',
                        help='Run inference on a GPU using CUDA')
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Enable cudnn benchmark mode')
    parser.add_argument('--fastpitch', type=str,
                        help='Full path to the generator checkpoint file (skip to use ground truth mels)')
    parser.add_argument('--waveglow', type=str,
                        help='Full path to the WaveGlow model checkpoint file (skip to only generate mels)')
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float,
                        help='WaveGlow sigma')
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float,
                        help='WaveGlow denoising')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--amp', action='store_true',
                        help='Inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--include-warmup', action='store_true',
                        help='Include warmup')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Repeat inference for benchmarking')
    parser.add_argument('--torchscript', action='store_true',
                        help='Apply TorchScript')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset (for loading extra data fields)')
    parser.add_argument('--speaker', type=int, default=0,
                        help='Speaker ID for a multi-speaker model')

    transform = parser.add_argument_group('transform')
    transform.add_argument('--fade-out', type=int, default=10,
                           help='Number of fadeout frames at the end')
    transform.add_argument('--pace', type=float, default=1.0,
                           help='Adjust the pace of speech')
    transform.add_argument('--pitch-transform-flatten', action='store_true',
                           help='Flatten the pitch')
    transform.add_argument('--pitch-transform-invert', action='store_true',
                           help='Invert the pitch wrt mean value')
    transform.add_argument('--pitch-transform-amplify', type=float, default=1.0,
                           help='Amplify pitch variability, typical values are in the range (1.0, 3.0).')
    transform.add_argument('--pitch-transform-shift', type=float, default=0.0,
                           help='Raise/lower the pitch by <hz>')
    transform.add_argument('--pitch-transform-custom', action='store_true',
                           help='Apply the transform from pitch_transform.py')

    text_processing = parser.add_argument_group('Text processing parameters')
    text_processing.add_argument('--text-cleaners', nargs='*',
                                 default=['english_cleaners'], type=str,
                                 help='Type of text cleaners for input text')
    text_processing.add_argument('--symbol-set', type=str, default='english_basic',
                                 help='Define symbol set for input text')

    cond = parser.add_argument_group('conditioning on additional attributes')
    cond.add_argument('--n-speakers', type=int, default=1,
                      help='Number of speakers in the model.')
    cond.add_argument('--model-conditions', type=str, required=False,
                          help='Experimental condition')
    cond.add_argument('--context-idx', type=int, required=False,
                          help='Experimental condition')
    return parser



samples = ['LJ001-0015', 'LJ001-0063', 'LJ001-0079', 'LJ001-0094', 'LJ001-0102',
           'LJ001-0153', 'LJ001-0173', 'LJ001-0186', 'LJ002-0096', 'LJ002-0171',
           'LJ002-0174', 'LJ002-0260', 'LJ002-0298', 'LJ002-0299', 'LJ005-0253',
           'LJ005-0265', 'LJ006-0044', 'LJ007-0076', 'LJ012-0189', 'LJ018-0130',
           'LJ019-0180', 'LJ019-0202', 'LJ019-0318', 'LJ021-0140', 'LJ024-0019']


device = torch.device('cuda')
parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
                                     allow_abbrev=False)
parser = parse_args(parser)
args, unk_args = parser.parse_known_args()

args.waveglow = 'pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    waveglow = load_and_setup_model(
        'WaveGlow', parser, args.waveglow, args.amp, device,
        unk_args=unk_args, forward_is_infer=True, ema=args.ema)
denoiser = Denoiser(waveglow).to(device)
waveglow = getattr(waveglow, 'infer', waveglow)


for sample in samples:

    mel = torch.load('LJSpeech-1.1/mels/'+sample+'.pt').float().cuda()
    print(mel.shape)
    with torch.no_grad():
        audios = waveglow(mel.unsqueeze(0), sigma=0.9)
        audios = denoiser(audios.float(),
                          strength=0.01
                          ).squeeze(1)

    for i, audio in enumerate(audios):
      audio = audio / torch.max(torch.abs(audio))
      audio_path = 'vocoded/voc_'+sample+'.wav'
      write(audio_path, 22050, audio.cpu().numpy())
