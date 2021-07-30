# Context-aware TTS

At the moment the code is based on this paper:
But we will be updating it with new features are research progress.
In this example we will use the LJ dataset.

## Installation

Please follow the original FastPitch installation instructions. You will to run for this repo the script to download Waveglow.

## Data pre-processing

### Obtain alignments

Because this model is based on FastPitch, the first thing you need to do is to obtain alignments for your data, ideally through phonetic transcriptions. We recommend to use Montreal Forced Aligner: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner

Some of the scripts in subsequent steps are written for the textgrid format ouput by MFA. If you use another aligner you will need to write your own version for those scripts.

### Provided alignments

You can download phonetic alignments for the LJ dataset here: https://drive.google.com/file/d/1pOHajsaz9A2R8px01Yda3U5nk375tkZS/view?usp=sharing
These alignments were obtain after pre-processing the data and manually adding some OOVs to the existing English dictionaries available for MFA.

Notice that MFA removes any punctuation marks. We pre-processed the data to map these to string such as . : 'xxperiodxx', to be kept during alignment and in subsequent steps. These are mapped back to symbols <.> for the TTS input in the next step.


### Data folder
Set up a data folder inside which you will store all the features that we will generate. We will call it <root-folder-for-data>.

## Feature extraction

### Extract durations and word lengths from alignments

From the texgrids we extract duraction vectors. These vectors have one integer per input symbol indicating how many frame each symbol comprises. For the word-level context method we use of word lengths to average representations at the word level. These are obtained in two steps, the first script extracts word boundaries within and the second one calculates the word lengths.

```
python extract_phone_dur.py --wavsf <path-to-wav-folder> --labsf <path-to-textgrid-folder> --durf <path-to-output-duration-folder> --outf <name-of-file-to-write-with-all-transcriptions> --outwb <path-to-output-word-boundaries-folder>
python get_word_lengths.py <word-boundaries-folder-path> <output-folder-path>
```
The first script should write many files to the --durf folder and the --outwb. Also, --outf will be the name of a file were all the textgrids will be transcribed in the format filename|transcription, we will call this file transcript.txt for subsequent steps. The second one will simply fill the output folder with the word lengths per sample.

## Extract mels and pitch 

This step uses a slightly modified version of the get_mels.py original script in FastPitch.
```
python extract_feats.py --cuda --dataset-path <root-folder-for-data> --wav-text-filelist transcript.txt --extract-mels --extract-pitch-char
```

## Context Feature extraction
  
In this section we extract features for the context-aware modules of the model. In theory, you can use any feature you want, as long as for the utt-level method you vector is (1, feat_dim) and for the word-level (N, feat_dim).
I will show here the features used for the paper, Deep Spectrum and BERT.
  
You can skip this section if you want to train a baseline model.
  
### Utt-level feature: Deep Spectrum
  
First, you need to install the original Deep Spectrum repo, follow the original instructions here: https://github.com/DeepSpectrum/DeepSpectrum

After installing, use the provided script to extract features with the same specifications used in our paper. In this we extract an utt-level vector per wav file. Make sure to be in an environment where Deep Spectrum is installed before running it. Because the output is one csv file, we break it to have one file per sample.

```
bash utt_level_deep_spectrum.sh <path-to-wav-folder> <name-of-csv-output-file>
python break_ds.py <name-of-csv-output-file> <path-to-output-folder>
```

### Word-level feature: BERT
To run BERT you need to make sure to have the Transformers library installed. First, we will convert all the labels into one csv, then that will be the input to obtain the BERT features. The input is the lab files used before alignment, which is one plain text file per sample with the text. The script extracts both word-level and utt-level BERT features, but in this tutorial we will only use the word-level ones.
  
```
python prepare_text.py <lab-folder>
python extract_BERT.py text_for_bert.csv <word-level-output-path-folder> <utt-level-output-path-folder>
```
## Experiment setup

Because the context-aware model uses more inputs the file list handling can get messy. Instead of using the FastPitch format for the training list (where they have mel_path|pitch_path|dur_path|transcription), we will use a JSON file as a config of the inputs to the model. This is an example of a config with both context features:

```
{
  "mels": {"path": "mels/", "ext": ".pt", "dim": 80},
  "phone_durs": {"path": "phone_durs/", "ext": ".pt", "dim": 1},
  "pitch_phone": {"path": "pitch_phone/", "ext": ".pt", "dim": 1},
  "utt_level_ctxt": {"path": "utt_ds/", "ext": ".npy", "dim": 4096},
  "word_level_ctxt": {"path": "word_bert/", "ext": ".npy", "dim": 768},
  "word_lengths": {"path": "word_lengths/", "ext": ".npy", "dim": 1}
}
```
The "paths" correspond to the path to the folder inside the root data directory. Because we might use either numpy or torch to save the features, "ext" indicates which of the two were used for that specific feature. Finally, "dim" contains the feature dimension.

A baseline config would look like this:


```
{
  "mels": {"path": "mels/", "ext": ".pt", "dim": 80},
  "phone_durs": {"path": "phone_durs/", "ext": ".pt", "dim": 1},
  "pitch_phone": {"path": "pitch_phone/", "ext": ".pt", "dim": 1}
}
```
This allows for a clean file list transcript.txt of the format: filename|transcript 
Notice that filename doesn't need an extension. This makes it easier to test input combinations without having to write multiple filelists.

Finally, you need to split your transcript.txt into train/val/test sets.

## Training conditions

The training script is the same as the original FastPitch, with two new arguments: --data-inputs-file which corresponds to the path to the JSON file created above, and --model-conditions which is a string that indicates the model the setup architecture and other options.

The supported options at the moment are: 
- Char or Phon, to specify the form of the input (Phon supports CMU phone list without stress)
- Dataset (for context-aware model): lj/ibm, to use appropriate regex to find previous sample
- For the context-aware model you need to add "ctxt" and if wanting to use word-level "word" and/or using utt-level "utt".
For example, a full context-aware model would be: --model-conditions Phon-lj-ctxt-utt-word
A baseline would be: --model-conditions Phon
  
## Inference
  
Unlike the original FastPitch, which requires a different format of file list for inference, this code supports the same filename|transcript format, assuming that filename is the name of the output.
The inference command uses the same extra options required for training, --model-conditions (which should match the conditions the checkpoint was trained with, but also allows for inference options, like control), and the --data-inputs-file, including the option --dataset-path <root-folder-for-data>
