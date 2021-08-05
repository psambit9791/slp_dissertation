#!/bin/bash

source activate fp_venv

FPPATH="/disk/scratch1/s1769454/slp_dissertation/mod_fp/context-aware-tts"
BLIZZARDPATH="/disk/scratch1/data/Blizzard-2018"
LJPATH="/disk/scratch1/data/LJSpeech-1.1"


# Your python commands below...
rm -rf $FPPATH/fp_lj_sentiment/*
python $FPPATH/train.py --cuda -o $FPPATH/fp_lj_sentiment/ --log-file $FPPATH/fp_lj_sentiment/nvlog_fp_lj_sentiment.json --dataset-path $LJPATH --training-files $LJPATH/filelists/train_filelist.txt --validation-files $LJPATH/filelists/val_filelist.txt --pitch-mean-std-file $LJPATH/pitch_phone_stats__transcript.json --data-inputs-file $LJPATH/sentiment_fp_data.json --model-conditions Phon-lj-ctxt-utt --epochs 500 --epochs-per-checkpoint 50 --optimizer lamb -lr 0.1 -bs 32
