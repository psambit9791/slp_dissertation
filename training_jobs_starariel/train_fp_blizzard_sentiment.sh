#!/bin/bash

source activate fp_venv

FPPATH="/disk/scratch1/s1769454/slp_dissertation/mod_fp/context-aware-tts"
BLIZZARDPATH="/disk/scratch1/data/Blizzard-2018"
LJPATH="/disk/scratch1/data/LJSpeech-1.1"


# Your python commands below...
rm -rf $FPPATH/fp_bl_sentiment/*
python $FPPATH/train.py --cuda -o $FPPATH/fp_bl_sentiment/ --log-file $FPPATH/fp_bl_sentiment/nvlog_fp_bl_sentiment.json --dataset-path $BLIZZARDPATH --training-files $BLIZZARDPATH/filelists/train_filelist.txt --validation-files $BLIZZARDPATH/filelists/val_filelist.txt --pitch-mean-std-file $BLIZZARDPATH/pitch_phone_stats__transcript.json --data-inputs-file $BLIZZARDPATH/sentiment_fp_data.json --model-conditions Phon-blizzard-ctxt-utt --epochs 500 --epochs-per-checkpoint 50 --optimizer lamb -lr 0.1 -bs 32
