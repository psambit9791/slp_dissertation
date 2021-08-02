#!/bin/sh
#
# grid engine options
#$ -cwd
#$ -l h_rt=48:00:00
#$ -l h_vmem=50G
#$ -pe gpu-titanx 1
#$ -o /exports/eddie/scratch/s1769454/train_log/fastpitch_train.stdout
#$ -e /exports/eddie/scratch/s1769454/train_log/fastpitch_train.stderr
#$ -M s1769454@ed.ac.uk
#$ -m beas
#
# initialise environment modules
. /etc/profile.d/modules.sh

module load anaconda
module load cuda/11.0.2
. /exports/applications/support/set_cuda_visible_devices.sh

source activate fp_venv

set -euo pipefail

FPPATH="/exports/eddie/scratch/s1769454/slp_dissertation/mod_fp/context-aware-tts"
BLIZZARDPATH="/exports/eddie/scratch/s1769454/data/Blizzard-2018"
LJPATH="/exports/eddie/scratch/s1769454/data/LJSpeech-1.1"

# Your python commands below...
rm -rf $FPPATH/fp_lj_sentiment/*
python $FPPATH/train.py --cuda -o $FPPATH/fp_lj_sentiment/ --log-file $FPPATH/fp_lj_sentiment/nvlog_fp_lj_sentiment.json --dataset-path $LJPATH --training-files $LJPATH/filelists/train_filelist.txt --validation-files $LJPATH/filelists/val_filelist.txt --pitch-mean-std-file $LJPATH/pitch_phone_stats__transcript.json --data-inputs-file $LJPATH/sentiment_fp_data.json --model-conditions Phon-lj-ctxt-utt --epochs 500 --epochs-per-checkpoint 50 --optimizer lamb -lr 0.1 -bs 32