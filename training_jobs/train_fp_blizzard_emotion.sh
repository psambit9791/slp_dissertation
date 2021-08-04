#!/bin/sh
#
# grid engine options
#$ -cwd
#$ -l h_rt=48:00:00
#$ -l h_vmem=62G
#$ -pe gpu-titanx 1
#$ -o /exports/eddie/scratch/s1769454/train_log/fastpitch_train_fp_bl_emo.stdout
#$ -e /exports/eddie/scratch/s1769454/train_log/fastpitch_train_fp_bl_emo.stderr
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
rm -rf $FPPATH/fp_bl_emotion/*
python $FPPATH/train.py --cuda -o $FPPATH/fp_bl_emotion/ --log-file $FPPATH/fp_bl_emotion/nvlog_fp_bl_emotion.json --dataset-path $BLIZZARDPATH --training-files $BLIZZARDPATH/filelists/train_filelist.txt --validation-files $BLIZZARDPATH/filelists/val_filelist.txt --pitch-mean-std-file $BLIZZARDPATH/pitch_phone_stats__transcript.json --data-inputs-file $BLIZZARDPATH/emotion_fp_data.json --model-conditions Phon-blizzard-ctxt-utt --epochs 500 --epochs-per-checkpoint 50 --optimizer lamb -lr 0.1 -bs 32
