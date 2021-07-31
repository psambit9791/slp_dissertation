#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#  job name: -N
#$ -N test_fp_training
# This is where the log files will be stored, in this case, the current directory
#$ -cwd
# Running time
#$ -l h_rt=02:00:00
#  (working) memory limit, the amount of memory you need may vary given your code and the amount of data
#$ -l h_vmem=16G
# GPU environment
#$ -pe gpu 1

# Load Anaconda environment and modules
. /etc/profile.d/modules.sh
module load anaconda
source activate fp_venv
module load cuda/11.0.2
source /exports/applications/support/set_cuda_visible_devices.sh
FPPATH="/exports/eddie/scratch/s1769454/slp_dissertation/mod_fp/context-aware-tts"
BLIZZARDPATH="/exports/eddie/scratch/s1769454/data/Blizzard-2018"
LJPATH="/exports/eddie/scratch/s1769454/data/LJSpeech-1.1"

# Your python commands below...
mkfir $FPPATH/fp_bl_sentiment/
python $FPPATH/train.py --cuda -o $FPPATH/fp_bl_sentiment/ --log-file $FPPATH/fp_bl_sentiment/nvlog_fp_bl_sentiment.json --dataset-path $BLIZZARDPATH --training-files $BLIZZARDPATH/filelists/train_filelist.txt --validation-files $BLIZZARDPATH/filelists/val_filelist.txt --pitch-mean-std-file $BLIZZARDPATH/pitch_phone_stats__transcript.json --data-inputs-file $BLIZZARDPATH/sentiment_fp_data.json --model-conditions Phon-blizzard-ctxt-utt --epochs 500 --optimizer lamb -lr 0.1 -bs 32