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
mkdir $FPPATH/fp_lj_sentiment/
python $FPPATH/train.py --cuda -o $FPPATH/fp_lj_sentiment/ --log-file $FPPATH/fp_lj_sentiment/nvlog_fp_lj_sentiment.json --dataset-path $LJPATH --training-files $LJPATH/filelists/train_filelist.txt --validation-files $LJPATH/filelists/val_filelist.txt --pitch-mean-std-file $LJPATH/pitch_phone_stats__transcript.json --data-inputs-file $LJPATH/sentiment_fp_data.json --model-conditions Phon-lj-ctxt-utt --epochs 500 --optimizer lamb -lr 0.1 -bs 32