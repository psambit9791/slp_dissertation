#!/bin/bash

export OMP_NUM_THREADS=1

: ${GPU_ID:=$1}
: ${NUM_GPUS:=1}
: ${GRAD_ACCUMULATION:=1}
: ${AMP:=false}
: ${BS:=32}

[ "$AMP" == "true" ] && AMP_FLAG="--amp"

# Adjust env variables to maintain the global batch size
#
#    NGPU x BS x GRAD_ACC = 256.
#
GBS=$(($NUM_GPUS * $BS * $GRAD_ACCUMULATION))
[ $GBS -ne 256 ] && echo -e "\nWARNING: Global batch size changed from 256 to ${GBS}.\n"

FPPATH="/exports/eddie/scratch/s1769454/slp_dissertation/mod_fp/context-aware-tts"
BLIZZARDPATH="/exports/eddie/scratch/s1769454/data/Blizzard-2018"
LJPATH="/exports/eddie/scratch/s1769454/data/LJSpeech-1.1"

echo -e "\nSetup: ${NUM_GPUS}x${BS}x${GRAD_ACCUMULATION} - global batch size ${GBS}\n"

gpu_id=$(python $FPPATH/gpu_utils/gpu_lock.py --id-to-hog)
export CUDA_VISIBLE_DEVICES=$gpu_id

# Your python commands below...
mkdir -p $FPPATH/fp_bl_sentiment/
python -m torch.distributed.launch --nproc_per_node 1 $FPPATH/train.py --cuda -o $FPPATH/fp_bl_sentiment/ --log-file $FPPATH/fp_bl_sentiment/nvlog_fp_bl_sentiment.json --dataset-path $BLIZZARDPATH --training-files $BLIZZARDPATH/filelists/train_filelist.txt --validation-files $BLIZZARDPATH/filelists/val_filelist.txt --pitch-mean-std-file $BLIZZARDPATH/pitch_phone_stats__transcript.json --data-inputs-file $BLIZZARDPATH/sentiment_fp_data.json --model-conditions Phon-blizzard-ctxt-utt --epochs 500 --optimizer lamb -lr 0.1 -bs 32


$FPPATH/gpu_utils/gpu_lock.py --free $gpu_id