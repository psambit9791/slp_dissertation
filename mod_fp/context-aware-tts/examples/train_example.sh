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

echo -e "\nSetup: ${NUM_GPUS}x${BS}x${GRAD_ACCUMULATION} - global batch size ${GBS}\n"

export CUDA_VISIBLE_DEVICES=$1
../../gpu_utils/gpu_lock.py --id-to-hog $1


python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} ../../train.py \
    --cuda \
    -o ./output \
    --log-file ./output/nvlog.json \
    --dataset-path /disk/scratch/pilaroplustil/icassp_2021/RG_speaker/ \
    --training-files ./filelists/rg_phoneme_train.txt \
    --validation-files ./filelists/rg_phoneme_val.txt \
    --pitch-mean-std-file ./filelists/rg_pitch_phone_stats.json \
    --data-inputs-file ./filelists/rg_data_inputs_base.json \
    --model-conditions Phon \
    --epochs 500 \
    --epochs-per-checkpoint 100 \
    --warmup-steps 1000 \
    -lr 0.1 \
    -bs $BS \
    --optimizer lamb \
    --grad-clip-thresh 1000.0 \
    --dur-predictor-loss-scale 0.1 \
    --pitch-predictor-loss-scale 0.1 \
    --weight-decay 1e-6 \
    --gradient-accumulation-steps ${GRAD_ACCUMULATION} \
    ${AMP_FLAG}

../../gpu_utils/gpu_lock.py --free $1
