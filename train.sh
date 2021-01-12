#!/bin/bash
python main.py --gpus 0,1,2,3 --checkpoint "./model" --start-epoch 1 \
    --batchSize 384 --nEpochs 35  --lr 0.0008 --step 10 --clip 0.1 \
    --gamma 0.5 --seed 418 \
    --dataset_path '/home/hejingwei/anaconda2/envs/rlcsc/RL-CSC/data/train.h5' \
    --pretrained ""  --resume "" \
    --comment ""
