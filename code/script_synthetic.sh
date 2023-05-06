#!/bin/bash

GITHASH=$(eval "git log -1 --format='%H'")

DATAPATH='data'
GROUPNAME='synthetic'
PROJECT='neurips'
LOGDIR='./logs'
NUM_WORKERS=4

function helper_synth_share_nogpu() {
    for ((j=${13};j<${14};j+=1))
    do
        python main_synth.py \
        --gitlog $GITHASH \
        --seed $j \
        --num_workers $NUM_WORKERS \
        --num_epochs $2 \
        --net 'synth' \
        --lr $3 \
        --batch_size $4 \
        --weight_decay 0 \
        --transfer ${5} \
        --use_transfer_until ${2} \
        --lamda ${6} \
        --lamda_l2 ${7} \
        --share_last_layer \
        --num_moments ${8} \
        --mean_computation_dataset ${9} \
        --mean_computation_batch_size ${10} \
        --shuffle 'True' \
        --weighted_sharing \
        --lr_alpha ${11} \
        --lrl2_alpha ${12} \
        --h_dim ${15} \
        --out_dim ${16} \
        --use_weight_transfer_after ${17} \
        --shuffle 'True' \
        --logdir $LOGDIR \
        --group_name $GROUPNAME \
        --project $PROJECT \
        --data_path $DATAPATH \
        --compute_divergence 

    done
}

function helper_synth_noshare_nogpu() {
    for ((j=${13};j<${14};j+=1))
    do
        python main_synth.py \
        --gitlog $GITHASH \
        --seed $j \
        --num_workers $NUM_WORKERS \
        --num_epochs $2 \
        --net 'synth' \
        --lr $3 \
        --batch_size $4 \
        --weight_decay 0 \
        --transfer ${5} \
        --use_transfer_until ${2} \
        --lamda ${6} \
        --lamda_l2 ${7} \
        --num_moments ${8} \
        --mean_computation_dataset ${9} \
        --mean_computation_batch_size ${10} \
        --shuffle 'True' \
        --weighted_sharing \
        --lr_alpha ${11} \
        --lrl2_alpha ${12} \
        --h_dim ${15} \
        --out_dim ${16} \
        --shuffle 'True' \
        --logdir $LOGDIR \
        --group_name $GROUPNAME \
        --project $PROJECT \
        --data_path $DATAPATH \
        --compute_divergence 

    done
}



helper_synth_noshare_nogpu 0 3000 0.1 64 'none' 0.0 0.0 1 'train-subset' 256 0.0 0.0 4 5 2 2 &
helper_synth_noshare_nogpu 1 3000 0.1 64 'fedcda' 1.0 0.0 1 'train-subset' 256 1.0 0.0 4 5 2 2 &
helper_synth_share_nogpu 2 3000 0.1 64 'fedcda' 0.0 0.01 1 'train-subset' 256 0.0 100.0 4 5 2 2 0 &
helper_synth_share_nogpu 3 3000 0.1 64 'fedcda' 1.0 0.01 1 'train-subset' 256 1.0 100.0 4 5 2 2 0 &
