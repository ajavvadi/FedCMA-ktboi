#!/bin/bash

GITHASH=$(eval "git log -1 --format='%H'")

DATAPATH='data/dirichlet-CIFAR10'
GROUPNAME='cifar'
PROJECT='neurips'
LOGDIR='./logs'

function helper_cifar() {
    for ((j=${18};j<${19};j+=1))
    do
        CUDA_VISIBLE_DEVICES=$1 python main_full.py \
        --gitlog $GITHASH \
        --data_seed 0 \
        --seed $j \
        --num_workers $2 \
        --num_epochs $3 \
        --dirichlet \
        --alpha $4 \
        --net $5 \
        --use_sigmoid 'sigmoid' \
        --lr $6 \
        --batch_size ${7} \
        --weight_decay 0.0005 \
        --transfer ${8} \
        --use_weight_transfer_after ${9} \
        --use_transfer_until ${10} \
        --lamda ${11} \
        --lamda_l2 ${12} \
        --share_last_layer \
        --num_moments ${13} \
        --mean_computation_dataset ${14} \
        --mean_computation_batch_size ${15} \
        --weighted_sharing \
        --shuffle 'False' \
        --lr_alpha ${16} \
        --lr_decay 'none' \
        --lr_gamma 0.0 \
        --lrl2_alpha ${17} \
        --lrl2_decay 'none' \
        --lrl2_gamma 0.0 \
        --logdir $LOGDIR \
        --group_name $GROUPNAME \
        --project $PROJECT \
        --data_path $DATAPATH \
        --compute_divergence \
        --discr_comp_interval 30 \
        --hetero_arch 

    done
}

function helper_cifar_noshare() {
    for ((j=${18};j<${19};j+=1))
    do
        CUDA_VISIBLE_DEVICES=$1 python main_full.py \
        --gitlog $GITHASH \
        --data_seed 0 \
        --seed $j \
        --num_workers $2 \
        --num_epochs $3 \
        --dirichlet \
        --alpha $4 \
        --net $5 \
        --use_sigmoid 'sigmoid' \
        --lr $6 \
        --batch_size ${7} \
        --weight_decay 0.0005 \
        --transfer ${8} \
        --use_weight_transfer_after ${9} \
        --use_transfer_until ${10} \
        --lamda ${11} \
        --lamda_l2 ${12} \
        --num_moments ${13} \
        --mean_computation_dataset ${14} \
        --mean_computation_batch_size ${15} \
        --shuffle 'False' \
        --lr_alpha ${16} \
        --lr_decay 'none' \
        --lr_gamma 0.0 \
        --lrl2_alpha ${17} \
        --lrl2_decay 'none' \
        --lrl2_gamma 0.0 \
        --logdir $LOGDIR \
        --group_name $GROUPNAME \
        --project $PROJECT \
        --data_path $DATAPATH \
        --compute_divergence \
        --discr_comp_interval 30 \
        --hetero_arch 

    done
}

#local baseline
helper_cifar_noshare 0 20 200 0.1 'lenet' 0.01 256 'none' 0 0 0 0 1 'train-subset' 256 0.0 0.0 0 1 &

helper_cifar 1 20 200 0.1 'lenet' 0.01 256 'fedcda' 0 200 1.0 5.0 1 'train-subset' 256 0.18 0.2 0 1 &
helper_cifar 2 20 200 0.1 'lenet' 0.01 256 'fedcda-marginal' 0 200 1.0 5.0 1 'train-subset' 256 0.18 0.2 0 1 &

#without sharing e
helper_cifar 3 20 200 0.1 'lenet' 0.01 256 'fedcda' 0 200 0.0 5 1 'train-subset' 256 0.0 0.2 0 1 &
#without sharing v
helper_cifar_noshare 0 20 200 0.1 'lenet' 0.01 256 'fedcda' 0 200 1.0 0.0 1 'train-subset' 256 0.18 0.0 0 1 &
