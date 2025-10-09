#!/bin/bash
# readme1的参数
para1=(
    --fairnet_lr=0.001
    --baisnet_lr=0.001
    --discri_F_lr=0.01
    --discri_B_lr=0.01
    --classify_lr=0.01
    --encoder_lr=0.01
    --de_feature_lr=0.001
    --de_edge_lr=0.001
    --epoch=15
    --df_epochs=20
    --db_epochs=20
    --class_epoch=40
    --ad_MLP_F_epochs=20
    --align_epochs=10
    --de_train=5
    --de_separate_epochs=3
    --de_separate_node_epochs=5
    --de_separate_edge_epochs=5
    --dataset=bail
    --inid=_B3
    --seed=46
    --runs=5
)


para2=(
    --fairnet_lr=0.001
    --baisnet_lr=0.001
    --discri_F_lr=0.01
    --discri_B_lr=0.01
    --classify_lr=0.01
    --encoder_lr=0.01
    --de_feature_lr=0.001
    --de_edge_lr=0.001
    --epoch=22
    --df_epochs=20
    --db_epochs=20
    --class_epoch=40
    --ad_MLP_F_epochs=28
    --align_epochs=15
    --de_train=5
    --de_separate_epochs=8
    --de_separate_node_epochs=5
    --de_separate_edge_epochs=5
    --dataset=bail
    --inid=_B3
    --seed=21
    --runs=5
)

LOG_FILE='result_fine/B3.txt'
echo $(date '+%Y-%m-%d %H:%M:%S')| tee -a $LOG_FILE
python data_aug_credit.py "${para2[@]}" 2>&1 | tee -a $LOG_FILE
