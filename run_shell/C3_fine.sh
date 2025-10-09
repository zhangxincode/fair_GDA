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
    --epoch=28
    --df_epochs=30
    --db_epochs=30
    --class_epoch=40
    --ad_MLP_F_epochs=29
    --align_epochs=15
    --de_train=5
    --de_separate_epochs=6
    --de_separate_node_epochs=6
    --de_separate_edge_epochs=5
    --dataset=credit
    --inid=_C3
    --seed=86
    --runs=5
)

LOG_FILE='result_fine/C3.txt'
echo $(date '+%Y-%m-%d %H:%M:%S')| tee -a $LOG_FILE
python data_aug_credit.py "${para1[@]}" 2>&1 | tee -a $LOG_FILE
