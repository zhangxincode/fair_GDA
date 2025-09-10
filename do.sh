#! /bin/bash

# 日志文件
LOG_FILE="log.txt"

# 清空之前的日志
> $LOG_FILE

#for i in _B1 _B2 _B3 _B4
#do
#  echo "===========当前数据集：$i===============" | tee -a $LOG_FILE
#  for j in {1..10}
#  do
#    echo "===========当前运行第 $j 次===============" | tee -a $LOG_FILE
#    python data_aug.py --dataset bail --inid $i 2>&1 | tee -a $LOG_FILE
#  done
#done




for i in {1..50}
do
  echo "===========当前运行seed $i ===============" | tee -a $LOG_FILE
  acc=()
  f1=()
  auc_roc=()
  parity=()
  equality=()
  for j in {0..4}
  do
    python data_aug.py --seed $i+$j --inid _B1 2>&1 | tee -a $LOG_FILE
    acc=$(echo "$output" | grep "Accuracy:" | awk '{print $2}')
  done
done
