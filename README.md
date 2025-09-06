# FairGDA

## source domain
在source domain (数据集B_0)中进行训练：
```
python train.py

```


在source domain (数据集B_0)中进行测试：

```
python test.py

```

## target domain

### 在target domain (数据集B_1)中进行fine：


result:

```
python data_aug.py --inid='_B1' --seed=14
```
Namespace(dataset='bail', inid='_B1', runs=5, encoder='GCN', prop='scatter', hidden=16, seed=14, gpu=0, dropout=0.5, top_k=10, alpha=1, fairnet_lr=0.01, baisnet_lr=0.01, discri_F_lr=0.01, discri_B_lr=0.01, classify_lr=0.01, encoder_lr=0.01, de_feature_lr=0.001, de_edge_lr=0.001, epochs=15, df_epochs=10, db_epochs=10, class_epochs=40, ad_MLP_F_epochs=20, align_epochs=10, de_train=5, de_traintype_switch=0, de_together_epochs=5, de_separate_epochs=3, de_separate_node_epochs=5, de_separate_edge_epochs=5, strlist=None, device=device(type='cpu'))
===========_B1_B1============
Acc     : ['0.79603', '0.78610', '0.75722', '0.73736', '0.74097']
auc_roc : ['0.78084', '0.71258', '0.70195', '0.59938', '0.55555']
F1      : ['0.55336', '0.35422', '0.19701', '0.00683', '0.04651']
parity  : ['0.03493', '0.01555', '0.01180', '0.00207', '0.00709']
equality: ['0.02930', '0.00721', '0.00176', '0.00165', '0.01064']
===========_B1_B1============
Acc     :76.35379
auc_roc :67.00603
F1      :23.15871
parity  :1.42853
equality:1.01114



(pytorch) zhangxin@zhangxindeMacBook-Pro FatraGNN-main % 
```
python data_aug.py --inid='_B1' --seed=19
```
Namespace(dataset='bail', inid='_B1', runs=5, encoder='GCN', prop='scatter', hidden=16, seed=19, gpu=0, dropout=0.5, top_k=10, alpha=1, fairnet_lr=0.01, baisnet_lr=0.01, discri_F_lr=0.01, discri_B_lr=0.01, classify_lr=0.01, encoder_lr=0.01, de_feature_lr=0.001, de_edge_lr=0.001, epochs=15, df_epochs=10, db_epochs=10, class_epochs=40, ad_MLP_F_epochs=20, align_epochs=10, de_train=5, de_traintype_switch=0, de_together_epochs=5, de_separate_epochs=3, de_separate_node_epochs=5, de_separate_edge_epochs=5, strlist=None, device=device(type='cpu'))
===========_B1_B1============
Acc     : ['0.75271', '0.80415', '0.77166', '0.75271', '0.66968']
auc_roc : ['0.74127', '0.69556', '0.72395', '0.60875', '0.59102']
F1      : ['0.52759', '0.47202', '0.32172', '0.25543', '0.32967']
parity  : ['0.02863', '0.00740', '0.00140', '0.00440', '0.01485']
equality: ['0.04559', '0.01852', '0.01567', '0.01663', '0.00700']
===========_B1_B1============
Acc     :75.01805 
auc_roc :67.21092
F1      :38.12853
parity  :1.13371
equality:2.06803







### 在target domain (数据集B_2)中进行fine：
result:

(pytorch) zhangxin@zhangxindeMacBook-Pro FatraGNN-main % python data_aug.py --inid='_B2' --seed=24
Namespace(dataset='bail', inid='_B2', runs=5, encoder='GCN', prop='scatter', hidden=16, seed=24, gpu=0, dropout=0.5, top_k=10, alpha=1, fairnet_lr=0.01, baisnet_lr=0.01, discri_F_lr=0.01, discri_B_lr=0.01, classify_lr=0.01, encoder_lr=0.01, de_feature_lr=0.001, de_edge_lr=0.001, epochs=15, df_epochs=10, db_epochs=10, class_epochs=40, ad_MLP_F_epochs=20, align_epochs=10, de_train=5, de_traintype_switch=0, de_together_epochs=5, de_separate_epochs=3, de_separate_node_epochs=5, de_separate_edge_epochs=5, strlist=None, device=device(type='cpu'))
===========_B2_B2============
Acc     : ['0.86895', '0.72871', '0.74374', '0.72621', '0.69866']
auc_roc : ['0.86854', '0.83888', '0.82799', '0.76248', '0.75568']
F1      : ['0.73070', '0.17722', '0.26024', '0.16327', '0.58553']
parity  : ['0.07045', '0.01461', '0.00675', '0.01645', '0.14049']
equality: ['0.04863', '0.01396', '0.00034', '0.01617', '0.01857']
===========_B2_B2============
Acc     :75.32554
auc_roc :81.07134
F1      :38.33917
parity  :4.97489
equality:1.95353











