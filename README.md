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

```
python data_aug.py --inid='_B1' --seed=27
```
result:


===========当前运行seed 27 ===============
Namespace(dataset='bail', inid='_B1', runs=5, encoder='GCN', prop='scatter', hidden=16, seed=27, gpu=0, dropout=0.5, top_k=10, alpha=1, fairnet_lr=0.001, baisnet_lr=0.001, discri_F_lr=0.01, discri_B_lr=0.01, classify_lr=0.01, encoder_lr=0.01, de_feature_lr=0.001, de_edge_lr=0.001, epochs=15, df_epochs=20, db_epochs=20, class_epochs=40, ad_MLP_F_epochs=20, align_epochs=10, de_train=5, de_traintype_switch=0, de_together_epochs=5, de_separate_epochs=3, de_separate_node_epochs=5, de_separate_edge_epochs=5, strlist=None, device=device(type='cuda', index=0))
===========_B1_B1============
Acc     : ['0.79152', '0.78881', '0.78791', '0.74639', '0.73014']
auc_roc : ['0.78248', '0.73220', '0.80859', '0.64964', '0.55423']
F1      : ['0.56983', '0.47297', '0.33048', '0.07869', '0.00664']
parity  : ['0.00359', '0.03523', '0.00817', '0.00249', '0.00025']
equality: ['0.02913', '0.00777', '0.01191', '0.00404', '0.00392']
===========_B1_B1============
Acc     :76.89531
auc_roc :70.54283
F1      :29.17245
parity  :0.99465
equality:1.13552






### 在target domain (数据集B_2)中进行fine：
```
python data_aug.py --inid='_B2' --seed=85
```
result:


===========当前运行seed 85 ===============
Namespace(dataset='bail', inid='_B2', runs=5, encoder='GCN', prop='scatter', hidden=16, seed=85, gpu=0, dropout=0.5, top_k=10, alpha=1, fairnet_lr=0.001, baisnet_lr=0.001, discri_F_lr=0.01, discri_B_lr=0.01, classify_lr=0.01, encoder_lr=0.01, de_feature_lr=0.001, de_edge_lr=0.001, epochs=15, df_epochs=20, db_epochs=20, class_epochs=40, ad_MLP_F_epochs=20, align_epochs=10, de_train=5, de_traintype_switch=0, de_together_epochs=5, de_separate_epochs=3, de_separate_node_epochs=5, de_separate_edge_epochs=5, strlist=None, device=device(type='cuda', index=0))
===========_B2_B2============
Acc     : ['0.83222', '0.80384', '0.70534', '0.70033', '0.70117']
auc_roc : ['0.87101', '0.86471', '0.85755', '0.87762', '0.75812']
F1      : ['0.64043', '0.52138', '0.03815', 'feile!!', '0.00556']
parity  : ['0.06517', '0.02569', '0.00381', 'feile!!', '0.00161']
equality: ['0.03257', '0.01744', '0.00140', 'feile!!', '0.00151']
===========_B2_B2============
Acc     :74.85810
auc_roc :84.58020
F1      :24.11034
parity  :1.92577
equality:1.05844


### 在target domain (数据集B_3)中进行fine：
```
python data_aug.py --inid='_B3' --seed=46
```
result:
===========当前运行seed 46 ===============
Namespace(dataset='bail', inid='_B3', runs=5, encoder='GCN', prop='scatter', hidden=16, seed=46, gpu=0, dropout=0.5, top_k=10, alpha=1, fairnet_lr=0.001, baisnet_lr=0.001, discri_F_lr=0.01, discri_B_lr=0.01, classify_lr=0.01, encoder_lr=0.01, de_feature_lr=0.001, de_edge_lr=0.001, epochs=15, df_epochs=20, db_epochs=20, class_epochs=40, ad_MLP_F_epochs=20, align_epochs=10, de_train=5, de_traintype_switch=0, de_together_epochs=5, de_separate_epochs=3, de_separate_node_epochs=5, de_separate_edge_epochs=5, strlist=None, device=device(type='cuda', index=0))
===========_B3_B3============
Acc     : ['0.71000', '0.65000', '0.57429', '0.57429', '0.54786']
auc_roc : ['0.88958', '0.88313', '0.89361', '0.90678', '0.85715']
F1      : ['0.78743', '0.75549', '0.71807', '0.71992', '0.70762']
parity  : ['0.06058', '0.03281', '0.01583', '0.00329', '0.00104']
equality: ['0.05126', '0.00674', '0.00731', '0.00487', '0.00073']
===========_B3_B3============
Acc     :61.12857
auc_roc :88.60471
F1      :73.77079
parity  :2.27122
equality:1.41828



### 在target domain (数据集B_4)中进行fine：
```
python data_aug.py --inid='_B4' --seed=27
```
result:
===========当前运行seed 27 ===============
Namespace(dataset='bail', inid='_B4', runs=5, encoder='GCN', prop='scatter', hidden=16, seed=27, gpu=0, dropout=0.5, top_k=10, alpha=1, fairnet_lr=0.001, baisnet_lr=0.001, discri_F_lr=0.01, discri_B_lr=0.01, classify_lr=0.01, encoder_lr=0.01, de_feature_lr=0.001, de_edge_lr=0.001, epochs=15, df_epochs=20, db_epochs=20, class_epochs=40, ad_MLP_F_epochs=20, align_epochs=10, de_train=5, de_traintype_switch=0, de_together_epochs=5, de_separate_epochs=3, de_separate_node_epochs=5, de_separate_edge_epochs=5, strlist=None, device=device(type='cuda', index=0))
===========_B4_B4============
Acc     : ['0.85260', '0.85260', '0.79732', '0.77889', '0.68509']
auc_roc : ['0.87653', '0.85334', '0.86843', '0.77582', '0.74689']
F1      : ['0.73653', '0.70861', '0.54340', '0.54167', '0.03093']
parity  : ['0.01362', '0.01428', '0.00408', '0.00182', '0.00898']
equality: ['0.06372', '0.04580', '0.04396', '0.05726', '0.01049']
===========_B4_B4============
Acc     :79.32998
auc_roc :82.42024
F1      :51.22254
parity  :0.85540
equality:4.42471






