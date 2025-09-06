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
result:
```
(pytorch) zhangxin@zhangxindeMacBook-Pro FatraGNN-main % python test.py
Namespace(dataset='bail', inid='_B0', outid='', dropout=0.5, top_k=10, alpha=1, runs=1, hidden=16, d_lr=0.01, c_lr=0.005, e_lr=0.001, gpus=0, epochs=20, dic_epochs=40, cla_epochs=40, g_epochs=10, encoder='GCN', prop='scatter', strlist=None, device=device(type='cpu'))
===========_B1============
Acc     :72.74368
auc_roc :81.29428
F1      :58.96739
parity  :2.95349
equality:6.49679
===========_B2============
Acc     :85.72621
auc_roc :89.16039
F1      :76.21697
parity  :7.91941
equality:9.23718
===========_B3============
Acc     :75.42857
auc_roc :90.16906
F1      :81.16101
parity  :8.10458
equality:5.41369
===========_B4============
Acc     :81.07203
auc_roc :90.28254
F1      :74.25968
parity  :7.98365
equality:8.85481

```

## target domain

### 在target domain (数据集B_1)中进行fine：

```
python data_aug.py --inid='_B1' --seed=27
```
result:

```
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
```





### 在target domain (数据集B_2)中进行fine：
```
python data_aug.py --inid='_B2' --seed=73
```
result:

```
===========当前运行seed 27 ===============
Namespace(dataset='bail', inid='_B2', runs=5, encoder='GCN', prop='scatter', hidden=16, seed=27, gpu=0, dropout=0.5, top_k=10, alpha=1, fairnet_lr=0.001, baisnet_lr=0.001, discri_F_lr=0.01, discri_B_lr=0.01, classify_lr=0.01, encoder_lr=0.01, de_feature_lr=0.001, de_edge_lr=0.001, epochs=15, df_epochs=20, db_epochs=20, class_epochs=40, ad_MLP_F_epochs=20, align_epochs=10, de_train=5, de_traintype_switch=0, de_together_epochs=5, de_separate_epochs=3, de_separate_node_epochs=5, de_separate_edge_epochs=5, strlist=None, device=device(type='cuda', index=0))
===========_B2_B2============
Acc     : ['0.86811', '0.77212', '0.76962', '0.70367', '0.70284']
auc_roc : ['0.89151', '0.86195', '0.85803', '0.82495', '0.69398']
F1      : ['0.73220', '0.38926', '0.37838', '0.02204', '0.01657']
parity  : ['0.07840', '0.02159', '0.02009', '0.00023', '0.00150']
equality: ['0.04234', '0.01500', '0.01384', '0.00070', '0.00116']
===========_B2_B2============
Acc     :76.32721
auc_roc :82.60849
F1      :30.76913
parity  :2.43610
equality:1.46097


===========当前运行seed 73 ===============
Namespace(dataset='bail', inid='_B2', runs=5, encoder='GCN', prop='scatter', hidden=16, seed=73, gpu=0, dropout=0.5, top_k=10, alpha=1, fairnet_lr=0.001, baisnet_lr=0.001, discri_F_lr=0.01, discri_B_lr=0.01, classify_lr=0.01, encoder_lr=0.01, de_feature_lr=0.001, de_edge_lr=0.001, epochs=15, df_epochs=20, db_epochs=20, class_epochs=40, ad_MLP_F_epochs=20, align_epochs=10, de_train=5, de_traintype_switch=0, de_together_epochs=5, de_separate_epochs=3, de_separate_node_epochs=5, de_separate_edge_epochs=5, strlist=None, device=device(type='cuda', index=0))
===========_B2_B2============
Acc     : ['0.85643', '0.73706', '0.71536', '0.70367', '0.75960']
auc_roc : ['0.90720', '0.80228', '0.62812', '0.80722', '0.79705']
F1      : ['0.71617', '0.22222', '0.10026', '0.02740', '0.33641']
parity  : ['0.02731', '0.01068', '0.00886', '0.00299', '0.01399']
equality: ['0.05305', '0.00884', '0.01001', '0.00570', '0.01221']
===========_B2_B2============
Acc     :75.44240
auc_roc :78.83732
F1      :28.04921
parity  :1.27659
equality:1.79616

```

### 在target domain (数据集B_3)中进行fine：
```
python data_aug.py --inid='_B3' --seed=46
```
result:
```
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
```


### 在target domain (数据集B_4)中进行fine：
```
python data_aug.py --inid='_B4' --seed=27
```
result:
```
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
```





