from dataset import get_dataset
from model import MLP_net, MLP_classifier, MLP_discriminator,GIN_encoder,GCN_encoder_spmm,GCN_encoder_scatter,SAGE_encoder
from utils import seed_everything,np
from learn import evaluate_ged4,evaluate_ged3
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import warnings
import torch.nn as nn
warnings.filterwarnings('ignore')
import math
from pandas import DataFrame
from utils import read_config




def run(data, args, data2):
    #criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    acc, f1, auc_roc, parity, equality = np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs, len(data2)])


    data = data.to(args.device)

    for i in range(len(data2)):
        data2[i] = data2[i].to(args.device)
        data2[i].test_mask = data2[i].test_mask | data2[i].val_mask | data2[i].test_mask


    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = torch.optim.Adam([
        dict(params=discriminator.lin.parameters())], lr=args.d_lr)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = torch.optim.Adam([
        dict(params=classifier.lin.parameters())], lr=args.c_lr)

    if (args.encoder == 'MLP'):
        encoder = MLP_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters())], lr=args.e_lr)
    elif (args.encoder == 'GCN'):
        if args.prop == 'scatter':
            encoder = GCN_encoder_scatter(args).to(args.device)
        else:
            encoder = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters()),
            dict(params=encoder.bias)], lr=args.e_lr)
    elif (args.encoder == 'GIN'):
        encoder = GIN_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv.parameters())], lr=args.e_lr)
    elif (args.encoder == 'SAGE'):
        encoder = SAGE_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv1.parameters()),
            dict(params=encoder.conv2.parameters())], lr=args.e_lr)

    t_idx_s0 = data.sens[data.train_mask] == 0
    t_idx_s1 = data.sens[data.train_mask] == 1
    t_idx_s0_y1 = torch.logical_and(t_idx_s0, data.y[data.train_mask] == 1)
    t_idx_s1_y1 = torch.logical_and(t_idx_s1, data.y[data.train_mask] == 1)
    t_idx_s0_y0 = torch.logical_and(t_idx_s0, data.y[data.train_mask] == 0)
    t_idx_s1_y0 = torch.logical_and(t_idx_s1, data.y[data.train_mask] == 0)
    t_num_s0_y1, t_num_s1_y1, t_num_s0_y0, t_num_s1_y0, = sum(t_idx_s0_y1), sum(t_idx_s1_y1), sum(t_idx_s0_y0), sum(t_idx_s1_y0),

    idx_s0 = data.sens == 0
    idx_s1 = data.sens == 1
    idx_s0_y1 = torch.logical_and(idx_s0, data.y == 1)
    idx_s1_y1 = torch.logical_and(idx_s1, data.y == 1)
    idx_s0_y0 = torch.logical_and(idx_s0, data.y == 0)
    idx_s1_y0 = torch.logical_and(idx_s1, data.y == 0)
    num_s0_y1, num_s1_y1, num_s0_y0, num_s1_y0, = sum(idx_s0_y1), sum(idx_s1_y1), sum(idx_s0_y0), sum(idx_s1_y0),


    eweight = torch.ones(data.edge_index.shape[1]).to(data.x.device)
    adj = torch.sparse_coo_tensor(data.edge_index, eweight, [data.x.shape[0], data.x.shape[0]])
    A2 = torch.spmm(adj, adj)

    for count in range(args.runs):
        print("========={}========".format(count+1))
        discriminator.reset_parameters()
        classifier.reset_parameters()
        encoder.reset_parameters()

        best_val_tradeoff = 0
        best_val_loss = math.inf
        for epoch in range(args.epochs):
            '训练鉴别器与encorder'
            discriminator.train()
            encoder.eval()
            for epoch_d in range(0, args.dic_epochs):
                optimizer_d.zero_grad()
                h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                output = discriminator(h)
                loss_d = criterion(output[data.train_mask].view(-1),
                                   data.x[data.train_mask][:, args.sens_idx])

                loss_d.backward()
                optimizer_d.step()

            'train classifier and encoder'
            classifier.train()
            encoder.train()
            for epoch_c in range(0, args.cla_epochs):
                optimizer_c.zero_grad()
                optimizer_e.zero_grad()
                h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                output = classifier(h)
                loss_c = criterion(output[data.train_mask],data.y[data.train_mask].unsqueeze(1)).to(args.device)
                loss_c.backward()
                optimizer_e.step()
                optimizer_c.step()


            '对抗训练encorder'
            discriminator.eval()
            encoder.train()
            optimizer_e.zero_grad()
            for epoch_g in range(args.g_epochs):
                optimizer_e.zero_grad()
                h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                output = torch.sigmoid(discriminator(h))
                loss_g = F.mse_loss(output[data.train_mask].view(-1),
                                    0.5 * torch.ones_like(output[data.train_mask].view(-1)))

                loss_g.backward()
                optimizer_e.step()

            "=====test======="
            encoder.eval()
            classifier.eval()
            discriminator.train()
            test_acc = [0 for n in range(len(data2))]
            best_val_tradeoff = [0 for n in range(len(data2))]
            test_auc_roc = [0 for n in range(len(data2))]
            test_f1 = [0 for n in range(len(data2))]
            test_parity = [0 for n in range(len(data2))]
            test_equality = [0 for n in range(len(data2))]


            for i in range(len(data2)):
                accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(
                    data2[i].x, classifier, discriminator,encoder, data2[i], args)


                if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (
                        tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff[i]:
                    test_acc[i] = accs['test']
                    test_auc_roc[i] = auc_rocs['test']
                    test_f1[i] = F1s['test']
                    test_parity[i], test_equality[i] = tmp_parity['test'], tmp_equality['test']

                    best_val_tradeoff[i] = auc_rocs['val'] + F1s['val'] + \
                                        accs['val'] - (tmp_parity['val'] + tmp_equality['val'])
                    #torch.save(encoder, './model_para/encoder_best_{}.pth'.format(count))
                    #torch.save(classifier, './model_para/classifier_best_{}.pth'.format(count))

        for i in range(len(args.strlist)):
            acc[count][i] = test_acc[i]
            f1[count][i] = test_f1[i]
            auc_roc[count][i] = test_auc_roc[i]
            parity[count][i] = test_parity[i]
            equality[count][i] = test_equality[i]



    return acc, f1, auc_roc, parity, equality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="数据集种类", default='bail')
    parser.add_argument('--inid', type=str, help="作为source data的训练数据", default='_B0')
    parser.add_argument('--outid', type=str, help="作为test的数据", default='all')
    parser.add_argument('--dropout', type=float, help="编码器encoder的dropout概率",default=0.5)
    parser.add_argument('--top_k', type=int,help="利用superman算法算得与敏感属性前K个相似的特征",default=10)
    parser.add_argument('--alpha', type=float, help="计算auc_rocs+F1+acc-args.alpha*(tmp_parity+tmp_equality)",default=1)
    parser.add_argument('--runs', type=int, help="运行次数", default=5)  # 5
    parser.add_argument('--hidden', type=int, help="编码器encoder输出特征的维度 ", default=16)

    parser.add_argument('--d_lr', type=float, help="分辨器的学习率", default=0.01)
    parser.add_argument('--c_lr', type=float, help="分类器的学习率", default=0.005)
    parser.add_argument('--e_lr', type=float, help="编码器的学习率", default=0.001)
    parser.add_argument('--gpus', type=int, help="gpu卡号", default=0)

    parser.add_argument('--epochs', type=int, help="训练轮数", default=20)
    parser.add_argument('--dic_epochs', type=int, help="鉴别器的训练轮数",default=40)
    parser.add_argument('--cla_epochs', type=int, help="分类器的训练轮数",default=40)
    parser.add_argument('--g_epochs', type=int, help="对抗网络的训练轮数",default=10)
    parser.add_argument('--encoder', type=str, help="编码器encoder的种类",default='GCN')
    parser.add_argument('--prop', type=str, default='scatter')


    args = parser.parse_args()
    args.strlist = None
    if args.outid == "all":
        args.outid = ""
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(args)
    data, args.sens_idx, args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_dataset(
        args.dataset, args.inid, args.top_k)

    data2 = []
    if args.dataset == "credit":

        args.strlist = ['_C1', '_C2', '_C3', '_C4']
        for i in range(len(args.strlist)):
            datatmp, _, _, _, _, _ = get_dataset(
                args.dataset,  args.strlist[i], args.top_k)
            data2.append(datatmp)
    elif args.dataset == "bail":
        args.strlist = ['_B0']
        for i in range(len(args.strlist)):
            datatmp, _, _, _, _, _ = get_dataset(
                args.dataset,  args.strlist[i], args.top_k)
            data2.append(datatmp)
    elif args.dataset == "pokec":
        args.strlist = ['_z', '_n',]
        args.inidIndex = args.strlist.index(args.inid)
        for i in range(len(args.strlist)):
            if args.inidIndex == i:
                data2.append(data)
                continue
            datatmp, _, _, _, _, _ = get_dataset(
                args.dataset,  args.strlist[i], args.top_k)
            data2.append(datatmp)

    args.num_features, args.num_classes = data.x.shape[1], len(data.y.unique()) - 1
    if args.dataset == "pokec":
        args.num_classes = 1
    args.train_ratio, args.val_ratio = torch.tensor(
        [(data.y[data.train_mask] == 0).sum(), (data.y[data.train_mask] == 1).sum()]), \
                                       torch.tensor(
                                           [(data.y[data.val_mask] == 0).sum(), (data.y[data.val_mask] == 1).sum()])
    args.train_ratio, args.val_ratio = torch.max(args.train_ratio) / args.train_ratio, \
                                       torch.max(args.val_ratio) / args.val_ratio
    args.train_ratio, args.val_ratio = args.train_ratio[data.y[data.train_mask].long()], \
                                       args.val_ratio[data.y[data.val_mask].long()]





    acc, f1, auc_roc, parity, equality = run(data, args, data2) # data is training data.data2 is testing data


    for i in range(len(args.strlist)):
        print("==========={}============".format(args.outid+args.strlist[i]))
        print('Acc: ', acc.T[i])
        print('auc_roc: ', auc_roc.T[i])
        print('parity: ', parity.T[i])
        print('equality: ', equality.T[i])
        #print('F1: ', np.mean(f1.T[i]))
    for i in range(len(args.strlist)):
        print("==========={}============".format(args.outid+args.strlist[i]))
        print('Acc: ', np.mean(acc.T[i]))
        print('auc_roc: ', np.mean(auc_roc.T[i]))
        print('parity: ', np.mean(parity.T[i]))
        print('equality: ', np.mean(equality.T[i]))
        #print('F1: ', np.mean(f1.T[i]))


'''
(pytorch) zhangxin@zhangxindeMacBook-Pro FatraGNN-main % python train.py
Namespace(dataset='bail', inid='_B0', outid='', dropout=0.5, top_k=10, alpha=1, runs=5, hidden=16, d_lr=0.01, c_lr=0.005, e_lr=0.001, gpus=0, epochs=20, dic_epochs=40, cla_epochs=40, g_epochs=10, encoder='GCN', prop='scatter', strlist=None, device=device(type='cpu'))
=========1========
=========2========
=========3========
=========4========
=========5========
===========_B1============
Acc:  [0.72743682 0.72202166 0.72021661 0.71389892 0.71750903]
auc_roc:  [0.81294277 0.81106886 0.8170788  0.81590577 0.81610303]
parity:  [0.02953486 0.00709896 0.01034382 0.01059547 0.00460903]
equality:  [0.06496785 0.05643433 0.051543   0.0549025  0.05319317]
===========_B2============
Acc:  [0.8572621  0.85642738 0.85475793 0.85392321 0.85893155]
auc_roc:  [0.89160395 0.8927643  0.89367731 0.89322413 0.89428156]
parity:  [0.07919411 0.08068981 0.06931019 0.07104029 0.08300592]
equality:  [0.09237175 0.09353527 0.07922368 0.08108588 0.08701899]
===========_B3============
Acc:  [0.75428571 0.75928571 0.78928571 0.78285714 0.775     ]
auc_roc:  [0.90169065 0.90691031 0.91086395 0.90938777 0.91273878]
parity:  [0.08104575 0.07947095 0.08739643 0.0842777  0.08410272]
equality:  [0.0541369  0.04948238 0.05580178 0.05360363 0.05093101]
===========_B4============
Acc:  [0.81072027 0.81909548 0.83082077 0.83082077 0.82579564]
auc_roc:  [0.90282542 0.90489516 0.90335414 0.90246434 0.90600418]
parity:  [0.07983652 0.04642426 0.05977778 0.06058605 0.05241229]
equality:  [0.08854811 0.06407258 0.07287568 0.06644481 0.06379151]
===========_B1============
Acc:  0.720216606498195
auc_roc:  0.8146198462261618
parity:  0.012436427209154489
equality:  0.056208170617858585
===========_B2============
Acc:  0.8562604340567613
auc_roc:  0.8931102486379527
parity:  0.076648063399933
equality:  0.08664711500266227
===========_B3============
Acc:  0.7721428571428571
auc_roc:  0.9083182934467299
parity:  0.0832587103082703
equality:  0.05279113856282973
===========_B4============
Acc:  0.8234505862646566
auc_roc:  0.9039086477703556
parity:  0.05980738143484893
equality:  0.0711465383491107



'''