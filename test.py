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

    classifier = torch.load('./model_para/classifier_best_0.pth', weights_only=False).to(args.device)


    encoder = torch.load('./model_para/encoder_best_0.pth', weights_only=False).to(args.device)

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

        "=====test======="
        encoder.eval()
        classifier.eval()
        test_acc = [0 for n in range(len(data2))]
        best_val_tradeoff = [0 for n in range(len(data2))]
        test_auc_roc = [0 for n in range(len(data2))]
        test_f1 = [0 for n in range(len(data2))]
        test_parity = [0 for n in range(len(data2))]
        test_equality = [0 for n in range(len(data2))]

        discriminator = None
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
    parser.add_argument('--runs', type=int, help="运行次数", default=1)  # 5
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
        args.strlist = ['_B1','_B2','_B3','_B4']
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
        print('Acc     :{:.5f}'.format( 100*np.mean(acc.T[i])))
        print('auc_roc :{:.5f}'.format( 100*np.mean(auc_roc.T[i])))
        print('F1      :{:.5f}'.format( 100*np.mean(f1.T[i])))
        print('parity  :{:.5f}'.format( 100*np.mean(parity.T[i])))
        print('equality:{:.5f}'.format( 100*np.mean(equality.T[i])))


    # for i in range(len(args.strlist)):
        # print("==========={}============".format(args.outid+args.strlist[i]))
        # print('Acc: ', np.mean(acc.T[i]))
        # print('auc_roc: ', np.mean(auc_roc.T[i]))
        # print('F1: ', np.mean(f1.T[i]))
        # print('parity: ', np.mean(parity.T[i]))
        # print('equality: ', np.mean(equality.T[i]))



