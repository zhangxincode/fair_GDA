import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torch.sparse


from dataset import *
from model import *
from utils import *
from learn import *
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
import math
from pandas import DataFrame
from utils import read_config
from torch.utils.tensorboard import SummaryWriter       # 引入包
import copy
writer = SummaryWriter('./runs/1')
import ipdb
from data_edit import DE_X,DE_A,train_data_edit,train_data_aug,DataAug


def eval_tool(pre,y):
    acc = ((pre == y).sum() / y.shape[0])  # 伪标签准确率
    acc_1 = ((pre == 1).int() * (y == 1).int()).sum() / (y == 1).int().sum()  # 伪标签为1的准确率
    acc_0 = ((pre == 0).int() * (y == 0).int()).sum() / (y == 0).int().sum()  # 伪标签为0的准确率
    TP = ((pre == 1).int() * (y == 1).int()).sum()
    TN = ((pre == 0).int() * (y == 0).int()).sum()
    FP = ((pre == 1).int() * (y == 0).int()).sum()
    FN = ((pre == 0).int() * (y == 1).int()).sum()
    F1 = 2*TP/(2*TP+FP+FN) # 针对数据不平衡问题
    parity, equality = fair_metric(pre.cpu().numpy(), y.cpu().numpy(), data.sens.cpu().numpy())
    result = {
        'acc': acc,
        'acc_1': acc_1,
        'acc_0': acc_0,
        'F1': F1,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'parity': parity,
        'equality': equality
    }
    return result


import torch
from torch_sparse import SparseTensor

@torch.no_grad()
def update_labels_by_neighbors_with_predictions(
    data, encoder, classifier,
    alpha: float = 0.25,   # 传播强度（越小越稳，0.05~0.2 常用）
    K: int = 3,           # 迭代步数（2~10，过大易过平滑）
    add_self_loop: bool = True
):
    """
    1) 用 encoder+classifier 生成初始预测 P0（概率分布）；
    2) 用对称归一化稀疏邻接 A_sym = D^{-1/2}(A+I)D^{-1/2} 做 K 次传播：
           P <- (1 - alpha) * P0 + alpha * (A_sym @ P)
       每步保持行为概率分布并数值安全；
    3) 输出 new_y = argmax(P)。
    """
    eps = 1e-12

    # ---------- 1) 初始预测 ----------
    encoder.eval(); classifier.eval()
    h = encoder(data.x, data.edge_index, getattr(data, "adj_norm_sp", None),data.edge_weight)
    logits = classifier(h)

    # 支持二分类/多分类：
    if logits.dim() == 1 or logits.size(-1) == 1:
        # 二分类：sigmoid -> (1-p, p)
        p = torch.sigmoid(logits.view(-1)).clamp(0.0 + 1e-6, 1.0 - 1e-6)
        P0 = torch.stack([1 - p, p], dim=1)  # (N, 2)
    else:
        # 多分类：softmax
        P0 = torch.softmax(logits, dim=1)

    P = P0.clone()

    # ---------- 2) 构建对称归一化稀疏邻接 A_sym ----------
    num_nodes = data.x.size(0)
    row, col = data.edge_index  # shape: (2, E)

    A = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    if add_self_loop:
        A = A.set_diag()

    deg = A.sum(dim=1).to(torch.float)          # 度 D (N,)
    deg = deg.clamp_min(eps)
    d_is = deg.pow(-0.5)                        # D^{-1/2}
    A_sym = A.mul(d_is.view(-1, 1)).mul(d_is.view(1, -1))  # D^{-1/2} A D^{-1/2}

    # ---------- 3) 稳定传播 ----------
    for _ in range(K):
        neighbor = A_sym.matmul(P)                       # 稀疏×稠密
        P = (1 - alpha) * P0 + alpha * neighbor
        P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)  # 保持为概率分布

    # ---------- 4) 输出 ----------
    new_labels = P.argmax(dim=1)
    data.new_y = new_labels
    data.new_probs = P      # 若后续要用概率，可顺带存下
    # import ipdb; ipdb.set_trace()
    eval_tool(torch.argmax(P0, dim=1), data.y)  # 直接预测的伪标签准确率
    eval_tool(data.new_y, data.y)  # 传播后的伪标签准确率
    # import ipdb; ipdb.set_trace()
    return data




def run(data, args, data2):
    criterion = nn.BCELoss()

    seed_everything(args.seed)
    if args.ood == 2:
        acc, f1, auc_roc, parity, equality = np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs, len(data2)])
    elif args.ood == 1:
        acc, f1, auc_roc, parity, equality = np.zeros([args.runs,len(args.strlist)]), np.zeros([args.runs,len(args.strlist)]), np.zeros([args.runs,len(args.strlist)]), np.zeros([args.runs,len(args.strlist)]), np.zeros([args.runs, len(args.strlist)])

    else:
        acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    data = data.to(args.device)

    if args.ood == 2:
        for i in range(len(data2)):
            data2[i] = data2[i].to(args.device)
            data2[i].test_mask = data2[i].test_mask | data2[i].val_mask | data2[i].test_mask
    elif data2 != None:
        data2 = data2.to(args.device)
    else:
        data2 = data2


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

    MLP_F = MLP_encoder(args).to(args.device)
    optimizer_F = torch.optim.Adam(params=MLP_F.parameters(), lr=0.01)

    MLP_B = MLP_encoder(args).to(args.device)
    optimizer_B = torch.optim.Adam(params=MLP_B.parameters(), lr=0.01)

    discriminator_F = MLP_discriminator(args).to(args.device)
    optimizer_D_F = torch.optim.Adam(params=discriminator_F.parameters(), lr=0.01)

    discriminator_B = MLP_discriminator(args).to(args.device)
    optimizer_D_B = torch.optim.Adam(params=discriminator_B.parameters(), lr=0.01)

    classifier = torch.load('./model_para/classifier_best_0.pth', weights_only=False).to(args.device)
    optimizer_C = torch.optim.Adam(params=classifier.parameters(), lr=0.01)

    encoder = torch.load('./model_para/encoder_best_0.pth', weights_only=False).to(args.device)
    optimizer_E = torch.optim.Adam(params=encoder.parameters(), lr=0.01)



    data3 = data.clone()
    data_aug = DataAug(classifier, encoder, data3.x.shape[0], data3.x.shape[1], data3.edge_index.shape[1])
    de_a = DE_A(encoder, data3.x.shape[0], data3.x.shape[1], data3.edge_index.shape[1])
    de_x = DE_X(encoder, data3.x.shape[0], data3.x.shape[1], data3.edge_index.shape[1])


    '==========train============='
    for count in range(args.runs):
        data1 = update_labels_by_neighbors_with_predictions(data3, encoder, classifier)
        pbar = tqdm(range(20), unit='epoch')# 20
        for epoch in pbar:
            ''' 训练判别器F'''
            encoder.eval()
            MLP_F.train()
            discriminator_F.train()
            for i in range(10):
                optimizer_F.zero_grad()
                optimizer_D_F.zero_grad()
                with torch.no_grad():
                    h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_F = MLP_F(h)
                pred_B = torch.sigmoid(discriminator_F(h_F))

                # 判别器的损失是两个分支损失之和
                # import ipdb; ipdb.set_trace()
                loss_D = criterion(pred_B.view(-1), data1.x[:, args.sens_idx])
                # loss_D = nn.MSELoss(pred_B.view(-1), data1.x[:, args.sens_idx])
                #writer.add_scalar('lossD', loss_D, global_step=epoch*10+i)
                loss_D.backward()
                optimizer_D_F.step()
                optimizer_F.step()


            '''   训练分类器'''
            encoder.eval()
            MLP_F.train()
            classifier.train()


            for i in range(50):
                optimizer_F.zero_grad()
                optimizer_C.zero_grad()
                with torch.no_grad():
                    h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_F = MLP_F(h)
                output_class = torch.sigmoid(classifier(h_F))
                # import ipdb; ipdb.set_trace()
                loss_c = criterion(output_class, data1.new_probs[:,1].unsqueeze(1).float())
                #writer.add_scalar('lossc', loss_c, global_step=epoch * 50 + i)
                loss_c.backward()
                optimizer_C.step()
                optimizer_F.step()



            ''' 对抗训练MLP_F '''
            discriminator_F.eval()
            classifier.eval()
            encoder.eval()
            MLP_F.train()


            for i in range(20):
                optimizer_F.zero_grad()
                with torch.no_grad():
                    h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_F = MLP_F(h)
                pred_F_adv = discriminator_F(h_F)
                # import ipdb; ipdb.set_trace()
                loss_adv_F = criterion(pred_F_adv.view(-1),0.5 * torch.ones_like(pred_F_adv.view(-1)))
                #writer.add_scalar('lossadv', loss_adv_F, global_step=epoch * 30 + i)
                loss_adv_F.backward()
                optimizer_F.step()


            '''训练判别器B'''
            encoder.eval()
            MLP_B.train()
            discriminator_B.train()
            for i in range(50):
                optimizer_B.zero_grad()
                optimizer_D_B.zero_grad()
                with torch.no_grad():
                    h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_B = MLP_B(h)
                pred_B_adv = torch.sigmoid(discriminator_B(h_B))
                loss_adv_B = criterion(pred_B_adv.view(-1), data1.x[:, args.sens_idx])
                #writer.add_scalar('lossb', loss_adv_B, global_step=epoch * 50 + i)
                loss_adv_B.backward()
                optimizer_B.step()
                optimizer_D_B.step()


            '''  异类疏远 '''
            encoder.train()
            MLP_F.eval()
            MLP_B.eval()
            classifier.train()


            for i in range(10):
                optimizer_C.zero_grad()
                optimizer_F.zero_grad()
                optimizer_B.zero_grad()
                optimizer_E.zero_grad()
                h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_F = MLP_F(h)
                h_B = MLP_B(h)
                logits = torch.sigmoid(classifier(h_F))
                task_loss = criterion(logits, data1.new_probs[:,1].unsqueeze(1).float())

                # 分离损失
                hF = F.normalize(h_F, dim=1)
                hB = F.normalize(h_B, dim=1)
                cos_sim = (hF * hB).sum(dim=1)
                loss_ortho = (cos_sim ** 2).mean()
                #writer.add_scalar('losssmi', loss_ortho, global_step=epoch * 10 + i)
                #writer.add_scalar('losssmi_C', task_loss, global_step=epoch * 10 + i)
                loss = 0.9*task_loss + 0.1 * loss_ortho
                #writer.add_scalar('lossall', loss, global_step=epoch * 50 + i)


                loss.backward()
                optimizer_C.step()
                optimizer_F.step()
                optimizer_B.step()
                optimizer_E.step()






            "=====test======="
            encoder.eval()
            discriminator_F.eval()
            classifier.eval()
            MLP_F.eval()
            MLP_B.eval()

            # 评价指标初始化
            if args.ood == 1:
                test_acc = [0 for n in range(len(args.strlist))]
                best_val_tradeoff = [0 for n in range(len(args.strlist))]
                test_auc_roc = [0 for n in range(len(args.strlist))]
                test_f1 = [0 for n in range(len(args.strlist))]
                test_parity = [0 for n in range(len(args.strlist))]
                test_equality = [0 for n in range(len(args.strlist))]
            elif args.ood == 2:
                test_acc = [0 for n in range(len(data2))]
                best_val_tradeoff = [0 for n in range(len(data2))]
                test_auc_roc = [0 for n in range(len(data2))]
                test_f1 = [0 for n in range(len(data2))]
                test_parity = [0 for n in range(len(data2))]
                test_equality = [0 for n in range(len(data2))]


            if args.ood == 2:
                for i in range(len(data2)):
                    accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged4(
                        data2[i].x, classifier,MLP_F,encoder, data2[i], args)
                    pbar.set_postfix({'acc': accs['test'], 'auc': auc_rocs['test'], 'f1': F1s['test'],'parity':tmp_parity['test'],'equality':tmp_equality['test']})


                    if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (
                            tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff[i]:
                        test_acc[i] = accs['test']
                        test_auc_roc[i] = auc_rocs['test']
                        test_f1[i] = F1s['test']
                        test_parity[i], test_equality[i] = tmp_parity['test'], tmp_equality['test']

                        best_val_tradeoff[i] = auc_rocs['val'] + F1s['val'] + \
                                            accs['val'] - (tmp_parity['val'] + tmp_equality['val'])
            elif args.ood == 1:
                for i in range(len(args.strlist)):
                    datatmp, _, _, _, _, _ = get_dataset(args.dataset, args.outid + args.strlist[i], args.top_k)
                    datatmp = datatmp.to(args.device)
                    datatmp.test_mask = datatmp.test_mask | datatmp.val_mask | datatmp.test_mask
                    accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged4(
                        datatmp.x, classifier, encoder, datatmp, args)



                    test_acc[i] = accs['test']
                    test_auc_roc[i] = auc_rocs['test']
                    test_f1[i] = F1s['test']
                    test_parity[i], test_equality[i] = tmp_parity['test'], tmp_equality['test']



            else:
                accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged4(
                    data.x, classifier, encoder, data, args)
                if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (
                        tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                    test_acc = accs['test']
                    test_auc_roc = auc_rocs['test']
                    test_f1 = F1s['test']
                    test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

                    best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
                                        accs['val'] - (tmp_parity['val'] + tmp_equality['val'])

            # 数据编辑训练
            data3 = train_data_aug(data1,data_aug,args) # xa合并训练
            # data3 = train_data_edit(data1,de_a,de_x,args)  # xa单独训练


        for i in range(len(args.strlist)):
            acc[count][i] = test_acc[i]
            f1[count][i] = test_f1[i]
            auc_roc[count][i] = test_auc_roc[i]
            parity[count][i] = test_parity[i]
            equality[count][i] = test_equality[i]



    return acc, f1, auc_roc, parity, equality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bail')#german
    parser.add_argument('--inid', type=str, default='_B4')
    parser.add_argument('--outid', type=str, default='all')
    parser.add_argument('--runs', type=int, default=1) # 5
    parser.add_argument('--start', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dic_epochs', type=int, default=2)
    parser.add_argument('--dtb_epochs', type=int, default=5)
    parser.add_argument('--cla_epochs', type=int, default=10)
    parser.add_argument('--clo_epochs', type=int, default=2)
    parser.add_argument('--a_epochs', type=int, default=5)
    parser.add_argument('--g_epochs', type=int, default=5)


    parser.add_argument('--g_lr', type=float, default=0.001)
    parser.add_argument('--g_wd', type=float, default=0)
    parser.add_argument('--d_lr', type=float, default=0.001)
    parser.add_argument('--d_wd', type=float, default=0)
    parser.add_argument('--c_lr', type=float, default=0.005)
    parser.add_argument('--c_wd', type=float, default=0)
    parser.add_argument('--e_lr', type=float, default=0.005)
    parser.add_argument('--e_wd', type=float, default=0)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--prop', type=str, default='scatter')
    parser.add_argument('--predictfile', type=str, default='tmp')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)#1
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--clip_e', type=float, default=1)
    parser.add_argument('--clip_c', type=float, default=1)
    parser.add_argument('--f_mask', type=str, default='no')
    parser.add_argument('--weight_clip', type=str, default='yes')
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--ood', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--close', type=int, default=1)
    parser.add_argument('--discri', type=int, default=1)
    parser.add_argument('--dropf', type=int, default=0)
    parser.add_argument('--dropf_rate', type=float, default=0.1)
    parser.add_argument('--disturb', type=int, default=1)
    parser.add_argument('--align', type=int, default=1)
    parser.add_argument('--modiStru', type=int, default=0)
    parser.add_argument('--drope_rate', type=float, default=0.5)
    parser.add_argument('--tune', type=str, default='True', help='if tune')
    parser.add_argument('--times', type=str, default='config')
    parser.add_argument('--labda', type=float, default=0.5)



    args = parser.parse_args()
    # seed_everything(args.seed)
    args.strlist = None
    if args.tune == 'True':
        args = read_config(args)
    if args.outid == "all":
        args.outid = ""
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(args)
    data, args.sens_idx, args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_dataset(
        args.dataset, args.inid, args.top_k)

    if args.ood == 1:
        data2 = []

        if args.dataset == "bail":
            if args.outid == "_md0":
                args.strlist = ['_0.56_0.35_0.54_0.25_0.06_0.56', '_0.51_0.32_0.49_0.24_0.00_0.56',
                           '_0.58_0.36_0.56_0.26_0.10_0.56',
                           '_0.49_0.30_0.47_0.25_0.00_0.56', '_0.63_0.39_0.61_0.30_0.20_0.56',
                           '_0.45_0.27_0.43_0.29_0.00_0.56',
                           '_0.67_0.42_0.65_0.36_0.29_0.56', '_0.41_0.25_0.40_0.33_0.00_0.56',
                           '_0.72_0.45_0.70_0.43_0.39_0.56',
                           '_0.37_0.23_0.37_0.37_0.00_0.44', '_0.76_0.48_0.74_0.51_0.48_0.56',
                           '_0.34_0.21_0.34_0.41_0.00_0.44',
                           '_0.81_0.51_0.79_0.59_0.58_0.56', '_0.31_0.19_0.32_0.44_0.00_0.44',
                           '_0.86_0.54_0.84_0.68_0.68_0.56',
                           '_0.29_0.17_0.30_0.47_0.00_0.44', '_0.90_0.57_0.89_0.78_0.79_0.56',
                           '_0.27_0.16_0.28_0.50_0.00_0.44',
                           '_0.95_0.60_0.94_0.89_0.89_0.56', '_0.25_0.15_0.27_0.52_0.00_0.44',
                           '_0.64_0.40_0.62_0.32_0.22_0.56',
                           '_0.43_0.26_0.42_0.30_0.00_0.56', '_0.69_0.43_0.67_0.38_0.32_0.56',
                           '_0.40_0.24_0.39_0.34_0.00_0.44',
                           '_0.73_0.46_0.71_0.45_0.42_0.56', '_0.36_0.22_0.36_0.38_0.00_0.44',
                           '_0.78_0.49_0.76_0.53_0.51_0.56',
                           '_0.33_0.20_0.34_0.42_0.00_0.44', '_0.82_0.52_0.80_0.62_0.61_0.56',
                           '_0.30_0.18_0.31_0.45_0.00_0.44',
                           '_0.87_0.55_0.86_0.71_0.71_0.56', '_0.28_0.17_0.30_0.48_0.00_0.44',
                           '_0.92_0.58_0.91_0.81_0.82_0.56',
                           '_0.26_0.16_0.28_0.51_0.00_0.44', '_0.97_0.61_0.96_0.92_0.92_0.56',
                           '_0.24_0.14_0.27_0.53_0.00_0.44',
                           '_0.61_0.38_0.59_0.28_0.15_0.56', '_0.47_0.29_0.45_0.27_0.00_0.56',
                           '_0.65_0.41_0.63_0.33_0.24_0.56',
                           '_0.43_0.26_0.41_0.31_0.00_0.56', '_0.70_0.43_0.68_0.39_0.34_0.56',
                           '_0.39_0.24_0.38_0.35_0.00_0.44',
                           '_0.74_0.46_0.72_0.47_0.43_0.56', '_0.36_0.22_0.35_0.39_0.00_0.44',
                           '_0.79_0.49_0.77_0.55_0.53_0.56',
                           '_0.33_0.20_0.33_0.43_0.00_0.44', '_0.83_0.52_0.81_0.64_0.63_0.56',
                           '_0.30_0.18_0.31_0.46_0.00_0.44',
                           '_0.88_0.56_0.86_0.73_0.73_0.56', '_0.28_0.17_0.29_0.48_0.00_0.44',
                           '_0.93_0.59_0.92_0.84_0.84_0.56',
                           '_0.26_0.15_0.28_0.51_0.00_0.44', '_0.98_0.62_0.97_0.94_0.94_0.56',
                           '_0.24_0.14_0.26_0.53_0.00_0.44']
            elif args.outid == "_md3":
                args.strlist = ['_0.60_0.30_0.60_0.25_0.18_0.48', '_0.46_0.23_0.46_0.23_0.00_0.48', '_0.65_0.32_0.64_0.32_0.27_0.48',
                 '_0.42_0.21_0.42_0.28_0.00_0.48', '_0.69_0.35_0.69_0.40_0.37_0.48', '_0.38_0.19_0.39_0.32_0.00_0.48',
                 '_0.74_0.37_0.74_0.48_0.46_0.48', '_0.35_0.17_0.36_0.36_0.00_0.48', '_0.79_0.39_0.78_0.57_0.56_0.48',
                 '_0.32_0.16_0.34_0.40_0.00_0.48', '_0.83_0.42_0.83_0.66_0.66_0.48', '_0.30_0.15_0.32_0.44_0.00_0.48',
                 '_0.88_0.44_0.88_0.76_0.75_0.48', '_0.27_0.13_0.30_0.47_0.00_0.48', '_0.93_0.47_0.93_0.85_0.85_0.48',
                 '_0.25_0.12_0.28_0.49_0.00_0.48', '_0.98_0.49_0.98_0.95_0.95_0.48', '_0.24_0.12_0.27_0.52_0.00_0.48',
                 '_0.56_0.28_0.55_0.21_0.09_0.48', '_0.51_0.25_0.50_0.20_0.02_0.48', '_0.58_0.29_0.57_0.23_0.13_0.48',
                 '_0.49_0.24_0.48_0.21_0.00_0.48', '_0.62_0.31_0.62_0.28_0.23_0.48', '_0.44_0.22_0.44_0.26_0.00_0.48',
                 '_0.67_0.34_0.67_0.36_0.32_0.48', '_0.40_0.20_0.40_0.30_0.00_0.48', '_0.72_0.36_0.72_0.44_0.41_0.48',
                 '_0.37_0.18_0.37_0.34_0.00_0.48', '_0.76_0.38_0.76_0.53_0.51_0.48', '_0.34_0.17_0.35_0.38_0.00_0.48',
                 '_0.81_0.41_0.81_0.62_0.61_0.48', '_0.31_0.15_0.33_0.42_0.00_0.48', '_0.86_0.43_0.85_0.71_0.70_0.48',
                 '_0.29_0.14_0.31_0.45_0.00_0.48', '_0.90_0.45_0.90_0.80_0.80_0.48', '_0.26_0.13_0.29_0.48_0.00_0.48',
                 '_0.95_0.48_0.95_0.90_0.90_0.48', '_0.24_0.12_0.27_0.51_0.00_0.48', '_0.64_0.32_0.64_0.31_0.25_0.48',
                 '_0.43_0.21_0.43_0.27_0.00_0.48', '_0.68_0.34_0.68_0.38_0.35_0.48', '_0.39_0.19_0.39_0.32_0.00_0.48',
                 '_0.73_0.37_0.73_0.46_0.44_0.48', '_0.36_0.18_0.37_0.36_0.00_0.48', '_0.78_0.39_0.78_0.55_0.54_0.48',
                 '_0.33_0.16_0.34_0.40_0.00_0.48', '_0.82_0.41_0.82_0.64_0.64_0.48', '_0.30_0.15_0.32_0.43_0.00_0.48',
                 '_0.87_0.44_0.87_0.74_0.73_0.48', '_0.28_0.14_0.30_0.46_0.00_0.48', '_0.92_0.46_0.92_0.83_0.83_0.48',
                 '_0.26_0.13_0.28_0.49_0.00_0.48', '_0.97_0.49_0.97_0.93_0.93_0.48', '_0.24_0.12_0.27_0.51_0.00_0.48']


        data2 = None
        args.in_hom = [0 for i in range(len(args.strlist))]
        args.edge_hom = [0 for i in range(len(args.strlist))]
        args.node_hom = [0 for i in range(len(args.strlist))]
        args.class_hom = [0 for i in range(len(args.strlist))]
        args.agg_hom = [0 for i in range(len(args.strlist))]

    elif args.ood == 2:
        # 设置 test set类别
        data2 = []
        if args.dataset == "credit":

            args.strlist = ['_C2', '_C3', '_C4']
            for i in range(len(args.strlist)):
                datatmp, _, _, _, _, _ = get_dataset(
                    args.dataset,  args.strlist[i], args.top_k)
                data2.append(datatmp)
        elif args.dataset == "bail":
            args.strlist = ['_B1',]
            for i in range(len(args.strlist)):
                datatmp, _, _, _, _, _ = get_dataset(
                    args.dataset,  args.strlist[i], args.top_k)
                data2.append(datatmp)
        elif args.dataset == "pokec":
            args.strlist = ['_n',]
            args.inidIndex = args.strlist.index(args.inid)
            for i in range(len(args.strlist)):
                if args.inidIndex == i:
                    data2.append(data)
                    continue
                datatmp, _, _, _, _, _ = get_dataset(
                    args.dataset,  args.strlist[i], args.top_k)
                data2.append(datatmp)


    else:
        data2 = None
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
        print('Acc: ', np.mean(acc.T[i]))
        print('auc_roc: ', np.mean(auc_roc.T[i]))
        print('parity: ', np.mean(parity.T[i]))
        print('equality: ', np.mean(equality.T[i]))
        print('f1: ', np.mean(f1.T[i]))



