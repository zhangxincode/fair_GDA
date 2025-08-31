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
    acc, f1, auc_roc, parity, equality = np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs, len(data2)])
    data = data.to(args.device)
    for i in range(len(data2)):
        data2[i] = data2[i].to(args.device)
        data2[i].test_mask = data2[i].test_mask | data2[i].val_mask | data2[i].test_mask



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
    de_a = DE_A(encoder, data3.x.shape[0], data3.x.shape[1], data3.edge_index.shape[1],args)
    de_x = DE_X(encoder, data3.x.shape[0], data3.x.shape[1], data3.edge_index.shape[1],args)
    data1 = update_labels_by_neighbors_with_predictions(data3, encoder, classifier)
    '==========train============='
    for count in range(args.runs):
        #pbar = tqdm(range(20), unit='epoch')# 20
        for epoch in range(args.epochs):
            ''' 训练判别器F'''
            encoder.eval()
            MLP_F.train()
            discriminator_F.train()
            for i in range(args.df_epochs):
                optimizer_F.zero_grad()
                optimizer_D_F.zero_grad()
                with torch.no_grad():
                    h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_F = MLP_F(h)
                pred_B = torch.sigmoid(discriminator_F(h_F))

                # 判别器的损失是两个分支损失之和
                # import ipdb; ipdb.set_trace()
                loss_D = criterion(pred_B.view(-1), data1.x[:,data1.sen_idx])
                loss_D.backward()
                optimizer_D_F.step()
                optimizer_F.step()


            '''   训练分类器'''
            encoder.eval()
            MLP_F.train()
            classifier.train()
            for i in range(args.class_epochs):
                optimizer_F.zero_grad()
                optimizer_C.zero_grad()
                with torch.no_grad():
                    h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_F = MLP_F(h)
                output_class = torch.sigmoid(classifier(h_F))
                loss_c = criterion(output_class, data1.new_probs[:,1].unsqueeze(1).float())
                loss_c.backward()
                optimizer_C.step()
                optimizer_F.step()





            ''' 对抗训练MLP_F '''
            discriminator_F.eval()
            classifier.eval()
            encoder.eval()
            MLP_F.train()
            for i in range(args.ad_MLP_F_epochs):
                optimizer_F.zero_grad()
                with torch.no_grad():
                    h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_F = MLP_F(h)
                pred_F_adv = discriminator_F(h_F)
                # import ipdb; ipdb.set_trace()
                loss_adv_F = criterion(pred_F_adv.view(-1),0.5 * torch.ones_like(pred_F_adv.view(-1)))

                loss_adv_F.backward()
                optimizer_F.step()


            '''训练判别器B'''
            encoder.eval()
            MLP_B.train()
            discriminator_B.train()
            for i in range(args.db_epochs):
                optimizer_B.zero_grad()
                optimizer_D_B.zero_grad()
                with torch.no_grad():
                    h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp,data1.edge_weight)
                h_B = MLP_B(h)
                pred_B_adv = torch.sigmoid(discriminator_B(h_B))
                loss_adv_B = criterion(pred_B_adv.view(-1), data1.x[:, data1.sen_idx])
                loss_adv_B.backward()
                optimizer_B.step()
                optimizer_D_B.step()


            '''  异类疏远 '''
            encoder.train()
            MLP_F.eval()
            MLP_B.eval()
            classifier.train()
            for i in range(args.align_epochs):
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
                loss = 0.9*task_loss + 0.1 * loss_ortho
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
            test_acc = [0 for n in range(len(data2))]
            best_val_tradeoff = [0 for n in range(len(data2))]
            test_auc_roc = [0 for n in range(len(data2))]
            test_f1 = [0 for n in range(len(data2))]
            test_parity = [0 for n in range(len(data2))]
            test_equality = [0 for n in range(len(data2))]



            for i in range(len(data2)):
                accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged4(
                    data2[i].x, classifier,MLP_F,encoder, data2[i], args)


                if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (
                        tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff[i]:
                    test_acc[i] = accs['test']
                    test_auc_roc[i] = auc_rocs['test']
                    test_f1[i] = F1s['test']
                    test_parity[i], test_equality[i] = tmp_parity['test'], tmp_equality['test']

                    best_val_tradeoff[i] = auc_rocs['val'] + F1s['val'] + \
                                        accs['val'] - (tmp_parity['val'] + tmp_equality['val'])

            # 数据编辑训练
            if (epoch+1)%args.de_train == 0:
                if args.de_traintype_switch==0:
                    data1 = train_data_edit(data1,de_a,de_x,args)  # xa单独训练
                elif args.de_traintype_switch==1:
                    data1 = train_data_aug(data1,data_aug,args) # xa合并训练


        for i in range(len(args.strlist)):
            acc[count][i] = test_acc[i]
            f1[count][i] = test_f1[i]
            auc_roc[count][i] = test_auc_roc[i]
            parity[count][i] = test_parity[i]
            equality[count][i] = test_equality[i]



    return acc, f1, auc_roc, parity, equality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,help="数据集种类",default='bail')#german
    parser.add_argument('--inid', type=str, help="作为输入的数据集",default='_B2')
    parser.add_argument('--runs', type=int, help="运行次数",default=1) # 5
    parser.add_argument('--prop', type=str, default='scatter')
    parser.add_argument('--encoder', type=str,help="编码器encoder种类", default='GCN')
    parser.add_argument('--hidden', type=int,help="编码器encoder输出特征的维度 ", default=16)
    parser.add_argument('--de_train', type=int, help="数据编辑每几个epoch进行一次训练", default=5)
    parser.add_argument('--de_traintype_switch', type=int, help="是采用结点特征和边共同训练(1)，还是分离训练开关(0)。", default=0)
    parser.add_argument('--seed', type=int,help="初始化种子",default=1)
    parser.add_argument('--gpu', type=int, help="使用的gpu编号,若没有自动变为cpu",default=0)

    parser.add_argument('--dropout', type=float, help="编码器encoder的dropout概率",default=0.5)

    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--alpha', type=float, help="计算auc_rocs+F1+acc-args.alpha*(tmp_parity+tmp_equality)",default=1)
    parser.add_argument('--beta', type=float, help="计算acc-args.beta * (tmp_parity + tmp_equality)作为选择最好模型的指标从而保存",default=1)



    # 训练轮数参数调整
    parser.add_argument('--epochs', type=int, help="整体模型微调的总轮数", default=20)
    parser.add_argument('--df_epochs', type=int, help="判别器F微调的总轮数", default=10)
    parser.add_argument('--class_epochs', type=int, help="训练分类器微调的总轮数", default=50)
    parser.add_argument('--ad_MLP_F_epochs', type=int, help="对抗训练公平网络F微调的总轮数", default=20)
    parser.add_argument('--db_epochs', type=int, help="判别器B微调的总轮数", default=50)
    parser.add_argument('--align_epochs', type=int, help="公平网络与偏见网络疏远，encoder,classify微调的总轮数",
                        default=10)
    parser.add_argument('--de_together_epochs', type=int, help="数据编辑中共同训练的总轮数", default=100)
    parser.add_argument('--de_separate_epochs', type=int, help="数据编辑中分离训练的总轮数", default=3)
    parser.add_argument('--de_separate_node_epochs', type=int, help="数据编辑中分离训练中结点特征的总轮数", default=10)
    parser.add_argument('--de_separate_edge_epochs', type=int, help="数据编辑中分离训练中边的权重特征的总轮数",
                        default=10)






    args = parser.parse_args()
    args.strlist = None
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(args)
    data, _ , args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_dataset(
        args.dataset, args.inid, args.top_k)


    # 设置 test set类别
    data2 = []
    if args.dataset == "credit":

        args.strlist = [args.inid]
        for i in range(len(args.strlist)):
            datatmp, _, _, _, _, _ = get_dataset(
                args.dataset,  args.strlist[i], args.top_k)
            data2.append(datatmp)
    elif args.dataset == "bail":
        args.strlist = [args.inid]
        for i in range(len(args.strlist)):
            datatmp, _, _, _, _, _ = get_dataset(
                args.dataset,  args.strlist[i], args.top_k)
            data2.append(datatmp)
    elif args.dataset == "pokec":
        args.strlist = [args.inid]
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

        print("==========={}============".format(args.inid+args.strlist[i]))
        print('Acc: ', np.mean(acc.T[i]))
        print('auc_roc: ', np.mean(auc_roc.T[i]))
        print('parity: ', np.mean(parity.T[i]))
        print('equality: ', np.mean(equality.T[i]))
        print('f1: ', np.mean(f1.T[i]))



