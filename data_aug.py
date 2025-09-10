'''
进行data edit学习的网络，
对于标签平滑，使用的是标签平滑正则化，即对标签进行平滑处理，使标签分布更加均匀，避免模型过拟合。
对于模型调整，使用的是解纠缠表示学习，即通过解纠缠表示学习，使模型的表示能力更强，从而提高模型的性能。
使用 将数据编辑中的相似性计算融入模型损失计算之中
'''



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
from data_edit import DE_X,DE_A,train_data_edit,train_data_aug,DataAug







def run(data, args, data2):
    acc, f1, auc_roc, parity, equality = np.zeros([args.runs, len(data2)]), np.zeros([args.runs, len(data2)]), np.zeros(
        [args.runs, len(data2)]), np.zeros([args.runs, len(data2)]), np.zeros([args.runs, len(data2)])
    '==========train============='
    for count in range(args.runs):
        criterion = nn.BCEWithLogitsLoss()
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
        t_num_s0_y1, t_num_s1_y1, t_num_s0_y0, t_num_s1_y0, = sum(t_idx_s0_y1), sum(t_idx_s1_y1), sum(t_idx_s0_y0), sum(
            t_idx_s1_y0),

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

        MLP_F = MLP_net(args).to(args.device)
        optimizer_F = torch.optim.Adam(params=MLP_F.parameters(), lr=args.fairnet_lr)

        MLP_B = MLP_net(args).to(args.device)
        optimizer_B = torch.optim.Adam(params=MLP_B.parameters(), lr=args.baisnet_lr)

        discriminator_F = MLP_discriminator(args).to(args.device)
        optimizer_D_F = torch.optim.Adam(params=discriminator_F.parameters(), lr=args.discri_F_lr)

        discriminator_B = MLP_discriminator(args).to(args.device)
        optimizer_D_B = torch.optim.Adam(params=discriminator_B.parameters(), lr=args.discri_B_lr)
        # 注意这的文件地址
        if args.dataset == 'bail':
            train_name = "_B0"
        elif args.dataset == "credit":
            train_name = "_C0"
        elif args.dataset == 'pokec':
            train_name = "_z"
        classifier = torch.load('./model_para/{}/{}_classifier_best_0.pth'.format(args.dataset, train_name),
                                weights_only=False).to(args.device)  # ,map_location='cpu'
        optimizer_C = torch.optim.Adam(params=classifier.parameters(), lr=args.classify_lr)

        encoder = torch.load('./model_para/{}/{}_encoder_best_0.pth'.format(args.dataset, train_name),
                             weights_only=False).to(args.device)
        optimizer_E = torch.optim.Adam(params=encoder.parameters(), lr=args.encoder_lr)

        data3 = data.clone()
        data_aug = DataAug(classifier, encoder, data3.x.shape[0], data3.x.shape[1], data3.edge_index.shape[1], args)
        de_a = DE_A(encoder, data3.x.shape[0], data3.x.shape[1], data3.edge_index.shape[1], args)
        de_x = DE_X(encoder, data3.x.shape[0], data3.x.shape[1], data3.edge_index.shape[1], args)
        data1 = update_labels_by_neighbors_with_predictions(data3, encoder, classifier)
        seed_everything(args.seed+count)
        discriminator_B.reset_parameters()
        discriminator_F.reset_parameters()
        classifier.reset_parameters()
        encoder.reset_parameters()
        MLP_F.reset_parameters()
        MLP_B.reset_parameters()

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
                pred_B = discriminator_F(h_F)
                # 判别器的损失是两个分支损失之和
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
                output_class = classifier(h_F)
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
                pred_F_adv = torch.sigmoid(discriminator_F(h_F))
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
                pred_B_adv = discriminator_B(h_B)
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

                # 分类损失
                h_F = MLP_F(h)
                h_B = MLP_B(h)
                logits = classifier(h_F)
                task_loss = criterion(logits, data1.new_probs[:,1].unsqueeze(1).float())

                # 分离损失
                hF = F.normalize(h_F, dim=1)
                hB = F.normalize(h_B, dim=1)
                cos_sim = (hF * hB).sum(dim=1)
                loss_ortho = (cos_sim ** 2).mean()

                # 相似性损失
                h0 = h.clone()
                h0[:, data.sen_idx] = 0

                h1 = h.clone()
                h1[:, data.sen_idx] = 1
                batch_size = h1.shape[0]
                z1 = F.normalize(h0)
                z2 = F.normalize(h1)
                sim = torch.mm(z1, z2.t()) / 0.5  # [batch_size, batch_size]

                # 构造正样本的相似度向量（对角线）
                pos_sim = sim.diag().view(batch_size, 1)  # [batch_size, 1]

                denom = torch.logsumexp(sim, dim=1, keepdim=True)  # [batch_size, 1]

                # 每个锚点的损失： - (正样本的相似度 - 分母)
                loss_i = - (pos_sim - denom).mean()

                # 对称地，从z2到z1
                denom_j = torch.logsumexp(sim.t(), dim=1, keepdim=True)
                loss_j = - (pos_sim - denom_j).mean()  # 注意，这里正样本的相似度还是对角线，但转置后对角线不变

                # 或者，也可以重新计算以z2为锚点的正样本，但注意，正样本还是对角线，所以可以直接用sim.t().diag()
                # 但这里我们直接使用对称性，用同样的pos_sim（因为对角线不变）

                loss_similar = (loss_i + loss_j) / 2

                #loss = (0.5 * task_loss + 0.2 * loss_ortho ) / (task_loss + loss_ortho )
                # if args.inid == '_B2':
                #     loss = (0.3*task_loss + 0.2 * loss_ortho + 0.5 * loss_similar)/(task_loss + loss_ortho + loss_similar)
                # else: #if args.inid == '_B1':
                loss = (0.5*task_loss + 0.2 * loss_ortho + 0.3 * loss_similar)/(task_loss + loss_ortho + loss_similar) # C2-4 :424
                if args.inid =='_C2' or args.inid =='_C3' or args.inid =='_C4':
                    loss = (0.4*task_loss + 0.2 * loss_ortho + 0.4 * loss_similar)/(task_loss + loss_ortho + loss_similar) # C2-4 :424
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
    parser.add_argument('--inid', type=str, help="作为输入的数据集",default='_B1')
    parser.add_argument('--runs', type=int, help="运行次数",default=5) # 5

    parser.add_argument('--encoder', type=str,help="编码器encoder种类", default='GCN')
    parser.add_argument('--prop', type=str, help="CCN的选择",default='scatter')

    parser.add_argument('--hidden', type=int,help="编码器encoder输出特征的维度 ", default=16)

    parser.add_argument('--seed', type=int,help="初始化种子",default=22)# B1->14
    parser.add_argument('--gpu', type=int, help="使用的gpu编号,若没有自动变为cpu",default=0)

    parser.add_argument('--dropout', type=float, help="编码器encoder的dropout概率",default=0.5)
    parser.add_argument('--top_k', type=int,help="利用superman算法算得与敏感属性前K个相似的特征",default=10)
    parser.add_argument('--alpha', type=float, help="计算auc_rocs+F1+acc-args.alpha*(tmp_parity+tmp_equality)",default=1)

    # 学习率参数
    parser.add_argument('--fairnet_lr', type=float, help="公平网络学习率 ", default=0.001)
    parser.add_argument('--baisnet_lr', type=float, help="偏见网络学习率", default=0.001)
    parser.add_argument('--discri_F_lr', type=float, help="判别器F的学习率 ", default=0.01)
    parser.add_argument('--discri_B_lr', type=float, help="判别器B的学习率 ", default=0.01)
    parser.add_argument('--classify_lr', type=float, help="分类器的学习率 ", default=0.01)
    parser.add_argument('--encoder_lr', type=float, help="编码器的学习率 ", default=0.01)

    parser.add_argument('--de_feature_lr', type=float, help="数据编辑结点特征训练的学习率 ", default=0.001)
    parser.add_argument('--de_edge_lr', type=float, help="数据编辑边特征训练的学习率 ", default=0.001)

    # 训练轮数参数调整
    parser.add_argument('--epochs', type=int, help="整体模型微调的总轮数", default=15)
    parser.add_argument('--df_epochs', type=int, help="判别器F微调的总轮数", default=20)
    parser.add_argument('--db_epochs', type=int, help="判别器B微调的总轮数", default=20)
    parser.add_argument('--class_epochs', type=int, help="训练分类器微调的总轮数", default=40)
    parser.add_argument('--ad_MLP_F_epochs', type=int, help="对抗训练公平网络F微调的总轮数", default=20)
    parser.add_argument('--align_epochs', type=int, help="公平网络与偏见网络疏远，encoder,classify微调的总轮数",default=10)


    # 数据编辑的相关参数

    parser.add_argument('--de_train', type=int, help="数据编辑每几个epoch进行一次训练", default=5)
    parser.add_argument('--de_traintype_switch', type=int, help="是采用结点特征和边共同训练(1)，还是分离训练开关(0)。",
                        default=0)
    parser.add_argument('--de_together_epochs', type=int, help="数据编辑中共同训练的总轮数", default=5)
    parser.add_argument('--de_separate_epochs', type=int, help="数据编辑中分离训练的总轮数", default=3)
    parser.add_argument('--de_separate_node_epochs', type=int, help="数据编辑中分离训练中结点特征的总轮数", default=5)
    parser.add_argument('--de_separate_edge_epochs', type=int, help="数据编辑中分离训练中边的权重特征的总轮数",
                        default=5)


    args = parser.parse_args()
    args.strlist = None
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(args)
    seed_everything(args.seed)

    # 获取数据
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
        print('Acc     :',['{:.5f}'.format(x)if x!=0 else "feile!!" for x in acc.T[i]])
        print('auc_roc :',['{:.5f}'.format(x)if x!=0 else "feile!!"for x in auc_roc.T[i]])
        print('F1      :',['{:.5f}'.format(x)if x!=0 else "feile!!"for x in f1.T[i]])
        print('parity  :',['{:.5f}'.format(x)if x!=0 else "feile!!"for x in parity.T[i]])
        print('equality:',['{:.5f}'.format(x)if x!=0 else "feile!!"for x in equality.T[i]])
    for i in range(len(args.strlist)):
        print("==========={}============".format(args.inid+args.strlist[i]))
        print('Acc     :{:.5f}'.format( 100*np.mean(acc.T[i])))
        print('auc_roc :{:.5f}'.format( 100*np.mean(auc_roc.T[i])))
        print('F1      :{:.5f}'.format( 100*np.mean(f1.T[i])))
        print('parity  :{:.5f}'.format( 100*np.mean(parity.T[i])))
        print('equality:{:.5f}'.format( 100*np.mean(equality.T[i])))




