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

# 创建一个tensorboard网页，并将文件放在文件地址中
writer = SummaryWriter('./log/bord1')

# 创建折线图，标题是唯一标识！

def update_labels_by_neighbors_with_predictions(data, encoder, classifier):
    """
    首先为所有节点生成预测标签。
    然后，将每个节点的标签修改为其邻居节点的【预测标签】中出现最多的类别。
    """
    # --- 1. 一次性为所有节点生成预测标签（优化后） ---
    # 将模型设置为评估模式
    encoder.eval()
    classifier.eval()
    with torch.no_grad():  # 在推理时不需要计算梯度
        # GNN模型一次性处理所有节点，效率更高
        h = encoder(data.x, data.edge_index, data.adj_norm_sp)  # 假设encoder不需要 adj_norm_sp
        output = classifier(h)
        # 获取每个节点的预测类别 (取概率最大的那个)
        predict_labels = torch.argmax(output, dim=1)

    num_nodes = data.x.shape[0]
    new_labels = data.y.clone()  # 初始化new_labels，可以基于原始标签或全零
    edge_index = data.edge_index

    # --- 2. 遍历每个节点，根据邻居的【预测标签】更新标签 ---
    for node_idx in range(num_nodes):
        # 找到当前节点的所有邻居 (修正后的逻辑)
        # 查找所有源节点是当前节点的边，其对应的目标节点就是邻居
        mask_source = edge_index[0] == node_idx
        neighbors_from_source = edge_index[1][mask_source]

        # 查找所有目标节点是当前节点的边，其对应的源节点也是邻居 (适用于无向图)
        mask_target = edge_index[1] == node_idx
        neighbors_from_target = edge_index[0][mask_target]

        # 合并所有邻居并去重
        all_neighbors = torch.unique(torch.cat([neighbors_from_source, neighbors_from_target]))

        # 如果节点有邻居
        if len(all_neighbors) > 0:
            # --- 3. 核心修改：使用邻居的`predict_labels`而不是`data.y` ---
            neighbor_predicted_labels = predict_labels[all_neighbors]

            # 统计每个预测标签的出现次数
            unique_labels, counts = torch.unique(neighbor_predicted_labels, return_counts=True)

            # 找到出现次数最多的预测标签
            if len(counts) > 0:
                most_common_label_idx = torch.argmax(counts)
                most_common_label = unique_labels[most_common_label_idx]

                # 更新当前节点的标签
                new_labels[node_idx] = most_common_label
        # else: 如果节点没有邻居，则其标签保持不变

    # 在data对象中创建一个新的属性来存储更新后的标签
    data.new_y = new_labels

    return data

def run(data, args, data2):
    #pbar = tqdm(range(args.runs), unit='run')
    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
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




    classifier = torch.load('classifier_model.pth', weights_only=False).to(args.device)
    encoder = torch.load('encoder_model.pth',weights_only=False).to(args.device)


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

    share_MLP = MLP_encoder(args).to(args.device)
    MLP_F = MLP_encoder(args).to(args.device)
    MLP_B = MLP_encoder(args).to(args.device)
    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_all = torch.optim.Adam([
        dict(params=encoder.lin.parameters(), weight_decay=args.e_wd),
        dict(params=classifier.lin.parameters(), weight_decay=args.c_wd),
        dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)], lr=0.01)
    
    '==========train============='
    for count in range(args.runs):
        seed_everything(count + args.seed)

        best_val_tradeoff = 0
        best_val_loss = math.inf
        encoder.train()
        classifier.train()
        discriminator.train()
        pbar = tqdm(range(200), unit='epoch')

        for epoch in pbar:
            pbar.set_description("epoch {}".format(epoch+1))
            optimizer_all.zero_grad()
            data1 = update_labels_by_neighbors_with_predictions(data, encoder, classifier)

            h = encoder(data1.x, data1.edge_index, data1.adj_norm_sp)
            h1 = share_MLP(h)
            h_F = MLP_F(h1)
            h_B = MLP_B(h1)
            output_F = discriminator(h_F)
            output_B = discriminator(h_B)
            loss_f = criterion(output_F.view(-1),
                               data1.x[:, args.sens_idx])
            loss_b = -criterion(output_B.view(-1),
                               data1.x[:, args.sens_idx])
            output = classifier(h_F)
            loss_c = F.binary_cross_entropy_with_logits(output[data1.train_mask],
                                                            data.new_y[data.train_mask].unsqueeze(1)).to(
                    args.device)
            loss = loss_f + loss_b + loss_c

            writer.add_scalar('loss_all', loss, global_step=epoch)
            writer.add_scalar('loss_c', loss_c, global_step=epoch)
            writer.add_scalar('loss_f', loss_f, global_step=epoch)
            writer.add_scalar('loss_b', loss_b, global_step=epoch)

            loss.backward()
            optimizer_all.step()
            discriminator.eval()

            # train classifier and encoder


        best_val_tradeoff = 0
        best_val_loss = math.inf

        encoder.eval()
        classifier.eval()
        "=====test======="
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
                    data2[i].x, classifier, encoder, data2[i], args)


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
    parser.add_argument('--inid', type=str, default='_B1')
    parser.add_argument('--outid', type=str, default='all')
    parser.add_argument('--runs', type=int, default=1) # 5
    parser.add_argument('--start', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=300)
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
    parser.add_argument('--seed', type=int, default=1)
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

