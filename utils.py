from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter
import random
import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import pandas as pd
from sklearn.manifold import TSNE
import yaml









def eval_tool(pre,y,sens):
    acc = ((pre == y).sum() / y.shape[0])  # 伪标签准确率
    acc_1 = ((pre == 1).int() * (y == 1).int()).sum() / (y == 1).int().sum()  # 伪标签为1的准确率
    acc_0 = ((pre == 0).int() * (y == 0).int()).sum() / (y == 0).int().sum()  # 伪标签为0的准确率
    TP = ((pre == 1).int() * (y == 1).int()).sum()
    TN = ((pre == 0).int() * (y == 0).int()).sum()
    FP = ((pre == 1).int() * (y == 0).int()).sum()
    FN = ((pre == 0).int() * (y == 1).int()).sum()
    F1 = 2*TP/(2*TP+FP+FN) # 针对数据不平衡问题
    parity, equality = fair_metric(pre.cpu().numpy(), y.cpu().numpy(), sens.cpu().numpy())
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
    eval_tool(torch.argmax(P0, dim=1), data.y,data.sens)  # 直接预测的伪标签准确率
    eval_tool(data.new_y, data.y,data.sens)  # 传播后的伪标签准确率
    return data













def drop_feature(x, drop_prob, sens_idx, sens_flag=True):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob

    x = x.clone()
    drop_mask[sens_idx] = False

    x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

    # Flip sensitive attribute
    if sens_flag:
        x[:, sens_idx] = 1-x[:, sens_idx]

    return x

def propagate(x, edge_index, edge_weight=None):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    if(edge_weight == None):
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def propagate_mask(x, edge_index, mask_node=None):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(
        edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    if(mask_node == None):
        mask_node = torch.ones_like(x[:, 0])

    mask_node = mask_node[row]
    mask_node[row == col] = 1

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row] * \
        mask_node.view(-1, 1)

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def propagate2(x, edge_index):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(
        edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def seed_everything(seed=0):

    # Python hash
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)
    torch.backends.cudnn.allow_tf32 = False

    # PyTorch CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # 强制 PyTorch 1.8+ 所有操作确定性
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)

    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.allow_tf32 = False
    #
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = True
    # # torch.use_deterministic_algorithms(True)
    #
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    # idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    # idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    idx_s0_y1 = idx_s0 == (labels == 1)
    idx_s1_y1 = idx_s1 == (labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) -
                 sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) -
                   sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()

def fair_metric2(pred, labels, idx_s0_y1, idx_s1_y1, num_s0_y1, num_s1_y1):
    # idx_s0 = sens == 0
    # idx_s1 = sens == 1
    # idx_s0_y1 = torch.logical_and(idx_s0, labels == 1)
    # idx_s1_y1 = torch.logical_and(idx_s1, labels == 1)
    # parity = abs(sum(pred[idx_s0]) / sum(idx_s0) -
    #              sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / num_s0_y1 -
                   sum(pred[idx_s1_y1]) / num_s1_y1)
    return equality


def visual(model, data, sens, dataname):
    model.eval()

    print(data.y, sens)
    hidden = model.encoder(data.x, data.edge_index).cpu().detach().numpy()
    sens, data.y = sens.cpu().numpy(), data.y.cpu().numpy()
    idx_s0, idx_s1, idx_s2, idx_s3 = (sens == 0) & (data.y == 0), (sens == 0) & (
        data.y == 1), (sens == 1) & (data.y == 0), (sens == 1) & (data.y == 1)

    tsne_hidden = TSNE(n_components=2)
    tsne_hidden_x = tsne_hidden.fit_transform(hidden)

    tsne_input = TSNE(n_components=2)
    tsne_input_x = tsne_input.fit_transform(data.x.detach().cpu().numpy())

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    items = [tsne_input_x, tsne_hidden_x]
    names = ['input', 'hidden']

    for ax, item, name in zip(axs, items, names):
        ax.scatter(item[idx_s0][:, 0], item[idx_s0][:, 1], s=1,
                   c='red', marker='o', label='class 1, group1')
        ax.scatter(item[idx_s1][:, 0], item[idx_s1][:, 1], s=1,
                   c='blue', marker='o', label='class 2, group1')
        ax.scatter(item[idx_s2][:, 0], item[idx_s2][:, 1], s=10,
                   c='red', marker='', label='class 1, group2')
        ax.scatter(item[idx_s3][:, 0], item[idx_s3][:, 1], s=10,
                   c='blue', marker='+', label='class 2, group2')

        ax.set_title(name)
    ax.legend(frameon=0, loc='upper center',
              ncol=4, bbox_to_anchor=(-0.2, 1.2))

    plt.savefig(dataname + 'visual_tsne.pdf',
                dpi=1000, bbox_inches='tight')


def visual_sub(model, data, sens, dataname, k=50):
    idx_c1, idx_c2 = torch.where((sens == 0) == True)[
        0], torch.where((sens == 1) == True)[0]

    idx_subc1, idx_subc2 = idx_c1[torch.randperm(
        idx_c1.shape[0])[:k]], idx_c2[torch.randperm(idx_c2.shape[0])[:k]]

    idx_sub = torch.cat([idx_subc1, idx_subc2]).cpu().numpy()
    sens = sens[idx_sub]
    y = data.y[idx_sub]

    model.eval()

    hidden = model.encoder(data.x, data.edge_index).cpu().detach().numpy()
    sens, y = sens.cpu().numpy(), y.cpu().numpy()
    idx_s0, idx_s1, idx_s2, idx_s3 = (sens == 0) & (y == 0), (sens == 0) & (
        y == 1), (sens == 1) & (y == 0), (sens == 1) & (y == 1)

    tsne_hidden = TSNE(n_components=2)
    tsne_hidden_x = tsne_hidden.fit_transform(hidden)

    tsne_input = TSNE(n_components=2)
    tsne_input_x = tsne_input.fit_transform(data.x.detach().cpu().numpy())

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    items = [tsne_input_x[idx_sub], tsne_hidden_x[idx_sub]]
    names = ['input', 'hidden']

    for ax, item, name in zip(axs, items, names):
        ax.scatter(item[idx_s0][:, 0], item[idx_s0][:, 1], s=1,
                   c='red', marker='.', label='group1 class1')
        ax.scatter(item[idx_s1][:, 0], item[idx_s1][:, 1], s=5,
                   c='red', marker='*', label='group1 class2')
        ax.scatter(item[idx_s2][:, 0], item[idx_s2][:, 1], s=1,
                   c='blue', marker='.', label='group2 class1')
        ax.scatter(item[idx_s3][:, 0], item[idx_s3][:, 1], s=5,
                   c='blue', marker='*', label='group2 class2')

        ax.set_title(name)
    ax.legend(frameon=0, loc='upper center',
              ncol=4, bbox_to_anchor=(-0.2, 1.2))

    plt.savefig(dataname + 'visual_tsne.pdf',
                dpi=1000, bbox_inches='tight')


def pos_neg_mask(label, nodenum, train_mask):
    pos_mask = torch.stack([(label == label[i]).float()
                            for i in range(nodenum)])
    neg_mask = 1 - pos_mask

    return pos_mask[train_mask, :][:, train_mask], neg_mask[train_mask, :][:, train_mask]


def pos_neg_mask_sens(sens_label, label, nodenum, train_mask):
    pos_mask = torch.stack([((label == label[i]) & (sens_label == sens_label[i])).float()
                            for i in range(nodenum)])
    neg_mask = torch.stack([((label == label[i]) & (sens_label != sens_label[i])).float()
                            for i in range(nodenum)])

    return pos_mask[train_mask, :][:, train_mask], neg_mask[train_mask, :][:, train_mask]


def similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


def InfoNCE(h1, h2, pos_mask, neg_mask, tau=0.2):
    num_nodes = h1.shape[0]

    sim = similarity(h1, h2) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)

    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)

    return loss.mean()


def random_aug(x, edge_index, args):
    x_flip = flip_sens_feature(x, args.sens_idx, args.flip_node_ratio)

    edge_index1 = random_mask_edge(edge_index, args)
    edge_index2 = random_mask_edge(edge_index, args)

    mask1 = random_mask_node(x, args)
    mask2 = random_mask_node(x, args)

    return x_flip, edge_index1, edge_index2, mask1, mask2


def random_aug2(x, edge_index, args):
    # x_flip = flip_sens_feature(x, args.sens_idx, args.flip_node_ratio)
    edge_index = random_mask_edge(edge_index, args)

    mask = random_mask_node(x, args)

    return edge_index, mask


def flip_sens_feature(x, sens_idx, flip_node_ratio):
    node_num = x.shape[0]
    idx = np.arange(0, node_num)
    samp_idx = np.random.choice(idx, size=int(
        node_num * flip_node_ratio), replace=False)

    x_flip = x.clone()
    x_flip[:, sens_idx] = 1 - x_flip[:, sens_idx]

    return x_flip


def random_mask_edge(edge_index, args):
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        node_num = edge_index.size(0)
        edge_index = torch.stack([row, col], dim=0)

        edge_num = edge_index.shape[1]
        idx = np.arange(0, edge_num)
        samp_idx = np.random.choice(idx, size=int(
            edge_num * args.mask_edge_ratio), replace=False)

        mask = torch.ones(edge_num, dtype=torch.bool)
        mask[samp_idx] = 0

        edge_index = edge_index[:, mask]

        edge_index = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            value=None, sparse_sizes=(node_num, node_num),
            is_sorted=True)

    else:
        edge_index, _ = add_remaining_self_loops(
            edge_index)
        edge_num = edge_index.shape[1]
        idx = np.arange(0, edge_num)
        samp_idx = np.random.choice(idx, size=int(
            edge_num * args.mask_edge_ratio), replace=False)

        mask = torch.ones_like(edge_index[0, :], dtype=torch.bool)
        mask[samp_idx] = 0

        edge_index = edge_index[:, mask]

    return edge_index


def random_mask_node(x, args):
    node_num = x.shape[0]
    idx = np.arange(0, node_num)
    samp_idx = np.random.choice(idx, size=int(
        node_num * args.mask_node_ratio), replace=False)

    mask = torch.ones_like(x[:, 0])
    mask[samp_idx] = 0

    return mask


def consis_loss(ps, temp=0.5):
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p

    avg_p = sum_p / len(ps)

    sharp_p = (torch.pow(avg_p, 1. / temp) /
               torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()

    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return 1 * loss


def sens_correlation(features, sens_idx):
    corr = pd.DataFrame(np.array(features)).corr()
    return corr[sens_idx].to_numpy()


def visualize(embeddings, y, s):
    X_embed = TSNE(n_components=2, learning_rate='auto',
                   init='random').fit_transform(embeddings)

    group1 = (y == 0) & (s == 0)
    group2 = (y == 0) & (s == 1)
    group3 = (y == 1) & (s == 0)
    group4 = (y == 1) & (s == 1)

    plt.scatter(X_embed[group1, 0], X_embed[group1, 1],
                s=5, c='tab:blue', marker='o')
    plt.scatter(X_embed[group2, 0], X_embed[group2, 1],
                s=5, c='tab:orange', marker='s')
    plt.scatter(X_embed[group3, 0], X_embed[group3, 1],
                s=5, c='tab:blue', marker='o')
    plt.scatter(X_embed[group4, 0], X_embed[group4, 1],
                s=5, c='tab:orange', marker='s')

def read_config(args):
    # specify the model family

    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    yamlPath = os.path.join(fileNamePath, 'config/{}.yaml'.format(args.times))
    print(yamlPath)
    with open(yamlPath, 'r', encoding='utf-8') as f:
        cont = f.read()
        config_dict = yaml.safe_load(cont)['g'][args.dataset]

    if args.gpu == -1:
        device = torch.device('cpu')
    elif args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device('cuda', int(args.gpu))
        else:
            print("cuda is not available, please set 'gpu' -1")
    for key, value in config_dict.items():
        args.__setattr__(key, value)

    return args




data = [
    (0.1, 0.1, 65.97473, 63.93735,  0.00927, 3.24839),
    (0.1, 0.3, 79.96390, 67.77624,  2.24094, 1.64228),
    (0.1, 0.5, 63.53791, 67.82912,  2.61840, 6.49941),
    (0.1, 0.7, 78.24910, 59.99177,  1.70322, 3.17607),
    (0.1, 0.9, 71.66065, 59.37231,  2.72568, 7.32778),
    (0.3, 0.1, 71.20939, 66.93308,  0.81982, 4.77233),
    (0.3, 0.3, 74.54874, 59.37147,  1.33106, 3.47915),
    (0.3, 0.5, 75.81227, 60.80303,  1.87937, 3.70663),
    (0.3, 0.7, 60.74007, 67.86773,  2.19194, 5.49288),
    (0.5, 0.1, 78.70036, 69.61813,  3.68192, 2.61134),
    (0.5, 0.3, 64.25993, 67.53587,  0.23178, 2.37860),
    (0.5, 0.5, 65.43321, 63.56777,  1.58137, 5.59413),
    (0.5, 0.2, 74.90975, 66.09547,  0.37084, 0.09073),
    (0.7, 0.1, 72.83394, 63.98033,  3.88191, 5.87551),
    (0.9, 0.1, 71.20939, 64.86659,  0.27681, 4.31738),
    (1.0, 0.0, 71.66065, 70.23309,  1.57872, 3.65140),
    (0.5, 0.0, 68.23105, 66.97529,  1.68335, 5.99713),
]