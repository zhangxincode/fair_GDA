import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torch.sparse

from model import *

from learn import *

import warnings

warnings.filterwarnings('ignore')
import math
from pandas import DataFrame
from utils import read_config
class DataAug(nn.Module):
    def __init__(self, classifier, encoder,num_nodes, num_features,num_edge,args):
        super(DataAug, self).__init__()
        self.classifier = classifier
        self.encoder = encoder
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.temperature = 0.7

        # 初始化扰动参数，确保这些参数是可训练的
        self.delta_X = torch.zeros_like(torch.ones(num_nodes,num_features), requires_grad=True,device=args.device)  # 节点特征扰动

        # `delta_A` 用于扰动边索引，所以它的维度应该和 edge_index 一样
        # self.delta_A = torch.zeros_like(torch.ones(2,num_edge), dtype=torch.float, requires_grad=True)  # 邻接矩阵扰动
        self.delta_A = torch.zeros_like(torch.ones(num_edge), dtype=torch.float, requires_grad=True,device=args.device)  # 邻接矩阵扰动

        # 冻结 encoder 和 classifier 的参数
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False


    def forward(self, data,args,eps = 1e-6):

        self.delta_X = self.delta_X .to(args.device)
        x1 = data.x + self.delta_X  # 可训练扰动


        h = self.encoder(x1, data.edge_index, data.adj_norm_sp,edge_weight=self.delta_A)
        h0 = h.clone()
        h0[:, h0.sen_idx] = 0

        h1 = h.clone()
        h1[:, h1.sen_idx] = 1


        batch_size = h1.shape[0]
        z1 = F.normalize(h0)
        z2 = F.normalize(h1)
        sim = torch.mm(z1, z2.t()) / self.temperature  # [batch_size, batch_size]

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

        loss = (loss_i + loss_j) / 2
        return loss, self.delta_X,self.delta_A


def train_data_aug(data,data_aug,args):
    data33 = data.clone()

    optimizer = optim.Adam([data_aug.delta_X,data_aug.delta_A], lr=0.5)
    # pbar1 =  tqdm(range(100))
    for epoch in range(args.de_together_epochs):#pbar1:
        # pbar1.set_description("epoch {}".format(epoch))
        optimizer.zero_grad()
        loss,_,_= data_aug(data33,args)
        loss.backward()
        optimizer.step()
        # pbar1.set_postfix({"loss": "{:.3f}".format(loss)})
    with torch.no_grad():
        loss,delta_X,delta_A= data_aug(data33,args)
        data33.x = data33.x + delta_X
        #对敏感属性不进行训练
        data33.x[:, 1] = data33.sens
        data33.edge_weight = delta_A
    return data33










class DE_X(nn.Module):
    def __init__(self, encoder, num_nodes, num_features, num_edge,args):
        super(DE_X, self).__init__()
        self.encoder = encoder
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.temperature = 0.7

        # 初始化扰动参数，确保这些参数是可训练的
        self.delta_X = torch.zeros_like(torch.ones(num_nodes, num_features), requires_grad=True,device=args.device)  # 节点特征扰动
        self.delta_A = torch.zeros_like(torch.ones(num_edge), dtype=torch.float, requires_grad=False)  # 邻接矩阵扰动

        # 冻结 encoder 和 classifier 的参数
        for param in self.encoder.parameters():
            param.requires_grad = False


    def forward(self, data,args, eps=1e-6):
        self.delta_X = self.delta_X.to(args.device)
        x1 = data.x + self.delta_X  # 可训练扰动


        h = self.encoder(x1, data.edge_index, data.adj_norm_sp, edge_weight=self.delta_A.to(args.device))

        h0 = h.clone()
        h0[:, data.sen_idx] = 0

        h1 = h.clone()
        h1[:, data.sen_idx] = 1
        batch_size = h1.shape[0]
        z1 = F.normalize(h0)
        z2 = F.normalize(h1)
        sim = torch.mm(z1, z2.t()) / self.temperature  # [batch_size, batch_size]

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

        loss = (loss_i + loss_j) / 2
        return loss, self.delta_X, self.delta_A

class DE_A(nn.Module):
    def __init__(self, encoder, num_nodes, num_features, num_edge,args):
        super(DE_A, self).__init__()
        self.encoder = encoder
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.temperature = 0.7

        # 初始化扰动参数，确保这些参数是可训练的
        self.delta_X = torch.zeros_like(torch.ones(num_nodes, num_features), requires_grad=False)  # 节点特征扰动
        self.delta_A = torch.zeros_like(torch.ones(num_edge), dtype=torch.float, requires_grad=True,device=args.device)  # 邻接矩阵扰动

        # 冻结 encoder 和 classifier 的参数
        for param in self.encoder.parameters():
            param.requires_grad = False


    def forward(self, data,args, eps=1e-6):
        self.delta_X = self.delta_X.to(args.device)
        x1 = data.x + self.delta_X  # 可训练扰动

        h = self.encoder(x1, data.edge_index, data.adj_norm_sp, edge_weight=self.delta_A)

        h0 = h.clone()
        h0[:, data.sen_idx] = 0

        h1 = h.clone()
        h1[:, data.sen_idx] = 1
        batch_size = h1.shape[0]
        z1 = F.normalize(h0)
        z2 = F.normalize(h1)
        sim = torch.mm(z1, z2.t()) / self.temperature  # [batch_size, batch_size]

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

        loss = (loss_i + loss_j) / 2
        return loss, self.delta_X, self.delta_A










def train_data_edit(data, de_a,de_x,args):
    data33 = data.clone()

    optimizer_X = optim.Adam([de_x.delta_X], lr=args.de_feature_lr)
    optimizer_A = optim.Adam([de_a.delta_A], lr=args.de_edge_lr)
    # pbar1 =  tqdm(range(100))

    for epoch in range(args.de_separate_epochs):  # pbar1:
        for epoch1 in range(args.de_separate_node_epochs):
            # pbar1.set_description("epoch {}".format(epoch))
            optimizer_X.zero_grad()
            loss1, _, _ = de_x(data33,args)
            loss1.backward()
            optimizer_X.step()
        for epoch2 in range(args.de_separate_edge_epochs):
            optimizer_A.zero_grad()
            loss2, _, _ = de_a(data33,args)
            loss2.backward()
            optimizer_A.step()

    with torch.no_grad():
        loss, delta_X, deA = de_x(data33,args)
        loss,dex,delta_A = de_a(data33,args)
        data33.x = data33.x + delta_X
        # 对敏感属性不进行训练
        data33.x[:, data33.sen_idx] = data.x[:, data.sen_idx]
        data33.edge_weight = delta_A
    return data33