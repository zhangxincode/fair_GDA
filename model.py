from torch.nn import Linear
from utils import *
from torch import nn
from torch.nn import Parameter
from rubish.source import GCNConv
from torch_geometric.nn import GINConv, SAGEConv


class MLP_discriminator(torch.nn.Module):
    def __init__(self, args):
        super(MLP_discriminator, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, 1)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None, mask_node=None):
        h = self.lin(h)

        return h# torch.sigmoid(h)


class MLP_classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_classifier, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, 1)#args.num_classes+1)

    def clip_parameters(self):
        for p in self.lin.parameters():
            p.data.clamp_(-self.args.clip_c, self.args.clip_c)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None):
        # import ipdb; ipdb.set_trace()
        h1 = self.lin(h)

        return h1


class MLP_net(torch.nn.Module):
    def __init__(self, args):
        super(MLP_net, self).__init__()
        self.args = args

        self.lin = nn.Sequential(
            nn.Linear(args.hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, args.hidden)
        )

    def reset_parameters(self):
        for layer in self.lin:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def forward(self, x, edge_index=None, mask_node=None,edge_weights=None):
        h = self.lin(x)

        return h










class GCN_encoder_scatter(torch.nn.Module):
    '''
    base model for GCN

    '''
    def __init__(self, args):
        super(GCN_encoder_scatter, self).__init__()

        self.args = args

        self.lin = Linear(args.num_features, args.hidden, bias=False)

        self.bias = Parameter(torch.Tensor(args.hidden))

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def propagate(self, x, edge_index, edge_weight=None):
        """ feature propagation procedure: sparsematrix
        """
        edge_index, _ = add_remaining_self_loops(
            edge_index, num_nodes=x.size(0))

        # calculate the degree normalize term
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
        if edge_weight is None:
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]  
        else:
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col] + edge_weight  
        # normalize the features on the starting point of the edge
        out = edge_weight.view(-1, 1) * x[row]

        return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')

    def forward(self, x, edge_index, adj_norm_sp,edge_weight=None):
        h = self.lin(x)
        if edge_weight is not None:
            h1 = self.propagate(h, edge_index, edge_weight) + self.bias
        else:
            h1 = self.propagate(h, edge_index)+ self.bias
        return h1















class GCN_2(nn.Module):
    def __init__(self, args):
        super(GCN_2, self).__init__()
        self.body = GCN_Body(args.num_features, args.hidden, args.dropout)
        self.fc = nn.Linear(args.hidden, args.hidden)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self):
        for m in self.modules():
            self.weights_init(m)

    def forward(self, x, edge_index, adj):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x



class GCN_encoder_spmm(torch.nn.Module):
    def __init__(self, args):
        super(GCN_encoder_spmm, self).__init__()

        self.args = args

        self.lin = Linear(args.num_features, args.hidden, bias=False)

        self.bias = Parameter(torch.Tensor(args.hidden))

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index, adj_norm_sp, edge_weight=None):
        h = 1#self.lin(x)
        h = torch.spmm(adj_norm_sp, h) + self.bias
        # h = propagate2(h, edge_index) + self.bias

        return h


class GIN_encoder(nn.Module):
    def __init__(self, args):
        super(GIN_encoder, self).__init__()

        self.args = args

        self.mlp = nn.Sequential(
            nn.Linear(args.num_features, args.hidden),
            # nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            # nn.Linear(args.hidden, args.hidden),
        )

        self.conv = GINConv(self.mlp)

    def clip_parameters(self, channel_weights):
        for i in range(self.mlp[0].weight.data.shape[1]):
            self.mlp[0].weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                      self.args.clip_e * channel_weights[i])

        # self.mlp[0].weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.mlp[0].weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

        # for p in self.conv.parameters():
        #     p.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.conv(x, edge_index)
        return h


class SAGE_encoder(nn.Module):
    def __init__(self, args):
        super(SAGE_encoder, self).__init__()

        self.args = args

        self.conv1 = SAGEConv(args.num_features, args.hidden, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Dropout(p=args.dropout)
        )
        self.conv2 = SAGEConv(args.hidden, args.hidden, normalize=True)
        self.conv2.aggr = 'mean'

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def clip_parameters(self, channel_weights):
        for i in range(self.conv1.lin_l.weight.data.shape[1]):
            self.conv1.lin_l.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                           self.args.clip_e * channel_weights[i])

        for i in range(self.conv1.lin_r.weight.data.shape[1]):
            self.conv1.lin_r.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                           self.args.clip_e * channel_weights[i])

        # for p in self.conv1.parameters():
        #     p.data.clamp_(-self.args.clip_e, self.args.clip_e)
        # for p in self.conv2.parameters():
        #     p.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def forward(self, x, edge_index, adj_norm_sp):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        h = self.conv2(x, edge_index)
        return h
