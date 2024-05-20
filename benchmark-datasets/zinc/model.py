import sys
sys.path.append('../..')

import dgl
import torch
from torch import nn
from models.conv import SIRConv, SIREConv
from models.utils import MLP, DropEdge
from models.norm import GetNorm


class SIREConv2(SIREConv):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, activation, dropout=0, bias=True, agg_type='sum'):
        super(SIREConv2, self).__init__(input_dim, edge_dim, hidden_dim, output_dim, activation, dropout, bias, agg_type)
        self.linear_edge = nn.Embedding(edge_dim, hidden_dim)


class SIRModel(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, num_layers=1, input_dropout=0, edge_dropout=0, dropout=0, norm='none', 
                 readout_layers=1, readout_dropout=0, readout_pooling='sum', jumping_knowledge=True,
                 residual=False, resid_layers=0, resid_dropout=0, feat_dropout=0, agg_type='sum', **kwargs):
        super(SIRModel, self).__init__()
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_dropout)
        self.edge_drop = DropEdge(edge_dropout)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.jumping_knowledge = jumping_knowledge
        self.node_encoder = nn.Embedding(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.resids = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(SIRConv(hidden_dim, hidden_dim, hidden_dim, self.activation, feat_dropout, agg_type=agg_type))
            # self.convs.append(SIREConv2(hidden_dim, edge_dim, hidden_dim, hidden_dim, self.activation, feat_dropout, agg_type=agg_type))
            self.resids.append(MLP(hidden_dim, hidden_dim, hidden_dim, resid_layers, resid_dropout, 'none', self.activation, False, False) if residual else None)
            self.norms.append(GetNorm(norm, True, hidden_dim))
            
        self.pool = dgl.nn.SumPooling() if readout_pooling == 'sum' else dgl.nn.AvgPooling()
        self.readouts = nn.ModuleList([MLP(hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, 'none', self.activation, False, False)
                                       for _ in range(num_layers * int(jumping_knowledge) + 1)])
    
    def forward(self, graphs, nfeats, efeats):
        nfeats = self.input_drop(self.node_encoder(nfeats))

        nfeats_list = [nfeats if self.jumping_knowledge else 0]
        for i in range(self.num_layers):
            graphs_, efeats_ = self.edge_drop(graphs, efeats)
            nfeats_resid = self.resids[i](nfeats) if self.resids[i] is not None else 0
            nfeats = self.convs[i](graphs_, nfeats) + nfeats_resid
            # nfeats = self.convs[i](graphs_, nfeats, efeats_) + nfeats_resid
            nfeats = self.norms[i](graphs, nfeats)
            nfeats = self.activation(nfeats)
            nfeats = self.drop(nfeats)
            nfeats_list.append(nfeats if self.jumping_knowledge else 0)

        score = torch.sum(torch.stack([self.readouts[i](nfeats) for i, nfeats in enumerate(nfeats_list)], dim=0), dim=0) if self.jumping_knowledge else self.readouts[0](nfeats)
        score = self.pool(graphs, score)
        return score


class GINModel(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, num_layers=1, input_dropout=0, edge_dropout=0, dropout=0, norm='none', 
                 readout_layers=1, readout_dropout=0, readout_pooling='sum', jumping_knowledge=True,
                 residual=False, resid_layers=0, resid_dropout=0, mlp_layers=1, agg_type='sum', **kwargs):
        super(GINModel, self).__init__()
        self.num_layers = num_layers
        self.input_drop = nn.Dropout(input_dropout)
        self.edge_drop = DropEdge(edge_dropout)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.jumping_knowledge = jumping_knowledge
        self.node_encoder = nn.Embedding(input_dim, hidden_dim)

        # self.embds = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.resids = nn.ModuleList()
        self.combs = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(dgl.nn.GINConv(aggregator_type=agg_type))
            # self.embds.append(nn.Embedding(edge_dim, hidden_dim))
            # self.convs.append(dgl.nn.GINEConv())
            self.resids.append(MLP(hidden_dim, hidden_dim, hidden_dim, resid_layers, resid_dropout, 'none', self.activation, False, False) if residual else None)
            self.combs.append(MLP(hidden_dim, hidden_dim, hidden_dim, mlp_layers, dropout, norm, self.activation))
            
        self.pool = dgl.nn.SumPooling() if readout_pooling == 'sum' else dgl.nn.AvgPooling()
        self.readouts = nn.ModuleList([MLP(hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, 'none', self.activation, False, False)
                                       for _ in range(num_layers * int(jumping_knowledge) + 1)])
    
    def forward(self, graphs, nfeats, efeats):
        nfeats = self.input_drop(self.node_encoder(nfeats))

        nfeats_list = [nfeats if self.jumping_knowledge else 0]
        for i in range(self.num_layers):
            graphs_, efeats_ = self.edge_drop(graphs, efeats)
            nfeats_resid = self.resids[i](nfeats) if self.resids[i] is not None else 0
            nfeats = self.convs[i](graphs_, nfeats)
            # nfeats = self.convs[i](graphs_, nfeats, self.embds[i](efeats_))
            nfeats = self.combs[i](graphs, nfeats) + nfeats_resid
            nfeats_list.append(nfeats if self.jumping_knowledge else 0)

        score = torch.sum(torch.stack([self.readouts[i](nfeats) for i, nfeats in enumerate(nfeats_list)], dim=0), dim=0) if self.jumping_knowledge else self.readouts[0](nfeats)
        score = self.pool(graphs, score)
        return score
