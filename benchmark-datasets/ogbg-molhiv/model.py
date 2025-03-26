import sys
sys.path.append('../..')

import dgl
import torch
from torch import nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from models.utils import MLP, VirtualNode, CentralityEncoder, DropEdge
from models.conv import SIRConv, SIREConv
from models.norm import GetNorm


# class SIREConv2(SIREConv):
#     def __init__(self, input_dim, hidden_dim, output_dim, activation, dropout=0, inner_bias=True, outer_bias=True, agg_type='sum'):
#         super(SIREConv2, self).__init__(input_dim, 1, hidden_dim, output_dim, activation, dropout, inner_bias, outer_bias, agg_type)
#         self.linear_edge = BondEncoder(hidden_dim)


# MLP architecture following EGC
class MLP_EGC(nn.Module):
    def __init__(self, layers, activation, dropout=0):
        super(MLP_EGC, self).__init__()
        self.num_layers = len(layers) - 1
        self.activation = activation
        self.drop = nn.Dropout(dropout)
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(self.num_layers):
            _input_dim = layers[i]
            _output_dim = layers[i + 1]
            self.linears.append(nn.Linear(_input_dim, _output_dim))
            
            if i < self.num_layers - 1:
                self.norms.append(nn.BatchNorm1d(_output_dim))
        
    def forward(self, feats):
        for i in range(self.num_layers - 1):
            feats = self.linears[i](feats)
            feats = self.norms[i](feats)
            feats = self.activation(feats)
            feats = self.drop(feats)
        
        feats = self.linears[-1](feats)
        return feats


# SIR-GCN model architecture following EGC
class SIRModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=1, input_dropout=0, edge_dropout=0, dropout=0, norm='none', 
                 readout_layers=1, readout_dropout=0, readout_pooling='sum', jumping_knowledge=True, 
                 virtual_node=False, vn_layers=0, vn_dropout=0, vn_residual=False, rand_feat=False, max_degree=0, 
                 residual=False, resid_layers=0, resid_dropout=0, feat_dropout=0, agg_type='sum', **kwargs):
        super(SIRModel, self).__init__()
        self.residual = residual
        self.num_layers = num_layers
        self.embedding = AtomEncoder(hidden_dim)
        self.input_dropout = nn.Dropout(input_dropout)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(SIRConv(hidden_dim, hidden_dim, hidden_dim, self.activation, feat_dropout, agg_type=agg_type))
            self.norms.append(GetNorm(norm, True, hidden_dim))

        self.pool = dgl.nn.SumPooling() if readout_pooling == 'sum' else dgl.nn.AvgPooling()
        self.readout = MLP_EGC([hidden_dim, hidden_dim // 2, hidden_dim // 4, output_dim], self.activation)
        
    def forward(self, graphs, nfeats, efeats, nfeats_perturb=0):
        nfeats = self.embedding(nfeats)
        nfeats = self.input_dropout(nfeats)

        for i in range(self.num_layers):
            nfeats_resid = nfeats
            nfeats = self.convs[i](graphs, nfeats)
            nfeats = self.norms[i](graphs, nfeats)
            nfeats = self.activation(nfeats)
            
            if self.residual:
                nfeats = nfeats + nfeats_resid

        nfeats = self.pool(graphs, nfeats)
        return self.readout(nfeats)


# class SIRModel(nn.Module):
#     def __init__(self, hidden_dim, output_dim, num_layers=1, input_dropout=0, edge_dropout=0, dropout=0, norm='none', 
#                  readout_layers=1, readout_dropout=0, readout_pooling='sum', jumping_knowledge=True, 
#                  virtual_node=False, vn_layers=0, vn_dropout=0, vn_residual=False, rand_feat=False, max_degree=0, 
#                  residual=False, resid_layers=0, resid_dropout=0, feat_dropout=0, agg_type='sum', **kwargs):
#         super(SIRModel, self).__init__()
#         self.num_layers = num_layers
#         self.drop = nn.Dropout(dropout)
#         self.input_drop = nn.Dropout(input_dropout)
#         self.edge_drop = DropEdge(edge_dropout)
        
#         self.rand_feat = rand_feat
#         self.jumping_knowledge = jumping_knowledge

#         self.activation = nn.LeakyReLU(0.2, inplace=True)
#         self.node_encoder = AtomEncoder(hidden_dim)
#         self.central_encoder = CentralityEncoder(max_degree, hidden_dim, 'in')
#         hidden_dim = hidden_dim + int(rand_feat)

#         self.vn = VirtualNode(use_vn=virtual_node, hidden_dim=hidden_dim, residual=vn_residual,
#                               mod_emb=MLP(hidden_dim, hidden_dim, hidden_dim, vn_layers, vn_dropout, 'bn', self.activation),
#                               mod_pool=dgl.nn.SumPooling())

#         self.convs = nn.ModuleList()
#         self.resids = nn.ModuleList()
#         self.norms = nn.ModuleList()
        
#         for _ in range(num_layers):
#             # self.convs.append(SIRConv(hidden_dim, hidden_dim, hidden_dim, self.activation, feat_dropout, agg_type=agg_type))
#             self.convs.append(SIREConv2(hidden_dim, hidden_dim, hidden_dim, self.activation, feat_dropout, agg_type=agg_type))
#             self.resids.append(MLP(hidden_dim, hidden_dim, hidden_dim, resid_layers, resid_dropout, 'none', self.activation, False, False) if residual else None)
#             self.norms.append(GetNorm(norm, True, hidden_dim))

#         self.pool = dgl.nn.SumPooling() if readout_pooling == 'sum' else dgl.nn.AvgPooling()
#         self.readouts = nn.ModuleList([MLP(hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, 'none', self.activation, False, False) 
#                                        for _ in range(num_layers * int(jumping_knowledge) + 1)])

#     def forward(self, graphs, nfeats, efeats, nfeats_perturb=0):
#         vnfeat = None
#         nfeats = self.input_drop(self.node_encoder(nfeats)) + nfeats_perturb
#         nfeats = self.central_encoder(graphs, nfeats)

#         if self.rand_feat:
#             nfeats = torch.cat((nfeats, torch.rand((nfeats.shape[0], 1), device=nfeats.device)), dim=-1)
        
#         nfeats_list = [nfeats if self.jumping_knowledge else 0]
#         for i in range(self.num_layers):
#             nfeats, vnfeat = self.vn.node_emb(graphs, nfeats, vnfeat)
#             graphs_, efeats_ = self.edge_drop(graphs, efeats)
#             nfeats_resid = self.resids[i](nfeats) if self.resids[i] is not None else 0
#             # nfeats = self.convs[i](graphs_, nfeats) + nfeats_resid
#             nfeats = self.convs[i](graphs_, nfeats, efeats_)
#             nfeats = self.norms[i](graphs, nfeats)
#             nfeats = self.activation(nfeats)
#             nfeats = nfeats + nfeats_resid
#             nfeats = self.drop(nfeats)
#             nfeats_list.append(nfeats if self.jumping_knowledge else 0)
#             vnfeat = self.vn.vn_emb(graphs, nfeats, vnfeat)

#         score = torch.sum(torch.stack([self.readouts[i](nfeats) for i, nfeats in enumerate(nfeats_list)], dim=0), dim=0) if self.jumping_knowledge else self.readouts[0](nfeats)
#         score = self.pool(graphs, score)
#         return score


class GINModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=1, input_dropout=0, edge_dropout=0, dropout=0, norm='none', 
                 readout_layers=1, readout_dropout=0, readout_pooling='sum', jumping_knowledge=True, 
                 virtual_node=False, vn_layers=0, vn_dropout=0, vn_residual=False, rand_feat=False, max_degree=0, 
                 residual=False, resid_layers=0, resid_dropout=0, mlp_layers=1, agg_type='sum', **kwargs):
        super(GINModel, self).__init__()
        self.num_layers = num_layers
        self.input_drop = nn.Dropout(input_dropout)
        self.edge_drop = DropEdge(edge_dropout)
        
        self.rand_feat = rand_feat
        self.jumping_knowledge = jumping_knowledge

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.node_encoder = AtomEncoder(hidden_dim)
        self.central_encoder = CentralityEncoder(max_degree, hidden_dim, 'in')
        hidden_dim = hidden_dim + int(rand_feat)

        self.vn = VirtualNode(use_vn=virtual_node, hidden_dim=hidden_dim, residual=vn_residual,
                              mod_emb=MLP(hidden_dim, hidden_dim, hidden_dim, vn_layers, vn_dropout, 'bn', self.activation),
                              mod_pool=dgl.nn.SumPooling())

        self.embds = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.resids = nn.ModuleList()
        self.combs = nn.ModuleList()
        
        for _ in range(num_layers):
            # self.convs.append(dgl.nn.GINConv(aggregator_type=agg_type))
            self.embds.append(BondEncoder(hidden_dim))
            self.convs.append(dgl.nn.GINEConv())
            self.resids.append(MLP(hidden_dim, hidden_dim, hidden_dim, resid_layers, resid_dropout, 'none', self.activation, False, False) if residual else None)
            self.combs.append(MLP(hidden_dim, hidden_dim, hidden_dim, mlp_layers, dropout, norm, self.activation))

        self.pool = dgl.nn.SumPooling() if readout_pooling == 'sum' else dgl.nn.AvgPooling()
        self.readouts = nn.ModuleList([MLP(hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, 'none', self.activation, False, False) 
                                       for _ in range(num_layers * int(jumping_knowledge) + 1)])

    def forward(self, graphs, nfeats, efeats, nfeats_perturb=0):
        vnfeat = None
        nfeats = self.input_drop(self.node_encoder(nfeats)) + nfeats_perturb
        nfeats = self.central_encoder(graphs, nfeats)

        if self.rand_feat:
            nfeats = torch.cat((nfeats, torch.rand((nfeats.shape[0], 1), device=nfeats.device)), dim=-1)
        
        nfeats_list = [nfeats if self.jumping_knowledge else 0]
        for i in range(self.num_layers):
            nfeats, vnfeat = self.vn.node_emb(graphs, nfeats, vnfeat)
            graphs_, efeats_ = self.edge_drop(graphs, efeats)
            nfeats_resid = self.resids[i](nfeats) if self.resids[i] is not None else 0
            # nfeats = self.convs[i](graphs_, nfeats)
            nfeats = self.convs[i](graphs_, nfeats, self.embds[i](efeats_))
            nfeats = self.combs[i](graphs, nfeats) + nfeats_resid
            nfeats_list.append(nfeats if self.jumping_knowledge else 0)
            vnfeat = self.vn.vn_emb(graphs, nfeats, vnfeat)

        score = torch.sum(torch.stack([self.readouts[i](nfeats) for i, nfeats in enumerate(nfeats_list)], dim=0), dim=0) if self.jumping_knowledge else self.readouts[0](nfeats)
        score = self.pool(graphs, score)
        return score
