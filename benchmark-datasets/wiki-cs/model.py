import sys
sys.path.append('../..')

import dgl
import torch
from torch import nn
from models.utils import MLP
from models.norm import GetNorm
from models.conv import SIRConv


class SIRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, input_dropout=0, edge_dropout=0, dropout=0, norm='none',
                 readout_layers=1, readout_dropout=0, jumping_knowledge=True, residual=False, resid_layers=0, resid_dropout=0, 
                 feat_dropout=0, agg_type='mean', **kwargs):
        super(SIRModel, self).__init__()
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_dropout)
        self.edge_drop = dgl.transforms.DropEdge(edge_dropout)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.jumping_knowledge = jumping_knowledge

        self.convs = nn.ModuleList()
        self.resids = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.readouts = nn.ModuleList([MLP(input_dim, hidden_dim, output_dim, readout_layers, readout_dropout, 'none', self.activation, False, False) if jumping_knowledge else None])

        for i in range(num_layers):
            _input_dim = hidden_dim if i > 0 else input_dim
            self.convs.append(SIRConv(_input_dim, hidden_dim, hidden_dim, self.activation, feat_dropout, agg_type=agg_type))
            self.resids.append(MLP(_input_dim, hidden_dim, hidden_dim, resid_layers, resid_dropout, 'none', self.activation, False, False) if residual else None)
            self.norms.append(GetNorm(norm, False, hidden_dim))
            self.readouts.append(MLP(hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, 'none', self.activation, False, False) if jumping_knowledge or i == num_layers - 1 else None)
    
    def forward(self, graph, feats):
        feats = self.input_drop(feats)

        feats_list = [feats if self.jumping_knowledge else 0]
        for i in range(self.num_layers):
            graph_ = self.edge_drop(graph)
            feats_resid = self.resids[i](feats) if self.resids[i] is not None else 0
            feats = self.convs[i](graph_, feats) + feats_resid
            feats = self.norms[i](feats)
            feats = self.activation(feats)
            feats = self.drop(feats)
            feats_list.append(feats if self.jumping_knowledge else 0)
            
        score = torch.sum(torch.stack([self.readouts[i](feats) for i, feats in enumerate(feats_list)], dim=0), dim=0) if self.jumping_knowledge else self.readouts[-1](feats)
        return score


class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, input_dropout=0, edge_dropout=0, dropout=0, norm='none',
                 readout_layers=1, readout_dropout=0, jumping_knowledge=True, residual=False,
                 num_heads=1, attn_dropout=0, **kwargs):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_dropout)
        self.edge_drop = dgl.transforms.DropEdge(edge_dropout)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.jumping_knowledge = jumping_knowledge

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.readouts = nn.ModuleList([MLP(input_dim, hidden_dim, output_dim, readout_layers, readout_dropout, 'none', self.activation, False, False) if jumping_knowledge else None])

        for i in range(num_layers):
            _input_dim = num_heads * hidden_dim if i > 0 else input_dim
            self.convs.append(dgl.nn.GATv2Conv(_input_dim, hidden_dim, num_heads, attn_drop=attn_dropout, residual=residual, 
                                               allow_zero_in_degree=True, bias=False, share_weights=True))
            self.norms.append(GetNorm(norm, False, num_heads * hidden_dim))
            self.readouts.append(MLP(num_heads * hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, 'none', self.activation, False, False) if jumping_knowledge or i == num_layers - 1 else None)
        
    def forward(self, graph, feats):
        feats = self.input_drop(feats)

        feats_list = [feats if self.jumping_knowledge else 0]
        for i in range(self.num_layers):
            graph_ = self.edge_drop(graph)
            feats = self.convs[i](graph_, feats)
            feats = feats.flatten(1)
            feats = self.norms[i](feats)
            feats = self.activation(feats)
            feats = self.drop(feats)
            feats_list.append(feats if self.jumping_knowledge else 0)
    
        score = torch.sum(torch.stack([self.readouts[i](feats) for i, feats in enumerate(feats_list)], dim=0), dim=0) if self.jumping_knowledge else self.readouts[-1](feats)
        return score
