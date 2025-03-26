import sys
sys.path.append('../..')

import dgl
import torch
from torch import nn
from models.utils import MLP
from models.norm import GetNorm
from models.conv import SIRConv


class SIRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, input_dropout=0, dropout=0, norm='none',
                 residual=False, feat_dropout=0, agg_type='mean', **kwargs):
        super(SIRModel, self).__init__()
        self.num_layers = num_layers
        self.residual = residual
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.input_drop = nn.Dropout(input_dropout)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        self.norms = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        
        for _ in range(num_layers):
            self.norms.append(GetNorm(norm, False, hidden_dim))
            self.convs.append(SIRConv(hidden_dim, hidden_dim, hidden_dim, self.activation, feat_dropout, agg_type=agg_type))
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.output_norm = GetNorm(norm, False, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph, feats):
        feats = self.input_linear(feats)
        feats = self.input_drop(feats)
        feats = self.activation(feats)

        for i in range(self.num_layers):
            feats_resid = feats
            feats = self.norms[i](feats)
            feats = self.convs[i](graph, feats)
            feats = self.drop(feats)
            feats = self.activation(feats)
            feats = self.linears[i](feats)
            feats = self.drop(feats)
        
            if self.residual:
                feats = feats + feats_resid

        feats = self.output_norm(feats)
        feats = self.output_linear(feats)

        return feats.squeeze(1)
