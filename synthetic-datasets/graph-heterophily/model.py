import sys
sys.path.append('../..')

import dgl
from torch import nn
from models.utils import MLP
from models.conv import SIRConv


class SIRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0, **kwargs):
        super(SIRModel, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)

        self.convs = nn.ModuleList([
            SIRConv(hidden_dim, hidden_dim, hidden_dim, nn.ReLU(inplace=True)) 
            for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

        self.regression = nn.Linear(hidden_dim, output_dim, bias=False)
        self.pooling = dgl.nn.SumPooling()

    def forward(self, graphs, feats):
        feats = self.embedding(feats)

        for i in range(self.num_layers):
            feats = self.convs[i](graphs, feats)
            feats = self.drop(feats)

        feats = self.regression(feats)
        feats = self.pooling(graphs, feats)
        return feats


class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0, **kwargs):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)

        self.convs = nn.ModuleList([
            dgl.nn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
            for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

        self.pooling = dgl.nn.SumPooling()
        self.regression = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, graphs, feats):
        feats = self.embedding(feats)

        for i in range(self.num_layers):
            feats = self.convs[i](graphs, feats)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.regression(feats)
        return feats


class SAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0, **kwargs):
        super(SAGEModel, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)

        self.convs = nn.ModuleList([
            dgl.nn.SAGEConv(hidden_dim, hidden_dim, 'pool')
            for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

        self.pooling = dgl.nn.SumPooling()
        self.regression = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, graphs, feats):
        feats = self.embedding(feats)

        for i in range(self.num_layers):
            feats = self.convs[i](graphs, feats)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.regression(feats)
        return feats


class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0, num_heads=1, **kwargs):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)

        self.convs = nn.ModuleList([
            dgl.nn.GATv2Conv(hidden_dim, hidden_dim, num_heads, allow_zero_in_degree=True, share_weights=True)
            for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

        self.pooling = dgl.nn.SumPooling()
        self.regression = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, graphs, feats):
        feats = self.embedding(feats)

        for i in range(self.num_layers):
            feats = self.convs[i](graphs, feats).mean(dim=1)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.regression(feats)
        return feats


class GINModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0, mlp_layers=1, **kwargs):
        super(GINModel, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)

        self.convs = nn.ModuleList([
            dgl.nn.GINConv(MLP(hidden_dim, hidden_dim, hidden_dim, mlp_layers, 0, 'none', nn.ReLU(inplace=True), True, False))
            for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

        self.pooling = dgl.nn.SumPooling()
        self.regression = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, graphs, feats):
        feats = self.embedding(feats)

        for i in range(self.num_layers):
            feats = self.convs[i](graphs, feats)
            feats = self.drop(feats)

        feats = self.pooling(graphs, feats)
        feats = self.regression(feats)
        return feats
