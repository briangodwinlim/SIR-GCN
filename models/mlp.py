import dgl
import torch
from torch import nn
import torch.nn.functional as F


class GraphNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, bias=True):
        super(GraphNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else 0
        self.mean_scale = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, graphs, feats):
        batch_nodes = graphs.batch_num_nodes().long()
        batch_id = dgl.broadcast_nodes(graphs, torch.arange(graphs.batch_size, device=graphs.device).unsqueeze(-1)).expand_as(feats)
        
        mean = torch.zeros(graphs.batch_size, *feats.shape[1:], device=feats.device)
        mean = mean.scatter_add_(0, batch_id, feats)
        mean = (mean.T / batch_nodes).T
        mean = mean.repeat_interleave(batch_nodes, dim=0)
        demean = feats - mean * self.mean_scale

        std = torch.zeros(graphs.batch_size, *feats.shape[1:], device=feats.device)
        std = std.scatter_add_(0, batch_id, torch.pow(demean, 2))
        std = torch.sqrt((std.T / batch_nodes).T + self.eps)
        std = std.repeat_interleave(batch_nodes, dim=0)
        return self.weight * demean / std + self.bias


class ContraNorm(nn.Module):
    def __init__(self, hidden_dim, scale=0, temp=1, use_scale=False):
        super(ContraNorm, self).__init__()
        self.temp = temp
        self.scale = scale
        self.use_scale = use_scale
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, feats):
        weights = F.softmax(torch.mm(feats.T, feats) / self.temp, dim=1)
        multiplier = 1 + int(self.use_scale) * self.scale
        feats = multiplier * feats - self.scale * torch.mm(feats, weights)
        feats = self.norm(feats)
        return feats


class GraphContraNorm(ContraNorm):
    def forward(self, graphs, feats):
        return super().forward(feats)


class BatchNorm(nn.BatchNorm1d):
    def forward(self, graphs, feats):
        return super().forward(feats)


class LayerNorm(nn.LayerNorm):
    def forward(self, graphs, feats):
        return super().forward(feats)


class GraphIdentity(nn.Module):
    def forward(self, graphs, feats):
        return feats


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, 
                 dropout, norm, activation, include_last=True, with_graph=True, **kwargs):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.include_last = include_last
        self.with_graph = with_graph
        self.drop = nn.Dropout(dropout)
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            _input_dim = hidden_dim if i > 0 else input_dim
            _output_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.linears.append(nn.Linear(_input_dim, _output_dim))
            
            if i < num_layers - 1 or include_last:
                if with_graph:
                    if norm == 'gn':
                        self.norms.append(GraphNorm(_output_dim, **kwargs))
                    if norm == 'cn':
                        self.norms.append(GraphContraNorm(_output_dim, **kwargs))
                    if norm == 'bn':
                        self.norms.append(BatchNorm(_output_dim, **kwargs))
                    if norm == 'ln':
                        self.norms.append(LayerNorm(_output_dim, **kwargs))
                    if norm == 'none':
                        self.norms.append(GraphIdentity())
                else:
                    if norm == 'gn':
                        raise Exception('Cannot use GraphNorm when with_graph = False')
                    if norm == 'cn':
                        self.norms.append(ContraNorm(_output_dim, **kwargs))
                    if norm == 'bn':
                        self.norms.append(nn.BatchNorm1d(_output_dim, **kwargs))
                    if norm == 'ln':
                        self.norms.append(nn.LayerNorm(_output_dim, **kwargs))
                    if norm == 'none':
                        self.norms.append(nn.Identity())
    
    def forward(self, *args):
        if self.with_graph:
            [graphs, feats] = args
            for i in range(self.num_layers):
                feats = self.linears[i](feats)
                if i < self.num_layers - 1 or self.include_last:
                    feats = self.norms[i](graphs, feats)
                    feats = self.activation(feats)
        
        else:
            [feats] = args
            for i in range(self.num_layers):
                feats = self.linears[i](feats)
                if i < self.num_layers - 1 or self.include_last:
                    feats = self.norms[i](feats)
                    feats = self.activation(feats)

        feats = self.drop(feats)
        return feats
