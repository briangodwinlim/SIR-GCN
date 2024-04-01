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


class GraphBatchNorm(nn.BatchNorm1d):
    def forward(self, graphs, feats):
        return super().forward(feats)


class GraphLayerNorm(nn.LayerNorm):
    def forward(self, graphs, feats):
        return super().forward(feats)


class GraphIdentity(nn.Identity):
    def forward(self, graphs, feats):
        return super().forward(feats)


class GetNorm(nn.Module):
    def __init__(self, norm, with_graph, *args, **kwargs):
        super(GetNorm, self).__init__()
        if with_graph:
            norm_list = {'gn': GraphNorm, 'cn': GraphContraNorm, 'bn': GraphBatchNorm, 'ln': GraphLayerNorm, 'none': GraphIdentity}
        else:
            norm_list = {'cn': ContraNorm, 'bn': nn.BatchNorm1d, 'ln': nn.LayerNorm, 'none': nn.Identity}
            
        if norm not in norm_list:
            raise NotImplementedError(f'norm = {norm} not implemented')
            
        self.norm = norm_list[norm](*args, **kwargs)
    
    def forward(self, *args):
        return self.norm(*args)
