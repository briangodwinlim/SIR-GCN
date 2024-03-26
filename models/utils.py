import dgl
import torch
from torch import nn


class VirtualNode(nn.Module):
    def __init__(self, use_vn, hidden_dim, residual, mod_emb, mod_pool):
        super(VirtualNode, self).__init__()
        self._use_vn = use_vn
        self.residual = residual
        self.mod_emb = mod_emb if use_vn else None
        self.mod_pool = mod_pool if use_vn else None
        self.init_emb = nn.Embedding(1, hidden_dim) if use_vn else None

    def node_emb(self, graphs, nfeats, vnfeat=None):
        if self._use_vn:
            vnfeat = self.init_emb(torch.zeros(graphs.batch_size, device=graphs.device).long()) if vnfeat is None else vnfeat
            batch_id = dgl.broadcast_nodes(graphs, torch.arange(graphs.batch_size, device=graphs.device).unsqueeze(-1)).squeeze() if graphs.batch_size > 1 else 0
            nfeats = nfeats + vnfeat[batch_id]
        return nfeats, vnfeat

    def vn_emb(self, graphs, nfeats, vnfeat):
        if self._use_vn:
            vnfeat_ = self.mod_pool(graphs, nfeats) + vnfeat
            vnfeat_ = self.mod_emb(graphs, vnfeat_)
            vnfeat = vnfeat_ + vnfeat if self.residual else vnfeat_
        return vnfeat


class CentralityEncoder(nn.Module):
    def __init__(self, max_degree, embedding_dim, direction='both'):
        super(CentralityEncoder, self).__init__()
        self.max_degree = max_degree
        self.direction = direction
        
        if direction in ['in', 'both'] and max_degree > 0:
            self.encoder_in = nn.Embedding(max_degree + 1, embedding_dim, padding_idx=0)
        if direction in ['out', 'both'] and max_degree > 0:
            self.encoder_out = nn.Embedding(max_degree + 1, embedding_dim, padding_idx=0)
            
    def forward(self, graphs, nfeats):
        if self.max_degree == 0:
            return nfeats
        else:
            in_degrees = graphs.in_degrees().clamp(min=0, max=self.max_degree)
            out_degrees = graphs.out_degrees().clamp(min=0, max=self.max_degree)

            if self.direction in ['in', 'both']:
                nfeats = nfeats + self.encoder_in(in_degrees)
            if self.direction in ['out', 'both']:
                nfeats = nfeats + self.encoder_out(out_degrees)
                
            return nfeats
