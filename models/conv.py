import torch
from torch import nn
from dgl import function as fn
from dgl.utils import expand_as_pair


class SIRConv(nn.Module):
    r"""Semi-Isomorphic Relational Graph Convolution Network (SIR-GCN)
    
    .. math::
        h_u^* = \sum_{v \in \mathcal{N}(u)} W_R ~ \sigma(W_Q h_u + W_K h_v)

    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden feature dimension
    output_dim : int
        Output feature dimension
    activation : a callable layer
        Activation function, the :math:`\sigma` in the formula
    dropout : float, optional
        Dropout rate for inner linear transformations, defaults to 0
    bias : bool, optional
        Whether to learn an additive bias, defaults to True
    agg_type : str, optional
        Aggregator type to use (``sum``, ``max``, ``mean``, or ``sym``), defaults to ``sum``
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation, dropout=0, bias=True, agg_type='sum'):
        super(SIRConv, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear_query = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.linear_key = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.linear_relation = nn.Linear(hidden_dim, output_dim, bias=bias)

        self._agg_type = agg_type
        self._agg_func = fn.sum if agg_type == 'sym' else getattr(fn, agg_type)
    
    def message_func(self, edges):
        if self._agg_type in ['sum', 'mean', 'sym']:
            return {'m': edges.src['norm'] * edges.dst['norm'] * self.activation(edges.dst['eq'] + edges.src['ek'])}
        else:
            return {'m': self.linear_relation(self.activation(edges.dst['eq'] + edges.src['ek']))}
        
    def forward(self, graph, feat):
        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1).to(graph.device)
            norm = torch.pow(degs, -0.5) if self._agg_type == 'sym' else torch.ones(graph.num_nodes(), device=graph.device)
            norm = norm.reshape((graph.num_nodes(),) + (1,) * (feat.dim() - 1))
            graph.ndata['norm'] = norm
 
            feat_key, feat_query = expand_as_pair(feat, graph)
            graph.ndata['ek'] = self.dropout(self.linear_key(feat_key))
            graph.ndata['eq'] = self.dropout(self.linear_query(feat_query))

            graph.update_all(self.message_func, self._agg_func('m', 'ft'))
            rst = graph.ndata.pop('ft')
            rst = self.linear_relation(rst) if self._agg_type in ['sum', 'mean', 'sym'] else rst
            
            return rst


class SIREConv(nn.Module):
    r"""Semi-Isomorphic Relational Graph Convolution Network (SIR-GCN) with Edge Features
    
    .. math::
        h_u^* = \sum_{v \in \mathcal{N}(u)} W_R ~ \sigma(W_Q h_u + W_E h_{u,v} + W_K h_v)

    Parameters
    ----------
    input_dim : int
        Input node feature dimension
    edge_dim : int
        Input edge feature dimension
    hidden_dim : int
        Hidden feature dimension
    output_dim : int
        Output feature dimension
    activation : a callable layer
        Activation function, the :math:`\sigma` in the formula
    dropout : float, optional
        Dropout rate for inner linear transformations, defaults to 0
    bias : bool, optional
        Whether to learn an additive bias, defaults to True
    agg_type : str, optional
        Aggregator type to use (``sum``, ``max``, ``mean``, or ``sym``), defaults to ``sum``
    """
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, activation, dropout=0, bias=True, agg_type='sum'):
        super(SIREConv, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear_query = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.linear_key = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.linear_edge = nn.Linear(edge_dim, hidden_dim, bias=bias)
        self.linear_relation = nn.Linear(hidden_dim, output_dim, bias=bias)

        self._agg_type = agg_type
        self._agg_func = fn.sum if agg_type == 'sym' else getattr(fn, agg_type)
    
    def message_func(self, edges):
        if self._agg_type in ['sum', 'mean', 'sym']:
            return {'m': edges.src['norm'] * edges.dst['norm'] * self.activation(edges.dst['eq'] + edges.src['ek'] + edges.data['e'])}
        else:
            return {'m': self.linear_relation(self.activation(edges.dst['eq'] + edges.src['ek'] + edges.data['e']))}
    
    def forward(self, graph, nfeat, efeat):
        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1).to(graph.device)
            norm = torch.pow(degs, -0.5) if self._agg_type == 'sym' else torch.ones(graph.num_nodes(), device=graph.device)
            norm = norm.reshape((graph.num_nodes(),) + (1,) * (nfeat.dim() - 1))
            graph.ndata['norm'] = norm
 
            nfeat_key, nfeat_query = expand_as_pair(nfeat, graph)
            graph.ndata['ek'] = self.dropout(self.linear_key(nfeat_key))
            graph.ndata['eq'] = self.dropout(self.linear_query(nfeat_query))
            graph.edata['e'] = self.dropout(self.linear_edge(efeat))

            graph.update_all(self.message_func, self._agg_func('m', 'ft'))
            rst = graph.ndata.pop('ft')
            rst = self.linear_relation(rst) if self._agg_type in ['sum', 'mean', 'sym'] else rst
            
            return rst


class SIRConvBase(nn.Module):
    r"""Semi-Isomorphic Relational Graph Convolution Network (SIR-GCN) Base Class
    
    .. math::
        h_u^* = \sum_{v \in \mathcal{N}(u)} g([h_u \Vert h_v])

    Parameters
    ----------
    message_func : callable layer
        The message function :math:`g` in the formula
    agg_type : str, optional
        Aggregator type to use (``sum``, ``max``, ``mean``, or ``sym``), defaults to ``sum``
    """
    def __init__(self, message_func, agg_type='sum'):
        super(SIRConvBase, self).__init__()
        self._agg_type = agg_type
        self._message_func = message_func
        self._agg_func = fn.sum if agg_type == 'sym' else getattr(fn, agg_type)
    
    def message_func(self, edges):
        message = torch.cat((edges.dst['eq'], edges.src['ek']), dim=-1)
        return {'m': edges.src['norm'] * edges.dst['norm'] * self._message_func(message)}
        
    def forward(self, graph, feat):
        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1).to(graph.device)
            norm = torch.pow(degs, -0.5) if self._agg_type == 'sym' else torch.ones(graph.num_nodes(), device=graph.device)
            norm = norm.reshape((graph.num_nodes(),) + (1,) * (feat.dim() - 1))
            graph.ndata['norm'] = norm
 
            feat_key, feat_query = expand_as_pair(feat, graph)
            graph.ndata['ek'] = feat_key
            graph.ndata['eq'] = feat_query

            graph.update_all(self.message_func, self._agg_func('m', 'ft'))
            rst = graph.ndata.pop('ft')
            
            return rst


class SIREConvBase(nn.Module):
    r"""Semi-Isomorphic Relational Graph Convolution Network (SIR-GCN) with Edge Features Base Class
    
    .. math::
        h_u^* = \sum_{v \in \mathcal{N}(u)} g([h_u \Vert h_{u,v} \Vert h_v])

    Parameters
    ----------
    message_func : callable layer
        The message function :math:`g` in the formula
    agg_type : str, optional
        Aggregator type to use (``sum``, ``max``, ``mean``, or ``sym``), defaults to ``sum``
    """
    def __init__(self, message_func, agg_type='sum'):
        super(SIREConvBase, self).__init__()
        self._agg_type = agg_type
        self._message_func = message_func
        self._agg_func = fn.sum if agg_type == 'sym' else getattr(fn, agg_type)
    
    def message_func(self, edges):
        message = torch.cat((edges.dst['eq'], edges.src['ek'], edges.data['e']), dim=-1)
        return {'m': edges.src['norm'] * edges.dst['norm'] * self._message_func(message)}
        
    def forward(self, graph, nfeat, efeat):
        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1).to(graph.device)
            norm = torch.pow(degs, -0.5) if self._agg_type == 'sym' else torch.ones(graph.num_nodes(), device=graph.device)
            norm = norm.reshape((graph.num_nodes(),) + (1,) * (nfeat.dim() - 1))
            graph.ndata['norm'] = norm
 
            nfeat_key, nfeat_query = expand_as_pair(nfeat, graph)
            graph.ndata['ek'] = nfeat_key
            graph.ndata['eq'] = nfeat_query
            graph.edata['e'] = efeat

            graph.update_all(self.message_func, self._agg_func('m', 'ft'))
            rst = graph.ndata.pop('ft')
            
            return rst
