import torch
from torch import nn
from dgl import function as fn
from dgl.utils import expand_as_pair


class SIRConv(nn.Module):
    r"""Semi-Isomorphic Relational Graph Convolution Network (SIR-GCN)
    
    .. math::
        h_u^* = \sum_{v \in \mathcal{N}(u)} W_R \cdot \sigma(W_Q h_u + W_K h_v)

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
        return {'m': edges.src['norm'] * edges.dst['norm'] * self.activation(edges.dst['eq'] + edges.src['ek'])}
    
    def forward(self, graph, feat):
        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1).to(graph.device)
            norm = torch.pow(degs, -0.5).unsqueeze(dim=1)
            norm = norm if self._agg_type == 'sym' else torch.ones((graph.num_nodes(), 1), device=norm.device)
            graph.ndata['norm'] = norm
 
            feat_key, feat_query = expand_as_pair(feat, graph)
            graph.ndata['ek'] = self.dropout(self.linear_key(feat_key))
            graph.ndata['eq'] = self.dropout(self.linear_query(feat_query))

            graph.update_all(self.message_func, self._agg_func('m', 'ft'))
            rst = graph.ndata.pop('ft')
            rst = self.linear_relation(rst)
            
            return rst
