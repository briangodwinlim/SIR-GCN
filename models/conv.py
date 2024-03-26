import torch
from torch import nn
from dgl import function as fn
from dgl.utils import expand_as_pair


class SIRConv(nn.Module):
    def __init__(self, mod_rel, mod_query=None, mod_key=None, mod_edge=None, mod_resid=None, mod_comb=None, agg_type='sum'):
        r"""Semi-Isomorphic Relational Graph Convolution Network (SIR-GCN)

        .. math::
            h_u^* = \sigma(\sum_{v \in N(u)} g_A(g_Q(h_u) + g_K(h_v) + g_E(h_{u,v}))).

        Parameters
        ----------
        mod_rel : a callable layer
            Apply this function to the summed node and edge features
            for feature transformation, the :math:`g_A` in the formula.
        mod_query : a callable layer, optional
            Apply this function to the node features of query nodes, 
            the :math:`g_Q` in the formula, defaults to Identity.
        mod_key : a callable layer, optional
            Apply this function to the node features of key nodes,
            the :math:`g_K` in the formula, defaults to Identity.
        mod_edge : a callable layer, optional
            Apply this function to the edge features, if applicable, 
            the :math:`g_E` in the formula, defaults to Identity.
        mod_resid : a callable layer, optional
            Apply this function to the node features of query nodes and add to 
            aggregated node features before calling ``mod_comb``, defaults to None.
        mod_comb : a callable layer, optional
            Apply this function to the aggregated node features, 
            the :math:`\sigma` in the formula, defaults to Identity.
        agg_type : str, optional
            Aggregator type to use (``sum``, ``max``, ``mean``, or ``sym``).
        """
        super(SIRConv, self).__init__()
        self.mod_rel = mod_rel if mod_rel else nn.Identity()
        self.mod_query = mod_query if mod_query else nn.Identity()
        self.mod_key = mod_key if mod_key else nn.Identity()
        self.mod_edge = mod_edge if mod_edge else nn.Identity()
        self.mod_resid = mod_resid
        self.mod_comb = mod_comb if mod_comb else nn.Identity()

        self._agg_type = agg_type
        self._agg_func = fn.sum if agg_type == 'sym' else getattr(fn, agg_type)
    
    def message_func(self, edges):
        return {'m': edges.src['norm'] * edges.dst['norm'] * self.mod_rel(edges.dst['eq'] + edges.src['ek'])}
    
    def message_func_edge(self, edges):
        return {'m': edges.src['norm'] * edges.dst['norm'] * self.mod_rel(edges.dst['eq'] + edges.src['ek'] + edges.data['e'])}
    
    def forward(self, graph, node_feat, edge_feat=None):
        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1).to(graph.device)
            norm = torch.pow(degs, -0.5).unsqueeze(dim=1)
            norm = norm if self._agg_type == 'sym' else torch.ones((graph.num_nodes(), 1), device=norm.device)
 
            feat_key, feat_query = expand_as_pair(node_feat, graph)
            graph.ndata.update({'eq': self.mod_query(feat_query), 'ek': self.mod_key(feat_key), 'norm': norm})
            message_func = self.message_func

            if edge_feat is not None:
                graph.edata.update({'e': self.mod_edge(edge_feat)})
                message_func = self.message_func_edge

            graph.update_all(message_func, self._agg_func('m', 'ft'))
            rst = graph.ndata.pop('ft')

            if self.mod_resid is not None:
                rst = rst + self.mod_resid(feat_query)

            rst = self.mod_comb(rst)
            return rst
