import dgl
import torch
import itertools
import numpy as np
from dgl.data import DGLDataset


# DictionaryLookup from Brody et al.
class DictionaryLookupDataset(DGLDataset):
    def __init__(self, num_nodes, num_samples=1000):
        super(DictionaryLookupDataset, self).__init__(name='DictionaryLookupDataset')
        self.num_nodes = num_nodes
        self.empty_id = num_nodes
        self.num_samples = num_samples

        self.data = []
        for perm in self._permutations():
            g = self.empty_graph
            g.ndata['feat'] = torch.tensor(self._node_feats(perm))
            g.ndata['mask'] = torch.tensor([True] * self.num_nodes + [False] * self.num_nodes)
            self.data.append(g)

    def _permutations(self):
        return [np.random.permutation(range(self.num_nodes)) for _ in range(self.num_samples)]
    
    @property
    def empty_graph(self):
        key = range(0, self.num_nodes)
        val = range(self.num_nodes, 2 * self.num_nodes)
        edges = tuple(list(i) for i in zip(*itertools.product(val, key)))
        return dgl.graph(edges)
    
    def _node_feats(self, perm):
        return ([(key, self.empty_id) for key in range(self.num_nodes)] + 
                [(key, val) for key, val in zip(range(self.num_nodes), perm)])
    
    def __getitem__(self, i):
        return self.data[i]
    
    def __len__(self):
        return self.num_samples
