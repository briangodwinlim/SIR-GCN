import dgl
import torch
import numpy as np
from dgl.data import DGLDataset


# Synthetic HeteroEdgeCount dataset for graph regression task
class HeteroEdgeCountDataset(DGLDataset):
    def __init__(self, max_nodes, num_classes, num_samples=1000, normalize=True):
        super(HeteroEdgeCountDataset, self).__init__(name='HeteroEdgeCountDataset')
        self.max_nodes = max_nodes
        self.num_classes = num_classes
        self.num_samples = num_samples

        self.graphs = []
        self.targets = []
        for _ in range(num_samples):
            g = self.empty_graph
            g.ndata['label'] = torch.randint(0, num_classes, (g.num_nodes(),))
            denom = g.number_of_edges() if normalize else 1.0
            target = (g.ndata['label'][g.edges()[0]] != g.ndata['label'][g.edges()[1]]).sum() / denom
            self.graphs.append(g)
            self.targets.append(target)

    @property
    def empty_graph(self):
        num_nodes = np.random.randint(2, self.max_nodes + 1)
        num_edges = np.random.randint(num_nodes * num_nodes // 4, num_nodes * num_nodes + 1)
        graph = dgl.rand_graph(num_nodes, num_edges)
        return graph
    
    def __getitem__(self, i):
        return self.graphs[i], self.targets[i].unsqueeze(0)
    
    def __len__(self):
        return self.num_samples
