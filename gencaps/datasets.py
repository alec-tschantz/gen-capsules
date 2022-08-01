import networkx as nx
from torch.utils import data


class VectorSceneDataset(data.Dataset):
    def __init__(self, nodes, edges):
        self.graph = nx.DiGraph()
        self.init_graph(nodes, edges)
        self.data = []

    def init_graph(self, nodes, edges):
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

    def root_nodes(self):
        return [n for n, d in self.graph.in_degree() if d == 0]

    def leaf_nodes(self):
        return [n for n, d in self.graph.out_degree() if d == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
