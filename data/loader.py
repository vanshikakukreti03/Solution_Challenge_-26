"""
Elliptic Bitcoin Transaction Dataset Loader.
Loads via PyTorch Geometric or falls back to synthetic data for demo.
"""
import os
import torch
import numpy as np
from torch_geometric.data import Data

EGO_FEATURE_COUNT = 94
STRUCTURAL_FEATURE_COUNT = 72


class EllipticDataLoader:
    def __init__(self, root='./data/elliptic'):
        self.root = root
        self.data = None

    def load(self):
        try:
            return self._load_pyg()
        except Exception as e:
            print(f"[INFO] PyG Elliptic dataset unavailable ({e}). Using synthetic demo data.")
            return self._generate_synthetic()

    def _load_pyg(self):
        from torch_geometric.datasets import EllipticBitcoinDataset
        dataset = EllipticBitcoinDataset(root=self.root)
        data = dataset[0]

        if data.x.size(1) > EGO_FEATURE_COUNT + STRUCTURAL_FEATURE_COUNT:
            data.x = data.x[:, :EGO_FEATURE_COUNT + STRUCTURAL_FEATURE_COUNT]

        y = data.y.clone()
        unique = torch.unique(y)
        if 2 in unique:
            known_mask = (y == 1) | (y == 2)
            y[data.y == 2] = 0
        else:
            known_mask = y >= 0

        data.y = y
        data.known_mask = known_mask
        self._split(data, known_mask)
        data.x = self._normalize(data.x)
        self.data = data
        return data

    def _generate_synthetic(self):
        np.random.seed(42)
        torch.manual_seed(42)
        N, E = 5000, 12000
        F = EGO_FEATURE_COUNT + STRUCTURAL_FEATURE_COUNT
        x = torch.randn(N, F)

        y = torch.full((N,), -1, dtype=torch.long)
        labeled = torch.randperm(N)[:int(0.5 * N)]
        for idx in labeled:
            y[idx] = 1 if np.random.random() < 0.1 else 0

        illicit = (y == 1).nonzero(as_tuple=True)[0]
        x[illicit, :EGO_FEATURE_COUNT] += 0.5
        neigh_fraud = torch.rand(len(illicit)) > 0.4
        for i, node in enumerate(illicit):
            if neigh_fraud[i]:
                x[node, EGO_FEATURE_COUNT:] += 1.5

        srcs = torch.randint(0, N, (E,))
        dsts = torch.randint(0, N, (E,))
        edge_index = torch.stack([srcs, dsts], dim=0)

        known_mask = y >= 0
        data = Data(x=self._normalize(x), edge_index=edge_index, y=y, known_mask=known_mask)
        self._split(data, known_mask)
        self.data = data
        return data

    def _split(self, data, known_mask):
        idx = known_mask.nonzero(as_tuple=True)[0]
        n = idx.size(0)
        perm = torch.randperm(n)
        s = int(0.8 * n)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[idx[perm[:s]]] = True
        data.test_mask[idx[perm[s:]]] = True

    @staticmethod
    def _normalize(x):
        m = x.mean(dim=0)
        s = x.std(dim=0)
        s[s == 0] = 1.0
        return (x - m) / s
