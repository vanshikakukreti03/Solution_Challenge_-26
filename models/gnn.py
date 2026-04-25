"""
FraudGCN: Graph Convolutional Network for fraud detection on the Elliptic dataset.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class FraudGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, out_channels=2, layers=3, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GCNConv(in_channels, hidden))
        self.bns.append(BatchNorm(hidden))
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(BatchNorm(hidden))
        self.convs.append(GCNConv(hidden, out_channels))

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def predict_proba(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            return F.softmax(self.forward(x, edge_index), dim=1)
