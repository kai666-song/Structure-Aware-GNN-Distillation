import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    """Graph Convolutional Network (Teacher Model)"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron (Student Model) - No graph structure used
    
    Basic 2-layer MLP for backward compatibility.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj=None):
        # MLP does NOT use adj (graph structure)
        # adj parameter kept for interface compatibility
        if x.is_sparse:
            x = x.to_dense()
        x = x.float()  # Ensure float32 dtype
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class MLPBatchNorm(nn.Module):
    """Enhanced MLP with BatchNorm - Better convergence for distillation
    
    Architecture: Linear -> BatchNorm -> ReLU -> Dropout -> Linear
    BatchNorm is critical for MLP convergence in KD settings.
    """
    def __init__(self, nfeat, nhid, nclass, dropout, num_layers=2):
        super(MLPBatchNorm, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.fc_in = nn.Linear(nfeat, nhid)
        self.bn_in = nn.BatchNorm1d(nhid)
        
        # Hidden layers (if num_layers > 2)
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(nhid, nhid))
            self.hidden_bns.append(nn.BatchNorm1d(nhid))
        
        # Output layer
        self.fc_out = nn.Linear(nhid, nclass)
        
    def forward(self, x, adj=None):
        # MLP does NOT use adj
        if x.is_sparse:
            x = x.to_dense()
        x = x.float()
        
        # Input layer with BatchNorm
        x = self.fc_in(x)
        x = self.bn_in(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Hidden layers
        for fc, bn in zip(self.hidden_layers, self.hidden_bns):
            x = fc(x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Output layer
        x = self.fc_out(x)
        return x