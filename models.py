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


# ============================================================================
# GAT: Graph Attention Network (Stronger Teacher)
# ============================================================================

try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: torch_geometric not found. GAT model unavailable.")


class GAT(nn.Module):
    """Graph Attention Network (Stronger Teacher Model)
    
    Uses multi-head attention in first layer, single head in output layer.
    Typically achieves 1-2% higher accuracy than GCN.
    
    Can return attention weights for soft topology distillation.
    
    Args:
        nfeat: Input feature dimension
        nhid: Hidden dimension (per head)
        nclass: Number of classes
        dropout: Dropout rate
        heads: Number of attention heads in first layer
        
    Note: GAT uses edge_index format, not adjacency matrix.
    Use convert_adj_to_edge_index() to convert if needed.
    """
    def __init__(self, nfeat, nhid, nclass, dropout, heads=8):
        super(GAT, self).__init__()
        
        if not HAS_PYG:
            raise ImportError("torch_geometric required for GAT. Install with: pip install torch_geometric")
        
        self.dropout = dropout
        self.heads = heads
        
        # First GAT layer: multi-head attention (return attention weights)
        # Output: nhid * heads features
        self.gat1 = GATConv(nfeat, nhid, heads=heads, dropout=dropout)
        
        # Second GAT layer: single head for classification
        # Input: nhid * heads, Output: nclass
        self.gat2 = GATConv(nhid * heads, nclass, heads=1, concat=False, dropout=dropout)
        
    def forward(self, x, edge_index, return_attention=False):
        """
        Args:
            x: Node features [N, nfeat]
            edge_index: Edge index [2, E] in PyG format
            return_attention: If True, also return attention weights
            
        Returns:
            x: Output logits [N, nclass]
            attn_weights: (optional) Tuple of (attn1, attn2) attention weights
        """
        if x.is_sparse:
            x = x.to_dense()
        x = x.float()
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention:
            x, (edge_index1, attn1) = self.gat1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, (edge_index2, attn2) = self.gat2(x, edge_index, return_attention_weights=True)
            # attn1: [E, heads], attn2: [E, 1]
            return x, (attn1, attn2, edge_index1)
        else:
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gat2(x, edge_index)
            return x


class GATv2(nn.Module):
    """GATv2: Improved Graph Attention Network
    
    Uses GATv2Conv which fixes the static attention problem in original GAT.
    Generally more expressive and achieves better results.
    Can return attention weights for soft topology distillation.
    """
    def __init__(self, nfeat, nhid, nclass, dropout, heads=8):
        super(GATv2, self).__init__()
        
        if not HAS_PYG:
            raise ImportError("torch_geometric required for GATv2")
        
        from torch_geometric.nn import GATv2Conv
        
        self.dropout = dropout
        self.heads = heads
        self.gat1 = GATv2Conv(nfeat, nhid, heads=heads, dropout=dropout)
        self.gat2 = GATv2Conv(nhid * heads, nclass, heads=1, concat=False, dropout=dropout)
        
    def forward(self, x, edge_index, return_attention=False):
        if x.is_sparse:
            x = x.to_dense()
        x = x.float()
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention:
            x, (edge_index1, attn1) = self.gat1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, (edge_index2, attn2) = self.gat2(x, edge_index, return_attention_weights=True)
            return x, (attn1, attn2, edge_index1)
        else:
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gat2(x, edge_index)
            return x


def convert_adj_to_edge_index(adj):
    """Convert adjacency matrix to edge_index format for PyG models.
    
    Args:
        adj: Sparse or dense adjacency matrix [N, N]
        
    Returns:
        edge_index: [2, E] tensor of edge indices
    """
    import torch
    
    if adj.is_sparse:
        adj = adj.coalesce()
        edge_index = adj.indices()
    else:
        edge_index = adj.nonzero().t().contiguous()
    
    return edge_index
