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


# ============================================================================
# GCNII: Deep GCN with Initial Residual and Identity Mapping
# Designed for heterophilic graphs - stronger teacher for Actor/Squirrel
# ============================================================================

class GCNII(nn.Module):
    """
    GCNII: Simple and Deep Graph Convolutional Networks
    
    Reference: Chen et al. "Simple and Deep Graph Convolutional Networks" (ICML 2020)
    
    Key innovations:
    1. Initial residual connection: H^(l+1) = ((1-α)H^(l) + αH^(0))W
    2. Identity mapping: Prevents over-smoothing in deep networks
    
    Works well on heterophilic graphs where standard GCN/GAT fail.
    
    Args:
        nfeat: Input feature dimension
        nhid: Hidden dimension
        nclass: Number of classes
        dropout: Dropout rate
        num_layers: Number of GCNII layers (can be deep, e.g., 64)
        alpha: Initial residual weight (typically 0.1-0.5)
        theta: Identity mapping weight (typically 0.5-1.0)
    """
    def __init__(self, nfeat, nhid, nclass, dropout, num_layers=8, alpha=0.1, theta=0.5):
        super(GCNII, self).__init__()
        
        if not HAS_PYG:
            raise ImportError("torch_geometric required for GCNII")
        
        from torch_geometric.nn import GCN2Conv
        
        self.dropout = dropout
        self.num_layers = num_layers
        self.alpha = alpha
        self.theta = theta
        
        # Initial projection
        self.fc_in = nn.Linear(nfeat, nhid)
        self.bn_in = nn.BatchNorm1d(nhid)
        
        # GCNII layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCN2Conv(nhid, alpha=alpha, theta=theta, layer=i+1))
            self.bns.append(nn.BatchNorm1d(nhid))
        
        # Output projection
        self.fc_out = nn.Linear(nhid, nclass)
        
    def forward(self, x, edge_index):
        if x.is_sparse:
            x = x.to_dense()
        x = x.float()
        
        # Initial projection
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc_in(x)
        x = self.bn_in(x)
        x = F.relu(x)
        
        x_0 = x  # Save initial representation for residual
        
        # GCNII layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, x_0, edge_index)
            x = bn(x)
            x = F.relu(x)
        
        # Output
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc_out(x)
        
        return x


# ============================================================================
# GPR-GNN: Generalized PageRank GNN (Adaptive for Heterophilic Graphs)
# ============================================================================

class GPRGNN(nn.Module):
    """
    GPR-GNN: Generalized PageRank Graph Neural Network
    
    Reference: Chien et al. "Adaptive Universal Generalized PageRank Graph Neural Network" (ICLR 2021)
    
    Key innovation: Learns adaptive weights for different propagation steps,
    allowing negative weights to handle heterophilic graphs.
    
    H = sum_{k=0}^{K} gamma_k * A^k * X * W
    
    where gamma_k are learnable (can be negative for heterophily).
    
    Args:
        nfeat: Input feature dimension
        nhid: Hidden dimension
        nclass: Number of classes
        dropout: Dropout rate
        K: Number of propagation steps
        alpha: Initial PageRank teleport probability
    """
    def __init__(self, nfeat, nhid, nclass, dropout, K=10, alpha=0.1):
        super(GPRGNN, self).__init__()
        
        self.dropout = dropout
        self.K = K
        
        # Feature transformation
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        
        # Learnable GPR weights (initialized with PPR)
        # gamma_k = alpha * (1-alpha)^k
        init_gamma = alpha * (1 - alpha) ** torch.arange(K + 1).float()
        self.gamma = nn.Parameter(init_gamma)
        
    def forward(self, x, edge_index):
        import torch
        from torch_sparse import SparseTensor
        
        if x.is_sparse:
            x = x.to_dense()
        x = x.float()
        
        num_nodes = x.size(0)
        
        # Feature transformation
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.fc2(x)
        
        # Build normalized adjacency
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Propagation with learnable weights
        out = self.gamma[0] * h
        h_k = h
        
        for k in range(1, self.K + 1):
            # Sparse matrix multiplication: A * h_k
            # Using scatter for efficiency
            h_k_new = torch.zeros_like(h_k)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            h_k_new.scatter_add_(0, col.unsqueeze(1).expand(-1, h_k.size(1)), 
                                  h_k[row] * norm.unsqueeze(1))
            h_k = h_k_new
            out = out + self.gamma[k] * h_k
        
        return out


# Simple wrapper to use GCNII with adjacency matrix format
class GCNIIWrapper(nn.Module):
    """Wrapper for GCNII that accepts adjacency matrix and converts to edge_index."""
    def __init__(self, nfeat, nhid, nclass, dropout, num_layers=8, alpha=0.1, theta=0.5):
        super().__init__()
        self.gcnii = GCNII(nfeat, nhid, nclass, dropout, num_layers, alpha, theta)
        
    def forward(self, x, adj_or_edge_index):
        # Check if input is edge_index (2D with shape [2, E]) or adjacency matrix
        if adj_or_edge_index.dim() == 2 and adj_or_edge_index.size(0) == 2:
            edge_index = adj_or_edge_index
        else:
            edge_index = convert_adj_to_edge_index(adj_or_edge_index)
        return self.gcnii(x, edge_index)


# ============================================================================
# Enhanced MLP for Structure-Aware Knowledge Distillation (Phase 3)
# ============================================================================

class EnhancedMLP(nn.Module):
    """
    Structure-Enhanced MLP Student for Spectral KD.
    
    Key innovations:
    1. Feature Fusion Layer: LayerNorm to handle heterogeneous features
       (sparse semantic features + dense positional encoding)
    2. Residual Connections: Preserve high-frequency information
    3. Deeper Architecture: 3-4 layers to digest structural information
    
    Architecture:
        Input (X || PE) -> LayerNorm -> Linear -> LayerNorm -> ReLU -> Dropout
        -> [Residual Block] x (num_layers - 2)
        -> Linear -> Output
    
    Args:
        nfeat: Input feature dimension (original features + PE)
        nhid: Hidden dimension
        nclass: Number of output classes
        dropout: Dropout rate
        num_layers: Total number of layers (default: 3)
        use_residual: Whether to use residual connections (default: True)
        norm_type: 'layer' or 'batch' normalization (default: 'layer')
    """
    
    def __init__(self, nfeat, nhid, nclass, dropout, 
                 num_layers=3, use_residual=True, norm_type='layer'):
        super(EnhancedMLP, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.norm_type = norm_type
        
        # Input fusion layer with normalization
        # Critical for handling heterogeneous features (sparse + dense)
        self.input_norm = self._get_norm(nfeat, norm_type)
        self.fc_in = nn.Linear(nfeat, nhid)
        self.norm_in = self._get_norm(nhid, norm_type)
        
        # Residual blocks for hidden layers
        self.res_blocks = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.res_blocks.append(
                ResidualBlock(nhid, dropout, norm_type)
            )
        
        # Output layer
        self.fc_out = nn.Linear(nhid, nclass)
        
        # Initialize weights
        self._init_weights()
        
    def _get_norm(self, dim, norm_type):
        """Get normalization layer."""
        if norm_type == 'layer':
            return nn.LayerNorm(dim)
        elif norm_type == 'batch':
            return nn.BatchNorm1d(dim)
        else:
            return nn.Identity()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, adj=None):
        """
        Forward pass.
        
        Args:
            x: Input features (N, nfeat) - can include PE
            adj: Adjacency matrix (unused, for interface compatibility)
        
        Returns:
            logits: Output logits (N, nclass)
        """
        # Handle sparse input
        if x.is_sparse:
            x = x.to_dense()
        x = x.float()
        
        # Input normalization (critical for feature fusion)
        x = self.input_norm(x)
        
        # Input projection
        x = self.fc_in(x)
        x = self.norm_in(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Output
        logits = self.fc_out(x)
        
        return logits
    
    def get_features(self, x, adj=None):
        """
        Get intermediate features (before output layer).
        Useful for feature analysis and visualization.
        """
        if x.is_sparse:
            x = x.to_dense()
        x = x.float()
        
        x = self.input_norm(x)
        x = self.fc_in(x)
        x = self.norm_in(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for EnhancedMLP.
    
    x_out = x + MLP(x)
    
    This preserves high-frequency information through skip connections,
    which is critical for heterophilic graphs.
    """
    
    def __init__(self, dim, dropout, norm_type='layer'):
        super(ResidualBlock, self).__init__()
        
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = dropout
        
        if norm_type == 'layer':
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        elif norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
    
    def forward(self, x):
        """
        Forward with residual connection.
        """
        residual = x
        
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.fc2(x)
        x = self.norm2(x)
        
        # Residual connection
        x = x + residual
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        return x


class ResMLP(nn.Module):
    """
    Simple Residual MLP - Lightweight version.
    
    A simpler alternative to EnhancedMLP with direct residual connections.
    
    Architecture:
        x -> Linear -> Norm -> ReLU -> Dropout -> Linear -> (+x) -> Output
    
    Args:
        nfeat: Input feature dimension
        nhid: Hidden dimension
        nclass: Number of classes
        dropout: Dropout rate
        num_layers: Number of residual layers
    """
    
    def __init__(self, nfeat, nhid, nclass, dropout, num_layers=2):
        super(ResMLP, self).__init__()
        
        self.dropout = dropout
        
        # Input projection (handles dimension mismatch)
        self.fc_in = nn.Linear(nfeat, nhid)
        self.ln_in = nn.LayerNorm(nhid)
        
        # Residual layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(nhid, nhid))
            self.norms.append(nn.LayerNorm(nhid))
        
        # Output
        self.fc_out = nn.Linear(nhid, nclass)
    
    def forward(self, x, adj=None):
        if x.is_sparse:
            x = x.to_dense()
        x = x.float()
        
        # Input projection
        x = self.fc_in(x)
        x = self.ln_in(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Residual layers
        for fc, ln in zip(self.layers, self.norms):
            residual = x
            x = fc(x)
            x = ln(x)
            x = F.relu(x)
            x = x + residual  # Residual connection
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Output
        return self.fc_out(x)
