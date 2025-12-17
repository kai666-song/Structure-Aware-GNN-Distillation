"""
Topology Consistency Distillation (TCD) Loss

Unlike vanilla RKD which ignores graph structure, TCD explicitly aligns
student's feature similarity with the graph topology (adjacency matrix).

Key Innovation:
- Only compute loss for node pairs that are connected (have edges)
- Force MLP to learn: "if nodes i,j are neighbors, their features should be similar"
- This transfers topological knowledge from GNN to MLP without requiring graph at inference

Reference: Structure-Aware Knowledge Distillation for Graph Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyConsistencyLoss(nn.Module):
    """
    Topology Consistency Distillation Loss
    
    Aligns student feature similarity with graph adjacency structure.
    Only penalizes dissimilarity for connected node pairs.
    
    Args:
        pos_weight: Weight for positive (connected) pairs to handle class imbalance
        temperature: Temperature for similarity scaling
        use_teacher_sim: If True, align with teacher similarity; if False, align with adjacency
    """
    def __init__(self, pos_weight=1.0, temperature=1.0, use_teacher_sim=True):
        super().__init__()
        self.pos_weight = pos_weight
        self.temperature = temperature
        self.use_teacher_sim = use_teacher_sim
        
    def forward(self, student_out, teacher_out, adj, mask=None):
        """
        Args:
            student_out: Student logits [N, C]
            teacher_out: Teacher logits [N, C]
            adj: Adjacency matrix (sparse or dense) [N, N]
            mask: Optional node mask for training nodes
        """
        # Get features (use softmax probabilities as features)
        student_feat = F.softmax(student_out / self.temperature, dim=1)
        teacher_feat = F.softmax(teacher_out / self.temperature, dim=1)
        
        # If mask provided, only use those nodes
        if mask is not None:
            student_feat = student_feat[mask]
            teacher_feat = teacher_feat[mask]
            # Extract subgraph adjacency
            if adj.is_sparse:
                adj = adj.to_dense()
            adj = adj[mask][:, mask]
        else:
            if adj.is_sparse:
                adj = adj.to_dense()
        
        # Compute cosine similarity matrices
        student_sim = self._cosine_similarity_matrix(student_feat)
        teacher_sim = self._cosine_similarity_matrix(teacher_feat)
        
        # Get edge mask (where adjacency > 0)
        edge_mask = (adj > 0).float()
        num_edges = edge_mask.sum()
        
        if num_edges == 0:
            return torch.tensor(0.0, device=student_out.device)
        
        if self.use_teacher_sim:
            # Align student similarity with teacher similarity (for connected pairs)
            diff = (student_sim - teacher_sim) ** 2
            loss = (diff * edge_mask * self.pos_weight).sum() / num_edges
        else:
            # Directly maximize similarity for connected pairs
            # Loss = 1 - similarity for connected pairs
            loss = ((1 - student_sim) * edge_mask * self.pos_weight).sum() / num_edges
        
        return loss
    
    def _cosine_similarity_matrix(self, feat):
        """Compute pairwise cosine similarity matrix."""
        feat_norm = F.normalize(feat, p=2, dim=1)
        sim = torch.mm(feat_norm, feat_norm.t())
        return sim


class AdaptiveTopologyLoss(nn.Module):
    """
    Memory-efficient Topology Loss with sampling for large graphs.
    
    For graphs with >10k nodes, computing full N×N similarity is expensive.
    This version samples edges and computes loss on sampled pairs.
    """
    def __init__(self, max_edges=4096, pos_weight=1.0, temperature=1.0):
        super().__init__()
        self.max_edges = max_edges
        self.pos_weight = pos_weight
        self.temperature = temperature
        
    def forward(self, student_out, teacher_out, edge_index, mask=None):
        """
        Args:
            student_out: Student logits [N, C]
            teacher_out: Teacher logits [N, C]
            edge_index: Edge index tensor [2, E] (PyG format)
            mask: Optional training node mask
        """
        # Get features
        student_feat = F.softmax(student_out / self.temperature, dim=1)
        teacher_feat = F.softmax(teacher_out / self.temperature, dim=1)
        
        # Filter edges to only include training nodes if mask provided
        if mask is not None:
            mask_set = set(mask.cpu().numpy().tolist()) if isinstance(mask, torch.Tensor) else set(mask)
            edge_mask = torch.tensor([
                (edge_index[0, i].item() in mask_set) and (edge_index[1, i].item() in mask_set)
                for i in range(edge_index.size(1))
            ], device=edge_index.device)
            edge_index = edge_index[:, edge_mask]
        
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return torch.tensor(0.0, device=student_out.device)
        
        # Sample edges if too many
        if num_edges > self.max_edges:
            perm = torch.randperm(num_edges, device=edge_index.device)[:self.max_edges]
            edge_index = edge_index[:, perm]
            num_edges = self.max_edges
        
        # Get source and target node features
        src, dst = edge_index[0], edge_index[1]
        
        student_src = F.normalize(student_feat[src], p=2, dim=1)
        student_dst = F.normalize(student_feat[dst], p=2, dim=1)
        teacher_src = F.normalize(teacher_feat[src], p=2, dim=1)
        teacher_dst = F.normalize(teacher_feat[dst], p=2, dim=1)
        
        # Compute pairwise similarity for edges
        student_sim = (student_src * student_dst).sum(dim=1)
        teacher_sim = (teacher_src * teacher_dst).sum(dim=1)
        
        # MSE loss between student and teacher similarity
        loss = F.mse_loss(student_sim, teacher_sim)
        
        return loss * self.pos_weight


class HybridTopologyLoss(nn.Module):
    """
    Combines RKD-style relational loss with topology-aware masking.
    
    L_hybrid = λ_rkd * L_rkd + λ_topo * L_topo
    
    This provides both:
    1. Global relational knowledge (RKD on all pairs)
    2. Local topological consistency (TCD on connected pairs)
    """
    def __init__(self, lambda_rkd=1.0, lambda_topo=1.0, max_samples=2048):
        super().__init__()
        self.lambda_rkd = lambda_rkd
        self.lambda_topo = lambda_topo
        self.max_samples = max_samples
        
    def forward(self, student_out, teacher_out, adj=None, edge_index=None, mask=None):
        """
        Args:
            student_out, teacher_out: Model outputs
            adj: Adjacency matrix (for dense computation)
            edge_index: Edge index (for sparse computation)
            mask: Training node mask
        """
        # RKD component (global relational)
        loss_rkd = self._rkd_loss(student_out, teacher_out, mask)
        
        # Topology component (local structural)
        if edge_index is not None:
            loss_topo = self._topo_loss_sparse(student_out, teacher_out, edge_index, mask)
        elif adj is not None:
            loss_topo = self._topo_loss_dense(student_out, teacher_out, adj, mask)
        else:
            loss_topo = torch.tensor(0.0, device=student_out.device)
        
        return self.lambda_rkd * loss_rkd + self.lambda_topo * loss_topo
    
    def _rkd_loss(self, student_out, teacher_out, mask=None):
        """Standard RKD distance loss."""
        if mask is not None:
            student_out = student_out[mask]
            teacher_out = teacher_out[mask]
        
        n = student_out.size(0)
        if n > self.max_samples:
            perm = torch.randperm(n, device=student_out.device)[:self.max_samples]
            student_out = student_out[perm]
            teacher_out = teacher_out[perm]
        
        # Normalize
        student_norm = F.normalize(student_out, p=2, dim=1)
        teacher_norm = F.normalize(teacher_out, p=2, dim=1)
        
        # Similarity matrices
        student_sim = torch.mm(student_norm, student_norm.t())
        teacher_sim = torch.mm(teacher_norm, teacher_norm.t())
        
        # MSE loss
        loss = F.mse_loss(student_sim, teacher_sim)
        return loss
    
    def _topo_loss_dense(self, student_out, teacher_out, adj, mask=None):
        """Topology loss with dense adjacency."""
        student_feat = F.softmax(student_out, dim=1)
        teacher_feat = F.softmax(teacher_out, dim=1)
        
        if mask is not None:
            student_feat = student_feat[mask]
            teacher_feat = teacher_feat[mask]
            if adj.is_sparse:
                adj = adj.to_dense()
            adj = adj[mask][:, mask]
        else:
            if adj.is_sparse:
                adj = adj.to_dense()
        
        # Compute similarities
        student_norm = F.normalize(student_feat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_feat, p=2, dim=1)
        student_sim = torch.mm(student_norm, student_norm.t())
        teacher_sim = torch.mm(teacher_norm, teacher_norm.t())
        
        # Masked loss (only for connected pairs)
        edge_mask = (adj > 0).float()
        num_edges = edge_mask.sum()
        
        if num_edges == 0:
            return torch.tensor(0.0, device=student_out.device)
        
        diff = (student_sim - teacher_sim) ** 2
        loss = (diff * edge_mask).sum() / num_edges
        return loss
    
    def _topo_loss_sparse(self, student_out, teacher_out, edge_index, mask=None):
        """Topology loss with sparse edge_index."""
        student_feat = F.softmax(student_out, dim=1)
        teacher_feat = F.softmax(teacher_out, dim=1)
        
        src, dst = edge_index[0], edge_index[1]
        
        # Sample if too many edges
        num_edges = edge_index.size(1)
        if num_edges > self.max_samples * 2:
            perm = torch.randperm(num_edges, device=edge_index.device)[:self.max_samples * 2]
            src, dst = src[perm], dst[perm]
        
        # Compute similarity for edge pairs
        student_norm = F.normalize(student_feat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_feat, p=2, dim=1)
        
        student_sim = (student_norm[src] * student_norm[dst]).sum(dim=1)
        teacher_sim = (teacher_norm[src] * teacher_norm[dst]).sum(dim=1)
        
        loss = F.mse_loss(student_sim, teacher_sim)
        return loss
