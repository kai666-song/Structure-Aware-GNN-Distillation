"""
Advanced Topology-Aware Knowledge Distillation Losses

This module implements SOTA topology-aware distillation methods:
1. ContrastiveTopologyLoss: InfoNCE-based contrastive learning with graph structure
2. SoftTopologyLoss: Align student similarity with teacher attention weights
3. AttentionDistillationLoss: Direct attention weight transfer from GAT

Key Innovation:
- Move beyond naive "force neighbors to be similar"
- Learn teacher's attention distribution (soft topology)
- Use contrastive learning to distinguish neighbors from non-neighbors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContrastiveTopologyLoss(nn.Module):
    """
    Simplified Contrastive Topology Loss - More Stable Version
    
    Uses margin-based triplet loss instead of InfoNCE for better stability:
    L = max(0, margin + sim(anchor, negative) - sim(anchor, positive))
    
    This is more stable than InfoNCE and works better with KD.
    
    Args:
        temperature: Temperature for similarity scaling
        margin: Margin for triplet loss
        max_samples: Maximum edge samples per batch
    """
    def __init__(self, temperature=0.5, margin=0.3, max_samples=2048):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.max_samples = max_samples
        
    def forward(self, student_out, teacher_out, edge_index, mask=None):
        """
        Compute contrastive loss encouraging neighbors to be more similar
        than non-neighbors.
        """
        # Get normalized features from both student and teacher
        student_feat = F.normalize(F.softmax(student_out / self.temperature, dim=1), p=2, dim=1)
        teacher_feat = F.normalize(F.softmax(teacher_out / self.temperature, dim=1), p=2, dim=1)
        
        # Sample edges
        num_edges = edge_index.size(1)
        if num_edges > self.max_samples:
            perm = torch.randperm(num_edges, device=edge_index.device)[:self.max_samples]
            src, dst = edge_index[0, perm], edge_index[1, perm]
        else:
            src, dst = edge_index[0], edge_index[1]
        
        num_sampled = len(src)
        if num_sampled == 0:
            return torch.tensor(0.0, device=student_out.device)
        
        # Positive pairs: connected nodes (edges)
        # Student should match teacher's similarity for these pairs
        student_pos_sim = (student_feat[src] * student_feat[dst]).sum(dim=1)
        teacher_pos_sim = (teacher_feat[src] * teacher_feat[dst]).sum(dim=1)
        
        # Loss 1: Align student similarity with teacher similarity for edges
        loss_align = F.mse_loss(student_pos_sim, teacher_pos_sim.detach())
        
        # Loss 2: Margin loss - positive pairs should have higher similarity
        # Sample random negative pairs
        neg_dst = torch.randint(0, student_out.size(0), (num_sampled,), device=student_out.device)
        student_neg_sim = (student_feat[src] * student_feat[neg_dst]).sum(dim=1)
        
        # Triplet margin loss: sim(pos) > sim(neg) + margin
        loss_margin = F.relu(self.margin + student_neg_sim - student_pos_sim).mean()
        
        return loss_align + 0.5 * loss_margin
    
    def _build_adj_dict(self, edge_index, num_nodes):
        """Build adjacency dictionary for fast lookup."""
        adj_dict = {i: [] for i in range(num_nodes)}
        src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        for s, d in zip(src, dst):
            adj_dict[s].append(d)
        return adj_dict


class SoftTopologyLoss(nn.Module):
    """
    Soft Topology Distillation Loss
    
    Instead of using binary adjacency matrix, use teacher's attention weights
    as soft topology targets. Student should learn to produce similar
    pairwise similarities as teacher's attention distribution.
    
    L = MSE(StudentSim, TeacherAttn) for connected pairs
    
    Args:
        temperature: Temperature for similarity computation
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, student_out, teacher_attn, edge_index, mask=None):
        """
        Args:
            student_out: Student logits [N, C]
            teacher_attn: Teacher attention weights [E,] or [E, H] for multi-head
            edge_index: Edge index [2, E]
            mask: Training node mask
        """
        # Normalize student features
        student_feat = F.normalize(F.softmax(student_out / self.temperature, dim=1), p=2, dim=1)
        
        # Get edge endpoints
        src, dst = edge_index[0], edge_index[1]
        
        # Filter to training edges if mask provided
        if mask is not None:
            mask_set = set(mask.cpu().numpy().tolist())
            edge_mask = torch.tensor([
                (src[i].item() in mask_set) or (dst[i].item() in mask_set)
                for i in range(edge_index.size(1))
            ], device=edge_index.device)
            src, dst = src[edge_mask], dst[edge_mask]
            if teacher_attn.dim() == 1:
                teacher_attn = teacher_attn[edge_mask]
            else:
                teacher_attn = teacher_attn[edge_mask]
        
        if len(src) == 0:
            return torch.tensor(0.0, device=student_out.device)
        
        # Sample if too many edges
        num_edges = len(src)
        if num_edges > 4096:
            perm = torch.randperm(num_edges, device=src.device)[:4096]
            src, dst = src[perm], dst[perm]
            if teacher_attn.dim() == 1:
                teacher_attn = teacher_attn[perm]
            else:
                teacher_attn = teacher_attn[perm]
        
        # Compute student pairwise similarity for edges
        student_sim = (student_feat[src] * student_feat[dst]).sum(dim=1)  # [E]
        student_sim = (student_sim + 1) / 2  # Scale to [0, 1]
        
        # Normalize teacher attention to [0, 1]
        if teacher_attn.dim() > 1:
            teacher_attn = teacher_attn.mean(dim=-1)  # Average over heads
        teacher_attn = (teacher_attn - teacher_attn.min()) / (teacher_attn.max() - teacher_attn.min() + 1e-8)
        
        # MSE loss
        loss = F.mse_loss(student_sim, teacher_attn)
        
        return loss


class AttentionDistillationLoss(nn.Module):
    """
    Direct Attention Weight Distillation
    
    Transfer GAT's learned attention weights to MLP by making MLP's
    feature similarity distribution match GAT's attention distribution.
    
    For each node i, we want:
    softmax(MLP_sim(i, neighbors)) ≈ GAT_attention(i, neighbors)
    
    Uses KL divergence for distribution matching.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, student_out, teacher_attn_dict, edge_index, mask=None):
        """
        Args:
            student_out: Student logits [N, C]
            teacher_attn_dict: Dict mapping node -> {neighbor: attn_weight}
            edge_index: Edge index [2, E]
            mask: Training node mask
        """
        student_feat = F.normalize(F.softmax(student_out / self.temperature, dim=1), p=2, dim=1)
        
        if mask is not None:
            nodes = mask
        else:
            nodes = torch.arange(student_out.size(0), device=student_out.device)
        
        # Sample nodes if too many
        if len(nodes) > 256:
            perm = torch.randperm(len(nodes), device=nodes.device)[:256]
            nodes = nodes[perm]
        
        total_loss = 0.0
        valid_nodes = 0
        
        for node in nodes:
            node_idx = node.item()
            if node_idx not in teacher_attn_dict:
                continue
            
            neighbor_attn = teacher_attn_dict[node_idx]
            if len(neighbor_attn) < 2:
                continue
            
            neighbors = list(neighbor_attn.keys())
            teacher_weights = torch.tensor([neighbor_attn[n] for n in neighbors], 
                                          device=student_out.device)
            
            # Normalize teacher attention to probability distribution
            teacher_dist = F.softmax(teacher_weights / self.temperature, dim=0)
            
            # Compute student similarity to neighbors
            neighbor_indices = torch.tensor(neighbors, device=student_out.device)
            node_feat = student_feat[node_idx:node_idx+1]
            neighbor_feat = student_feat[neighbor_indices]
            student_sim = torch.mm(node_feat, neighbor_feat.t()).squeeze(0)
            student_dist = F.softmax(student_sim / self.temperature, dim=0)
            
            # KL divergence
            loss = F.kl_div(student_dist.log(), teacher_dist, reduction='sum')
            total_loss += loss
            valid_nodes += 1
        
        if valid_nodes == 0:
            return torch.tensor(0.0, device=student_out.device)
        
        return total_loss / valid_nodes


class HybridContrastiveLoss(nn.Module):
    """
    Hybrid Loss combining:
    1. Contrastive topology loss (margin-based, stable)
    2. Soft topology loss (attention alignment)
    3. Standard RKD loss (global relational)
    
    L = λ_con * L_contrastive + λ_soft * L_soft + λ_rkd * L_rkd
    
    Default weights tuned for stability with KD.
    """
    def __init__(self, lambda_con=0.1, lambda_soft=0.5, lambda_rkd=1.0,
                 temperature=0.5, max_samples=2048):
        super().__init__()
        self.lambda_con = lambda_con
        self.lambda_soft = lambda_soft
        self.lambda_rkd = lambda_rkd
        
        # Use stable margin-based contrastive loss
        self.contrastive_loss = ContrastiveTopologyLoss(temperature=temperature, margin=0.3)
        self.soft_loss = SoftTopologyLoss(temperature=1.0)
        self.max_samples = max_samples
        
    def forward(self, student_out, teacher_out, edge_index, 
                teacher_attn=None, mask=None):
        """
        Args:
            student_out: Student logits
            teacher_out: Teacher logits
            edge_index: Graph edges
            teacher_attn: Optional teacher attention weights
            mask: Training mask
        """
        losses = {}
        
        # Contrastive loss
        if self.lambda_con > 0:
            loss_con = self.contrastive_loss(student_out, teacher_out, edge_index, mask)
            losses['contrastive'] = self.lambda_con * loss_con
        
        # Soft topology loss (if attention provided)
        if self.lambda_soft > 0 and teacher_attn is not None:
            loss_soft = self.soft_loss(student_out, teacher_attn, edge_index, mask)
            losses['soft_topo'] = self.lambda_soft * loss_soft
        
        # RKD loss
        if self.lambda_rkd > 0:
            loss_rkd = self._rkd_loss(student_out, teacher_out, mask)
            losses['rkd'] = self.lambda_rkd * loss_rkd
        
        total_loss = sum(losses.values())
        return total_loss
    
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
        
        student_norm = F.normalize(student_out, p=2, dim=1)
        teacher_norm = F.normalize(teacher_out, p=2, dim=1)
        
        student_sim = torch.mm(student_norm, student_norm.t())
        teacher_sim = torch.mm(teacher_norm, teacher_norm.t())
        
        loss = F.mse_loss(student_sim, teacher_sim)
        return loss


# ============================================================================
# Graph Mixup for Data Augmentation
# ============================================================================

class GraphMixup:
    """
    Graph Mixup Augmentation
    
    Performs mixup in feature space while respecting graph structure.
    Key for improving MLP robustness and preventing overfitting.
    
    Reference: Graph-less Neural Networks, MixHop, etc.
    """
    def __init__(self, alpha=0.2, mode='input'):
        """
        Args:
            alpha: Beta distribution parameter for mixup ratio
            mode: 'input' for input space, 'latent' for latent space
        """
        self.alpha = alpha
        self.mode = mode
        
    def mixup_features(self, features, labels, edge_index=None):
        """
        Perform mixup on node features.
        
        Args:
            features: Node features [N, F]
            labels: Node labels [N]
            edge_index: Optional edge index for neighbor-aware mixup
            
        Returns:
            mixed_features, mixed_labels, lambda
        """
        batch_size = features.size(0)
        
        # Sample mixup ratio from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Random permutation for mixing partners
        if edge_index is not None:
            # Neighbor-aware mixup: prefer mixing with neighbors
            mix_indices = self._neighbor_aware_permutation(edge_index, batch_size, features.device)
        else:
            mix_indices = torch.randperm(batch_size, device=features.device)
        
        # Mix features
        mixed_features = lam * features + (1 - lam) * features[mix_indices]
        
        return mixed_features, mix_indices, lam
    
    def _neighbor_aware_permutation(self, edge_index, num_nodes, device):
        """Generate permutation preferring neighbors."""
        # Build adjacency
        adj_dict = {}
        src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        for s, d in zip(src, dst):
            if s not in adj_dict:
                adj_dict[s] = []
            adj_dict[s].append(d)
        
        # For each node, sample from neighbors with 50% prob, else random
        perm = []
        for i in range(num_nodes):
            if i in adj_dict and len(adj_dict[i]) > 0 and np.random.random() < 0.5:
                perm.append(np.random.choice(adj_dict[i]))
            else:
                perm.append(np.random.randint(0, num_nodes))
        
        return torch.tensor(perm, device=device)
    
    def mixup_loss(self, pred, labels_a, labels_b, lam, criterion):
        """
        Compute mixup loss.
        
        Args:
            pred: Model predictions
            labels_a: Original labels
            labels_b: Mixed partner labels
            lam: Mixup ratio
            criterion: Loss function
        """
        return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


# Keep backward compatibility
TopologyConsistencyLoss = ContrastiveTopologyLoss
AdaptiveTopologyLoss = ContrastiveTopologyLoss
HybridTopologyLoss = HybridContrastiveLoss
