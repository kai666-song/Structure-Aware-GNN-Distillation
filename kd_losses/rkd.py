"""
Relational Knowledge Distillation (RKD) Loss Module

This module implements structure-aware distillation by forcing the student
to mimic the pairwise relationships between samples learned by the teacher.

Reference: Park et al. "Relational Knowledge Distillation" (CVPR 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RKDLoss(nn.Module):
    """
    Relational Knowledge Distillation Loss
    
    Forces student to learn the same pairwise sample relationships as teacher.
    Uses cosine similarity to compute relation matrices.
    """
    
    def __init__(self, eps=1e-8):
        super(RKDLoss, self).__init__()
        self.eps = eps
    
    def forward(self, student_features, teacher_features, sample_indices=None):
        """
        Compute RKD loss between student and teacher features.
        
        Args:
            student_features: Student output features [N, D]
            teacher_features: Teacher output features [N, D] (will be detached)
            sample_indices: Optional indices for subsampling (memory efficiency)
        
        Returns:
            RKD loss value
        """
        # Detach teacher features (no gradient flow to teacher)
        teacher_features = teacher_features.detach()
        
        # Subsample if indices provided (for memory efficiency)
        if sample_indices is not None:
            student_features = student_features[sample_indices]
            teacher_features = teacher_features[sample_indices]
        
        # Compute pairwise cosine similarity matrices
        student_relations = self._compute_relations(student_features)
        teacher_relations = self._compute_relations(teacher_features)
        
        # Compute MSE loss between relation matrices
        loss = F.mse_loss(student_relations, teacher_relations)
        
        return loss
    
    def _compute_relations(self, features):
        """
        Compute normalized pairwise cosine similarity matrix.
        
        Args:
            features: Feature matrix [N, D]
        
        Returns:
            Normalized relation matrix [N, N]
        """
        # L2 normalize features (for cosine similarity)
        features_norm = F.normalize(features, p=2, dim=1, eps=self.eps)
        
        # Compute cosine similarity matrix: [N, N]
        relations = torch.mm(features_norm, features_norm.t())
        
        # Row-wise L2 normalization (normalize the relation distribution)
        relations = F.normalize(relations, p=2, dim=1, eps=self.eps)
        
        return relations


class RKDDistanceLoss(nn.Module):
    """
    RKD Distance Loss - uses Euclidean distance instead of cosine similarity.
    
    Computes pairwise distance relationships and aligns them.
    """
    
    def __init__(self, eps=1e-8):
        super(RKDDistanceLoss, self).__init__()
        self.eps = eps
    
    def forward(self, student_features, teacher_features, sample_indices=None):
        """
        Compute RKD distance loss.
        """
        teacher_features = teacher_features.detach()
        
        if sample_indices is not None:
            student_features = student_features[sample_indices]
            teacher_features = teacher_features[sample_indices]
        
        # Compute pairwise distance matrices
        student_dist = self._compute_distances(student_features)
        teacher_dist = self._compute_distances(teacher_features)
        
        # Normalize distances (mean normalization)
        student_dist = student_dist / (student_dist.mean() + self.eps)
        teacher_dist = teacher_dist / (teacher_dist.mean() + self.eps)
        
        # Huber loss for robustness
        loss = F.smooth_l1_loss(student_dist, teacher_dist)
        
        return loss
    
    def _compute_distances(self, features):
        """
        Compute pairwise Euclidean distance matrix.
        """
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        sq_norm = (features ** 2).sum(dim=1, keepdim=True)
        distances = sq_norm + sq_norm.t() - 2 * torch.mm(features, features.t())
        distances = torch.clamp(distances, min=0)  # Numerical stability
        distances = torch.sqrt(distances + self.eps)
        return distances


class AdaptiveRKDLoss(nn.Module):
    """
    Adaptive RKD Loss with automatic subsampling for large graphs.
    
    Automatically handles memory constraints by sampling nodes when
    the full relation matrix would be too large.
    """
    
    def __init__(self, max_samples=2048, eps=1e-8):
        """
        Args:
            max_samples: Maximum number of samples for relation computation.
                        If N > max_samples, random sampling is applied.
            eps: Small constant for numerical stability.
        """
        super(AdaptiveRKDLoss, self).__init__()
        self.max_samples = max_samples
        self.eps = eps
        self.rkd = RKDLoss(eps=eps)
    
    def forward(self, student_features, teacher_features, mask=None):
        """
        Compute adaptive RKD loss with automatic subsampling.
        
        Args:
            student_features: Student output features [N, D]
            teacher_features: Teacher output features [N, D]
            mask: Optional boolean mask to select specific nodes (e.g., train_mask)
        
        Returns:
            RKD loss value
        """
        # Apply mask if provided
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                indices = mask.nonzero(as_tuple=True)[0]
            else:
                indices = torch.tensor(mask, device=student_features.device)
            student_features = student_features[indices]
            teacher_features = teacher_features[indices]
        
        N = student_features.size(0)
        
        # Subsample if too many nodes
        if N > self.max_samples:
            sample_indices = torch.randperm(N, device=student_features.device)[:self.max_samples]
            return self.rkd(student_features, teacher_features, sample_indices)
        else:
            return self.rkd(student_features, teacher_features)
