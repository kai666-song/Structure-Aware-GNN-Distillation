"""
Spectral-Decoupled Adaptive Knowledge Distillation Loss
=======================================================

This is the CORE CONTRIBUTION of the paper.

Key Idea: Decompose Teacher's knowledge into low-frequency (smooth) and 
high-frequency (sharp) components, then adaptively weight them based on
each node's local homophily.

Mathematical Formulation:
-------------------------
1. Signal Decomposition:
   - Low-pass:  Z_low = D^{-1} A Z  (neighbor averaging)
   - High-pass: Z_high = Z - Z_low  (residual = local deviation)

2. Adaptive Gating:
   - For high-homophily nodes (h → 1): emphasize low-frequency consistency
   - For low-homophily nodes (h → 0): emphasize high-frequency consistency

3. Final Loss:
   L = h * L_low(T, S) + (1 - h) * L_high(T, S)

Reference:
- Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
- This work: Spectral decomposition for heterophilic graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


class SpectralDecomposer(nn.Module):
    """
    Decompose signals into low-frequency and high-frequency components
    using graph structure.
    
    Low-pass filter: Z_low = D^{-1} A Z (random walk normalization)
    High-pass filter: Z_high = Z - Z_low
    """
    
    def __init__(self, adj, device='cpu'):
        """
        Args:
            adj: scipy sparse adjacency matrix (N x N)
            device: torch device
        """
        super().__init__()
        self.device = device
        
        # Precompute normalized adjacency for low-pass filtering
        # P = D^{-1} A (random walk matrix)
        self.P = self._compute_random_walk_matrix(adj, device)
        
    def _compute_random_walk_matrix(self, adj, device):
        """Compute random walk matrix P = D^{-1} A"""
        if not sp.isspmatrix_csr(adj):
            adj = adj.tocsr()
        
        # Compute degree
        degree = np.array(adj.sum(axis=1)).flatten()
        degree[degree == 0] = 1  # Avoid division by zero
        
        # D^{-1}
        d_inv = 1.0 / degree
        d_inv_diag = sp.diags(d_inv, format='csr')
        
        # P = D^{-1} A
        P = d_inv_diag @ adj
        
        # Convert to torch sparse tensor
        P_coo = P.tocoo()
        indices = torch.LongTensor(np.vstack([P_coo.row, P_coo.col]))
        values = torch.FloatTensor(P_coo.data)
        P_sparse = torch.sparse_coo_tensor(indices, values, P_coo.shape)
        
        return P_sparse.to(device)
    
    def decompose(self, Z):
        """
        Decompose signal Z into low and high frequency components.
        
        Args:
            Z: Signal tensor (N, C) - can be logits or features
            
        Returns:
            Z_low: Low-frequency component (N, C)
            Z_high: High-frequency component (N, C)
        """
        # Low-pass: neighbor averaging
        # Z_low = P @ Z where P = D^{-1} A
        Z_low = torch.sparse.mm(self.P, Z)
        
        # High-pass: residual (local deviation from neighbors)
        Z_high = Z - Z_low
        
        return Z_low, Z_high


class AdaptiveSpectralKDLoss(nn.Module):
    """
    Adaptive Spectral Knowledge Distillation Loss.
    
    Decomposes teacher and student logits into spectral components,
    then applies homophily-weighted loss.
    
    L = Σ_i [ h_i * L_low(T_i, S_i) + (1 - h_i) * L_high(T_i, S_i) ]
    
    Args:
        adj: scipy sparse adjacency matrix
        homophily_weights: per-node homophily scores (N, 1), values in [0, 1]
        temperature: softmax temperature for KD
        alpha_low: weight for low-frequency loss (default: 1.0)
        alpha_high: weight for high-frequency loss (default: 1.0)
        high_freq_scale: scaling factor for high-freq loss to prevent vanishing
        device: torch device
    """
    
    def __init__(self, adj, homophily_weights, temperature=4.0,
                 alpha_low=1.0, alpha_high=1.0, high_freq_scale=2.0,
                 device='cpu'):
        super().__init__()
        
        self.decomposer = SpectralDecomposer(adj, device)
        self.T = temperature
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.high_freq_scale = high_freq_scale
        
        # Register homophily weights as buffer (not trainable)
        # h: (N, 1) -> (N,)
        h = homophily_weights.squeeze()
        if not isinstance(h, torch.Tensor):
            h = torch.tensor(h, dtype=torch.float32)
        self.register_buffer('homophily', h.to(device))
        
    def forward(self, logits_student, logits_teacher, mask=None):
        """
        Compute adaptive spectral KD loss.
        
        Args:
            logits_student: Student output logits (N, C) - PRE-SOFTMAX
            logits_teacher: Teacher output logits (N, C) - PRE-SOFTMAX
            mask: Optional boolean mask for which nodes to compute loss
            
        Returns:
            loss: Scalar loss value
            loss_dict: Dictionary with component losses for logging
        """
        # Decompose both teacher and student logits
        T_low, T_high = self.decomposer.decompose(logits_teacher)
        S_low, S_high = self.decomposer.decompose(logits_student)
        
        # Compute KL divergence for low-frequency components
        # Use temperature scaling
        loss_low = self._kl_div_loss(S_low, T_low, self.T)
        
        # Compute MSE for high-frequency components
        # High-freq captures local deviations, MSE is more appropriate
        loss_high = self._mse_loss(S_high, T_high) * self.high_freq_scale
        
        # Get homophily weights
        h = self.homophily  # (N,)
        
        # Apply mask if provided
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.bool, device=h.device)
            h = h[mask]
            loss_low = loss_low[mask]
            loss_high = loss_high[mask]
        
        # Adaptive weighting: h * low + (1-h) * high
        # High homophily -> emphasize low-freq (smooth)
        # Low homophily -> emphasize high-freq (sharp)
        weighted_loss = (
            self.alpha_low * h * loss_low + 
            self.alpha_high * (1 - h) * loss_high
        )
        
        # Mean over nodes
        loss = weighted_loss.mean()
        
        # Return component losses for logging
        loss_dict = {
            'loss_total': loss.item(),
            'loss_low': (h * loss_low).mean().item(),
            'loss_high': ((1 - h) * loss_high).mean().item(),
            'mean_homophily': h.mean().item()
        }
        
        return loss, loss_dict
    
    def _kl_div_loss(self, logits_s, logits_t, T):
        """
        KL divergence loss with temperature scaling.
        Returns per-node loss (N,).
        """
        p_s = F.log_softmax(logits_s / T, dim=1)
        p_t = F.softmax(logits_t / T, dim=1)
        
        # KL divergence per node: sum over classes
        kl = F.kl_div(p_s, p_t, reduction='none').sum(dim=1)
        
        # Scale by T^2 (standard KD practice)
        return kl * (T * T)
    
    def _mse_loss(self, logits_s, logits_t):
        """
        MSE loss for high-frequency components.
        Returns per-node loss (N,).
        """
        # MSE per node: mean over classes
        mse = ((logits_s - logits_t) ** 2).mean(dim=1)
        return mse


class HybridAdaptiveLoss(nn.Module):
    """
    Complete hybrid loss combining:
    1. Task loss (CrossEntropy with ground truth)
    2. Adaptive Spectral KD loss
    3. Optional: Standard soft target KD loss
    
    L_total = L_CE + λ_spectral * L_spectral + λ_soft * L_soft
    
    Args:
        adj: scipy sparse adjacency matrix
        homophily_weights: per-node homophily (N, 1)
        lambda_spectral: weight for spectral KD loss
        lambda_soft: weight for standard soft target loss
        temperature: KD temperature
        alpha_low: weight for low-freq in spectral loss
        alpha_high: weight for high-freq in spectral loss
        device: torch device
    """
    
    def __init__(self, adj, homophily_weights, 
                 lambda_spectral=1.0, lambda_soft=0.5,
                 temperature=4.0, alpha_low=1.0, alpha_high=1.0,
                 high_freq_scale=2.0, device='cpu'):
        super().__init__()
        
        self.lambda_spectral = lambda_spectral
        self.lambda_soft = lambda_soft
        self.T = temperature
        
        # Spectral KD loss
        self.spectral_loss = AdaptiveSpectralKDLoss(
            adj, homophily_weights, temperature,
            alpha_low, alpha_high, high_freq_scale, device
        )
        
        # Standard CE loss
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits_student, logits_teacher, labels, 
                train_mask=None, compute_all=True):
        """
        Compute hybrid loss.
        
        Args:
            logits_student: Student logits (N, C)
            logits_teacher: Teacher logits (N, C)
            labels: Ground truth labels (N,)
            train_mask: Boolean mask for training nodes
            compute_all: If True, compute spectral loss on all nodes
                        If False, only on train_mask nodes
        
        Returns:
            loss: Total loss
            loss_dict: Component losses for logging
        """
        loss_dict = {}
        
        # 1. Task loss (CE) - only on training nodes
        if train_mask is not None:
            if isinstance(train_mask, np.ndarray):
                train_mask_t = torch.tensor(train_mask, dtype=torch.bool, 
                                           device=logits_student.device)
            else:
                train_mask_t = train_mask
            loss_ce = self.ce_loss(logits_student[train_mask_t], labels[train_mask_t])
        else:
            loss_ce = self.ce_loss(logits_student, labels)
        loss_dict['loss_ce'] = loss_ce.item()
        
        # 2. Spectral KD loss
        spectral_mask = None if compute_all else train_mask
        loss_spectral, spectral_dict = self.spectral_loss(
            logits_student, logits_teacher, mask=spectral_mask
        )
        loss_dict.update({f'spectral_{k}': v for k, v in spectral_dict.items()})
        
        # 3. Standard soft target loss (optional)
        if self.lambda_soft > 0:
            loss_soft = self._soft_target_loss(logits_student, logits_teacher)
            loss_dict['loss_soft'] = loss_soft.item()
        else:
            loss_soft = 0
        
        # Total loss
        loss_total = (
            loss_ce + 
            self.lambda_spectral * loss_spectral +
            self.lambda_soft * loss_soft
        )
        loss_dict['loss_total'] = loss_total.item()
        
        return loss_total, loss_dict
    
    def _soft_target_loss(self, logits_s, logits_t):
        """Standard soft target KD loss."""
        p_s = F.log_softmax(logits_s / self.T, dim=1)
        p_t = F.softmax(logits_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T * self.T)
        return loss
