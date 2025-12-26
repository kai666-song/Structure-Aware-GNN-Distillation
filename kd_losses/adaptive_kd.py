"""
Adaptive Frequency-Decoupled Knowledge Distillation (AFD-KD)
============================================================

UPGRADED VERSION: From static low/high-pass to learnable Bernstein polynomial
spectral filtering.

Core Innovation:
----------------
1. Bernstein Polynomial Basis Functions:
   B_{k,K}(x) = C(K,k) * x^k * (1-x)^{K-k}
   
   These polynomials form a partition of unity on [0,1] and provide stable,
   smooth basis functions for spectral filtering.

2. Learnable Spectral Filter:
   h(L) = Σ_{k=0}^{K} θ_k * B_{k,K}(L̃)
   
   where L̃ = L/2 is the scaled normalized Laplacian (eigenvalues in [0,1]),
   and θ_k are learnable parameters optimized via backpropagation.

3. Frequency-Decoupled Distillation:
   Z_filtered = h(L) @ Z
   L_AFD = ||h(L) @ Z_T - h(L) @ Z_S||^2
   
   The filter automatically learns which frequency bands are important
   for knowledge transfer on heterophilic graphs.

Mathematical Foundation:
------------------------
- Bernstein polynomials are numerically stable (no gradient explosion)
- They can approximate any continuous function on [0,1]
- The learned θ_k directly control frequency band importance
- Softmax normalization ensures θ_k form a valid distribution

Reference:
- He et al. "BernNet: Learning Arbitrary Graph Spectral Filters via 
  Bernstein Approximation" (NeurIPS 2021)
- Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from scipy.special import comb


# =============================================================================
# Bernstein Polynomial Basis Functions
# =============================================================================

class BernsteinBasis(nn.Module):
    """
    Compute Bernstein polynomial basis functions applied to graph Laplacian.
    
    B_{k,K}(L̃) for k = 0, 1, ..., K
    
    where L̃ = L/2 is the scaled normalized Laplacian with eigenvalues in [0,1].
    
    Implementation uses the explicit formula:
    B_{k,K}(L) @ x = Σ_{j=0}^{K-k} (-1)^j * C(K,k) * C(K-k,j) * L^{k+j} @ x
    
    This avoids eigendecomposition and works directly with sparse matrices.
    """
    
    def __init__(self, K, adj, device='cpu'):
        """
        Args:
            K: Polynomial order (number of basis functions = K + 1)
            adj: scipy sparse adjacency matrix (without self-loops)
            device: torch device
        """
        super().__init__()
        self.K = K
        self.device = device
        
        # Precompute scaled normalized Laplacian L̃ = L/2
        # L = I - D^{-1/2} A D^{-1/2}, so L̃ = (I - D^{-1/2} A D^{-1/2}) / 2
        self.L_scaled = self._compute_scaled_laplacian(adj, device)
        
        # Precompute binomial coefficients for efficiency
        self._precompute_coefficients()
    
    def _compute_scaled_laplacian(self, adj, device):
        """
        Compute scaled normalized Laplacian L̃ = L/2.
        
        We store the normalized adjacency A_norm = D^{-1/2} A D^{-1/2}
        and compute L̃ @ x = 0.5 * x - 0.5 * A_norm @ x
        """
        if not sp.isspmatrix_csr(adj):
            adj = adj.tocsr()
        
        n = adj.shape[0]
        
        # Add self-loops for normalization
        adj_with_loops = adj + sp.eye(n)
        
        # Compute D^{-1/2}
        degree = np.array(adj_with_loops.sum(axis=1)).flatten()
        degree[degree == 0] = 1
        d_inv_sqrt = 1.0 / np.sqrt(degree)
        D_inv_sqrt = sp.diags(d_inv_sqrt, format='csr')
        
        # Normalized adjacency: A_norm = D^{-1/2} A D^{-1/2}
        A_norm = D_inv_sqrt @ adj_with_loops @ D_inv_sqrt
        
        # Convert to torch sparse tensor
        A_norm_coo = A_norm.tocoo()
        indices = torch.LongTensor(np.vstack([A_norm_coo.row, A_norm_coo.col]))
        values = torch.FloatTensor(A_norm_coo.data.astype(np.float32))
        A_norm_sparse = torch.sparse_coo_tensor(indices, values, (n, n))
        
        return A_norm_sparse.to(device)
    
    def _precompute_coefficients(self):
        """Precompute binomial coefficients for Bernstein basis."""
        K = self.K
        # coeffs[k][j] = (-1)^j * C(K,k) * C(K-k,j)
        self.coeffs = []
        for k in range(K + 1):
            coeffs_k = []
            for j in range(K - k + 1):
                c = ((-1) ** j) * comb(K, k, exact=True) * comb(K - k, j, exact=True)
                coeffs_k.append(c)
            self.coeffs.append(coeffs_k)
    
    def _apply_scaled_laplacian(self, x):
        """
        Apply L̃ = (I - A_norm) / 2 to signal x.
        L̃ @ x = 0.5 * x - 0.5 * A_norm @ x
        """
        return 0.5 * x - 0.5 * torch.sparse.mm(self.L_scaled, x)
    
    def _compute_laplacian_powers(self, x):
        """
        Compute L̃^0 @ x, L̃^1 @ x, ..., L̃^K @ x iteratively.
        """
        powers = [x]  # L̃^0 @ x = x
        current = x
        for _ in range(self.K):
            current = self._apply_scaled_laplacian(current)
            powers.append(current)
        return powers
    
    def forward(self, x):
        """
        Compute Bernstein basis functions applied to signal x.
        
        Args:
            x: Input signal (N, C)
            
        Returns:
            basis_outputs: List of K+1 tensors, each (N, C)
                          basis_outputs[k] = B_{k,K}(L̃) @ x
        """
        # Compute powers of L̃ @ x
        L_powers = self._compute_laplacian_powers(x)
        
        # Compute B_{k,K}(L̃) @ x using explicit formula
        basis_outputs = []
        for k in range(self.K + 1):
            # B_{k,K}(L̃) @ x = Σ_{j=0}^{K-k} coeffs[k][j] * L̃^{k+j} @ x
            result = torch.zeros_like(x)
            for j, coeff in enumerate(self.coeffs[k]):
                if k + j <= self.K:
                    result = result + coeff * L_powers[k + j]
            basis_outputs.append(result)
        
        return basis_outputs


# =============================================================================
# Learnable Spectral Filter
# =============================================================================

class LearnableBernsteinFilter(nn.Module):
    """
    Learnable spectral filter using Bernstein polynomial basis.
    
    h(L̃) = Σ_{k=0}^{K} θ_k * B_{k,K}(L̃)
    
    where θ_k are learnable parameters normalized via softmax.
    
    The filter can learn arbitrary frequency responses:
    - Low-pass: θ concentrated on small k
    - High-pass: θ concentrated on large k  
    - Band-pass: θ concentrated on middle k
    - Custom: any combination learned from data
    """
    
    def __init__(self, K, adj, init_mode='uniform', device='cpu'):
        """
        Args:
            K: Polynomial order
            adj: scipy sparse adjacency matrix
            init_mode: Initialization for θ
                - 'uniform': All θ_k equal
                - 'low_pass': Emphasize low frequencies
                - 'high_pass': Emphasize high frequencies
                - 'band_pass': Emphasize middle frequencies
            device: torch device
        """
        super().__init__()
        self.K = K
        self.device = device
        
        # Bernstein basis computer
        self.basis = BernsteinBasis(K, adj, device)
        
        # Learnable coefficients θ_k (will be normalized via softmax)
        theta_init = self._init_theta(K, init_mode)
        self.theta = nn.Parameter(theta_init.to(device))
    
    def _init_theta(self, K, mode):
        """Initialize θ coefficients based on desired frequency response."""
        if mode == 'uniform':
            # Equal weights
            return torch.zeros(K + 1)
        elif mode == 'low_pass':
            # Emphasize low frequencies (small k = low eigenvalues)
            return -torch.arange(K + 1).float()
        elif mode == 'high_pass':
            # Emphasize high frequencies (large k = high eigenvalues)
            return torch.arange(K + 1).float()
        elif mode == 'band_pass':
            # Emphasize middle frequencies
            center = K / 2
            return -((torch.arange(K + 1).float() - center) ** 2) / (K / 2)
        else:
            return torch.zeros(K + 1)
    
    def forward(self, x):
        """
        Apply learnable spectral filter to signal x.
        
        Args:
            x: Input signal (N, C)
            
        Returns:
            filtered: Filtered signal (N, C)
        """
        # Get basis outputs: [B_0(L)@x, B_1(L)@x, ..., B_K(L)@x]
        basis_outputs = self.basis(x)
        
        # Normalize θ via softmax to get valid distribution
        theta_normalized = F.softmax(self.theta, dim=0)
        
        # Weighted sum: h(L) @ x = Σ_k θ_k * B_{k,K}(L) @ x
        filtered = torch.zeros_like(x)
        for k, basis_k in enumerate(basis_outputs):
            filtered = filtered + theta_normalized[k] * basis_k
        
        return filtered
    
    def get_frequency_response(self, num_points=100):
        """
        Compute the learned frequency response h(λ) for visualization.
        
        Returns:
            lambdas: Frequency values in [0, 1]
            response: Filter response at each frequency
        """
        with torch.no_grad():
            theta_normalized = F.softmax(self.theta, dim=0).cpu().numpy()
        
        lambdas = np.linspace(0, 1, num_points)
        response = np.zeros_like(lambdas)
        
        for k in range(self.K + 1):
            # B_{k,K}(λ) = C(K,k) * λ^k * (1-λ)^{K-k}
            binom = comb(self.K, k, exact=True)
            B_k = binom * (lambdas ** k) * ((1 - lambdas) ** (self.K - k))
            response += theta_normalized[k] * B_k
        
        return lambdas, response
    
    def get_theta_distribution(self):
        """Get normalized θ distribution."""
        with torch.no_grad():
            return F.softmax(self.theta, dim=0).cpu().numpy()


# =============================================================================
# AFD-KD Loss (Upgraded)
# =============================================================================

class AFDLoss(nn.Module):
    """
    Adaptive Frequency-Decoupled Knowledge Distillation Loss.
    
    UPGRADED from static low/high-pass to learnable Bernstein polynomial filter.
    
    Loss = ||h(L) @ Z_T - h(L) @ Z_S||^2
    
    where h(L) = Σ_{k=0}^{K} θ_k * B_{k,K}(L) is a learnable spectral filter.
    
    The filter automatically discovers which frequency bands are important
    for knowledge transfer, adapting to the graph's spectral properties.
    """
    
    def __init__(self, adj, K=10, init_mode='uniform', 
                 loss_type='mse', temperature=4.0, device='cpu'):
        """
        Args:
            adj: scipy sparse adjacency matrix
            K: Bernstein polynomial order (higher = finer frequency resolution)
            init_mode: Initialization for filter coefficients
            loss_type: 'mse' or 'kl' for alignment loss
            temperature: Temperature for KL divergence (if used)
            device: torch device
        """
        super().__init__()
        self.K = K
        self.loss_type = loss_type
        self.T = temperature
        self.device = device
        
        # Learnable spectral filter
        self.filter = LearnableBernsteinFilter(K, adj, init_mode, device)
    
    def forward(self, logits_student, logits_teacher, mask=None):
        """
        Compute AFD loss between student and teacher.
        
        Args:
            logits_student: Student logits (N, C)
            logits_teacher: Teacher logits (N, C)
            mask: Optional boolean mask for nodes to include
            
        Returns:
            loss: Scalar loss value
            loss_dict: Dictionary with component losses for logging
        """
        # Apply learnable spectral filter to both
        Z_T_filtered = self.filter(logits_teacher)
        Z_S_filtered = self.filter(logits_student)
        
        # Compute alignment loss
        if self.loss_type == 'mse':
            # MSE loss on filtered signals
            loss_per_node = ((Z_S_filtered - Z_T_filtered) ** 2).mean(dim=1)
        elif self.loss_type == 'kl':
            # KL divergence on filtered logits
            p_s = F.log_softmax(Z_S_filtered / self.T, dim=1)
            p_t = F.softmax(Z_T_filtered / self.T, dim=1)
            loss_per_node = F.kl_div(p_s, p_t, reduction='none').sum(dim=1) * (self.T ** 2)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Apply mask if provided
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
            loss_per_node = loss_per_node[mask]
        
        loss = loss_per_node.mean()
        
        # Logging info
        loss_dict = {
            'afd_loss': loss.item(),
            'theta_entropy': self._compute_theta_entropy(),
        }
        
        return loss, loss_dict
    
    def _compute_theta_entropy(self):
        """Compute entropy of θ distribution (measure of filter sharpness)."""
        with torch.no_grad():
            theta_norm = F.softmax(self.filter.theta, dim=0)
            entropy = -(theta_norm * torch.log(theta_norm + 1e-10)).sum()
            return entropy.item()
    
    def get_frequency_response(self):
        """Get learned frequency response for visualization."""
        return self.filter.get_frequency_response()
    
    def get_theta_distribution(self):
        """Get learned θ distribution."""
        return self.filter.get_theta_distribution()


# =============================================================================
# Multi-Band AFD Loss (Advanced)
# =============================================================================

class MultiBandAFDLoss(nn.Module):
    """
    Multi-Band Adaptive Frequency-Decoupled Loss.
    
    Uses multiple learnable filters with different initializations to capture
    different aspects of the frequency spectrum.
    
    Loss = Σ_b w_b * ||h_b(L) @ Z_T - h_b(L) @ Z_S||^2
    
    where h_b are different learnable filters and w_b are learnable band weights.
    """
    
    def __init__(self, adj, num_bands=3, K=10, temperature=4.0, device='cpu'):
        """
        Args:
            adj: scipy sparse adjacency matrix
            num_bands: Number of frequency bands (filters)
            K: Bernstein polynomial order
            temperature: Temperature for soft targets
            device: torch device
        """
        super().__init__()
        self.num_bands = num_bands
        self.K = K
        self.T = temperature
        self.device = device
        
        # Create multiple filters with different initializations
        init_modes = ['low_pass', 'band_pass', 'high_pass', 'uniform']
        self.filters = nn.ModuleList([
            LearnableBernsteinFilter(K, adj, init_modes[i % len(init_modes)], device)
            for i in range(num_bands)
        ])
        
        # Learnable band weights
        self.band_weights = nn.Parameter(torch.zeros(num_bands))
    
    def forward(self, logits_student, logits_teacher, mask=None):
        """
        Compute multi-band AFD loss.
        
        Args:
            logits_student: Student logits (N, C)
            logits_teacher: Teacher logits (N, C)
            mask: Optional boolean mask
            
        Returns:
            loss: Scalar loss
            loss_dict: Component losses for logging
        """
        # Normalize band weights
        band_weights = F.softmax(self.band_weights, dim=0)
        
        total_loss = 0
        loss_dict = {}
        
        for b, (filter_b, weight_b) in enumerate(zip(self.filters, band_weights)):
            # Apply filter to both teacher and student
            T_filtered = filter_b(logits_teacher)
            S_filtered = filter_b(logits_student)
            
            # MSE loss on filtered signals
            band_loss = ((S_filtered - T_filtered) ** 2).mean(dim=1)
            
            # Apply mask
            if mask is not None:
                if isinstance(mask, np.ndarray):
                    mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
                band_loss = band_loss[mask]
            
            weighted_loss = weight_b * band_loss.mean()
            total_loss = total_loss + weighted_loss
            
            loss_dict[f'band_{b}_loss'] = band_loss.mean().item()
            loss_dict[f'band_{b}_weight'] = weight_b.item()
        
        loss_dict['total_afd_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def get_all_frequency_responses(self):
        """Get frequency responses for all bands."""
        responses = []
        band_weights = F.softmax(self.band_weights, dim=0).detach().cpu().numpy()
        
        for i, filter_b in enumerate(self.filters):
            lambdas, response = filter_b.get_frequency_response()
            responses.append({
                'band': i,
                'lambdas': lambdas,
                'response': response,
                'weight': band_weights[i],
                'theta': filter_b.get_theta_distribution()
            })
        return responses


# =============================================================================
# Complete Hybrid Loss (CE + Soft KD + AFD)
# =============================================================================

class HybridAFDLoss(nn.Module):
    """
    Complete hybrid loss combining:
    1. Cross-entropy loss on hard labels
    2. Standard soft-target KD loss  
    3. Adaptive Frequency-Decoupled (AFD) loss
    
    L_total = L_CE + λ_soft * L_soft + λ_afd * L_AFD
    
    All learnable parameters (θ in AFD filters) are optimized jointly
    with the student model.
    """
    
    def __init__(self, adj, K=10, num_bands=1, temperature=4.0,
                 lambda_soft=1.0, lambda_afd=0.5, 
                 use_multi_band=False, device='cpu'):
        """
        Args:
            adj: scipy sparse adjacency matrix
            K: Bernstein polynomial order
            num_bands: Number of frequency bands (if use_multi_band=True)
            temperature: KD temperature
            lambda_soft: Weight for soft-target loss
            lambda_afd: Weight for AFD loss
            use_multi_band: Whether to use multi-band AFD
            device: torch device
        """
        super().__init__()
        self.T = temperature
        self.lambda_soft = lambda_soft
        self.lambda_afd = lambda_afd
        self.device = device
        
        # AFD loss (single or multi-band)
        if use_multi_band:
            self.afd_loss = MultiBandAFDLoss(adj, num_bands, K, temperature, device)
        else:
            self.afd_loss = AFDLoss(adj, K, 'uniform', 'mse', temperature, device)
        
        # Standard CE loss
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits_student, logits_teacher, labels, train_mask=None):
        """
        Compute complete hybrid loss.
        
        Args:
            logits_student: Student logits (N, C)
            logits_teacher: Teacher logits (N, C)
            labels: Ground truth labels (N,)
            train_mask: Boolean mask for training nodes
            
        Returns:
            loss: Total loss
            loss_dict: Component losses for logging
        """
        loss_dict = {}
        
        # 1. CE loss on training nodes
        if train_mask is not None:
            if isinstance(train_mask, np.ndarray):
                train_mask = torch.tensor(train_mask, dtype=torch.bool, device=self.device)
            loss_ce = self.ce_loss(logits_student[train_mask], labels[train_mask])
        else:
            loss_ce = self.ce_loss(logits_student, labels)
        loss_dict['loss_ce'] = loss_ce.item()
        
        # 2. Standard soft-target KD loss
        if self.lambda_soft > 0:
            p_s = F.log_softmax(logits_student / self.T, dim=1)
            p_t = F.softmax(logits_teacher / self.T, dim=1)
            loss_soft = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
            loss_dict['loss_soft'] = loss_soft.item()
        else:
            loss_soft = 0
        
        # 3. AFD loss (learnable spectral alignment)
        if self.lambda_afd > 0:
            loss_afd, afd_dict = self.afd_loss(logits_student, logits_teacher)
            loss_dict.update({f'afd_{k}': v for k, v in afd_dict.items()})
        else:
            loss_afd = 0
        
        # Total loss
        loss_total = loss_ce + self.lambda_soft * loss_soft + self.lambda_afd * loss_afd
        loss_dict['loss_total'] = loss_total.item()
        
        return loss_total, loss_dict
    
    def get_frequency_response(self):
        """Get learned frequency response(s)."""
        if isinstance(self.afd_loss, MultiBandAFDLoss):
            return self.afd_loss.get_all_frequency_responses()
        else:
            return self.afd_loss.get_frequency_response()


# =============================================================================
# Legacy Classes (Backward Compatibility)
# =============================================================================

class SpectralDecomposer(nn.Module):
    """
    [LEGACY] Static low/high-pass decomposition.
    Kept for backward compatibility.
    """
    
    def __init__(self, adj, device='cpu'):
        super().__init__()
        self.device = device
        self.P = self._compute_random_walk_matrix(adj, device)
        
    def _compute_random_walk_matrix(self, adj, device):
        if not sp.isspmatrix_csr(adj):
            adj = adj.tocsr()
        
        degree = np.array(adj.sum(axis=1)).flatten()
        degree[degree == 0] = 1
        d_inv = 1.0 / degree
        d_inv_diag = sp.diags(d_inv, format='csr')
        P = d_inv_diag @ adj
        
        P_coo = P.tocoo()
        indices = torch.LongTensor(np.vstack([P_coo.row, P_coo.col]))
        values = torch.FloatTensor(P_coo.data)
        P_sparse = torch.sparse_coo_tensor(indices, values, P_coo.shape)
        
        return P_sparse.to(device)
    
    def decompose(self, Z):
        Z_low = torch.sparse.mm(self.P, Z)
        Z_high = Z - Z_low
        return Z_low, Z_high


class AdaptiveSpectralKDLoss(nn.Module):
    """
    [LEGACY] Static homophily-weighted spectral KD loss.
    Kept for backward compatibility. Use AFDLoss for new experiments.
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
        
        h = homophily_weights.squeeze()
        if not isinstance(h, torch.Tensor):
            h = torch.tensor(h, dtype=torch.float32)
        self.register_buffer('homophily', h.to(device))
        
    def forward(self, logits_student, logits_teacher, mask=None):
        T_low, T_high = self.decomposer.decompose(logits_teacher)
        S_low, S_high = self.decomposer.decompose(logits_student)
        
        loss_low = self._kl_div_loss(S_low, T_low, self.T)
        loss_high = self._mse_loss(S_high, T_high) * self.high_freq_scale
        
        h = self.homophily
        
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.bool, device=h.device)
            h = h[mask]
            loss_low = loss_low[mask]
            loss_high = loss_high[mask]
        
        weighted_loss = (
            self.alpha_low * h * loss_low + 
            self.alpha_high * (1 - h) * loss_high
        )
        
        loss = weighted_loss.mean()
        
        loss_dict = {
            'loss_total': loss.item(),
            'loss_low': (h * loss_low).mean().item(),
            'loss_high': ((1 - h) * loss_high).mean().item(),
            'mean_homophily': h.mean().item()
        }
        
        return loss, loss_dict
    
    def _kl_div_loss(self, logits_s, logits_t, T):
        p_s = F.log_softmax(logits_s / T, dim=1)
        p_t = F.softmax(logits_t / T, dim=1)
        kl = F.kl_div(p_s, p_t, reduction='none').sum(dim=1)
        return kl * (T * T)
    
    def _mse_loss(self, logits_s, logits_t):
        mse = ((logits_s - logits_t) ** 2).mean(dim=1)
        return mse


# Alias for backward compatibility
HybridAdaptiveLoss = HybridAFDLoss


# =============================================================================
# Gated Adaptive Frequency Fusion (GAFF) - Core Innovation
# =============================================================================

class NodeHomophilyComputer(nn.Module):
    """
    Compute node-level local homophily scores.
    
    For each node, measures how similar its features are to its neighbors.
    High score = homophilic (similar to neighbors)
    Low score = heterophilic (different from neighbors)
    
    This provides the "gate signal" for adaptive frequency fusion.
    """
    
    def __init__(self, adj, device='cpu'):
        """
        Args:
            adj: scipy sparse adjacency matrix
            device: torch device
        """
        super().__init__()
        self.device = device
        
        # Store adjacency for neighbor aggregation
        self._setup_adjacency(adj, device)
    
    def _setup_adjacency(self, adj, device):
        """Convert adjacency to torch sparse tensor."""
        if not sp.isspmatrix_csr(adj):
            adj = adj.tocsr()
        
        n = adj.shape[0]
        
        # Row-normalize adjacency (for averaging neighbors)
        degree = np.array(adj.sum(axis=1)).flatten()
        degree[degree == 0] = 1
        d_inv = 1.0 / degree
        D_inv = sp.diags(d_inv, format='csr')
        adj_norm = D_inv @ adj
        
        # Convert to torch sparse
        adj_coo = adj_norm.tocoo()
        indices = torch.LongTensor(np.vstack([adj_coo.row, adj_coo.col]))
        values = torch.FloatTensor(adj_coo.data.astype(np.float32))
        self.adj_norm = torch.sparse_coo_tensor(indices, values, (n, n)).to(device)
        
        # Store degree for weighting
        self.degree = torch.FloatTensor(degree).to(device)
    
    def forward(self, features):
        """
        Compute node-level homophily scores based on feature similarity.
        
        Args:
            features: Node features or logits (N, C)
            
        Returns:
            homophily_scores: Per-node homophily in [0, 1], shape (N,)
        """
        # Normalize features for cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Aggregate neighbor features: mean of neighbors
        neighbor_features = torch.sparse.mm(self.adj_norm, features_norm)
        
        # Cosine similarity between node and its neighbor average
        # Higher = more similar to neighbors = more homophilic
        similarity = (features_norm * neighbor_features).sum(dim=1)
        
        # Clamp to [0, 1] (cosine similarity can be negative)
        homophily_scores = (similarity + 1) / 2  # Map [-1, 1] to [0, 1]
        homophily_scores = torch.clamp(homophily_scores, 0, 1)
        
        return homophily_scores


class GatedFrequencyFusion(nn.Module):
    """
    Gated Adaptive Frequency Fusion (GAFF) Module.
    
    Core Innovation: Dynamically blend high-frequency AFD loss and standard
    soft-target KD loss based on each node's local homophily.
    
    For heterophilic nodes (low homophily):
        → Emphasize AFD loss (spectral filtering helps)
    
    For homophilic nodes (high homophily):
        → Emphasize standard KD loss (simple soft targets work well)
    
    Mathematical Formulation:
    -------------------------
    Let h_i ∈ [0,1] be node i's local homophily score.
    
    Gate signal: g_i = σ(w * (h_i - τ) + b)
    
    where:
        - σ is sigmoid
        - τ is a learnable threshold (default 0.5)
        - w controls gate sharpness
        - b is bias
    
    Per-node loss:
        L_i = g_i * L_soft_i + (1 - g_i) * L_AFD_i
    
    This allows the model to "see people, speak human; see ghosts, speak ghost"
    (见人说人话，见鬼说鬼话).
    """
    
    def __init__(self, adj, K=10, temperature=4.0, 
                 gate_init_threshold=0.5, gate_sharpness=5.0,
                 learnable_gate=True, device='cpu'):
        """
        Args:
            adj: scipy sparse adjacency matrix
            K: Bernstein polynomial order for AFD filter
            temperature: KD temperature
            gate_init_threshold: Initial threshold for gate (default 0.5)
            gate_sharpness: How sharp the gate transition is
            learnable_gate: Whether gate parameters are learnable
            device: torch device
        """
        super().__init__()
        self.T = temperature
        self.device = device
        self.K = K
        
        # Homophily computer
        self.homophily_computer = NodeHomophilyComputer(adj, device)
        
        # AFD filter for high-frequency alignment
        self.afd_filter = LearnableBernsteinFilter(K, adj, 'high_pass', device)
        
        # Gate parameters
        if learnable_gate:
            self.gate_threshold = nn.Parameter(torch.tensor(gate_init_threshold))
            self.gate_sharpness = nn.Parameter(torch.tensor(gate_sharpness))
            self.gate_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('gate_threshold', torch.tensor(gate_init_threshold))
            self.register_buffer('gate_sharpness', torch.tensor(gate_sharpness))
            self.register_buffer('gate_bias', torch.tensor(0.0))
    
    def compute_gate(self, homophily_scores):
        """
        Compute gate values from homophily scores.
        
        High homophily → high gate → more standard KD
        Low homophily → low gate → more AFD
        
        Args:
            homophily_scores: Per-node homophily (N,)
            
        Returns:
            gate: Per-node gate values in [0, 1], shape (N,)
        """
        # Gate: sigmoid(sharpness * (homophily - threshold) + bias)
        gate_input = self.gate_sharpness * (homophily_scores - self.gate_threshold) + self.gate_bias
        gate = torch.sigmoid(gate_input)
        return gate
    
    def forward(self, logits_student, logits_teacher, features_for_homophily=None):
        """
        Compute gated frequency fusion loss.
        
        Args:
            logits_student: Student logits (N, C)
            logits_teacher: Teacher logits (N, C)
            features_for_homophily: Features to compute homophily (default: teacher logits)
            
        Returns:
            loss_per_node: Per-node loss (N,)
            gate: Per-node gate values (N,)
            homophily: Per-node homophily scores (N,)
        """
        # Use teacher logits for homophily if not specified
        if features_for_homophily is None:
            features_for_homophily = logits_teacher
        
        # Compute node-level homophily
        homophily = self.homophily_computer(features_for_homophily)
        
        # Compute gate values
        gate = self.compute_gate(homophily)
        
        # Standard soft-target KD loss (per-node)
        p_s = F.log_softmax(logits_student / self.T, dim=1)
        p_t = F.softmax(logits_teacher / self.T, dim=1)
        loss_soft = F.kl_div(p_s, p_t, reduction='none').sum(dim=1) * (self.T ** 2)
        
        # AFD loss (per-node): MSE on filtered signals
        Z_T_filtered = self.afd_filter(logits_teacher)
        Z_S_filtered = self.afd_filter(logits_student)
        loss_afd = ((Z_S_filtered - Z_T_filtered) ** 2).mean(dim=1)
        
        # Gated fusion: gate * soft + (1 - gate) * AFD
        # High gate (homophilic) → more soft KD
        # Low gate (heterophilic) → more AFD
        loss_per_node = gate * loss_soft + (1 - gate) * loss_afd
        
        return loss_per_node, gate, homophily
    
    def get_gate_stats(self, homophily_scores):
        """Get statistics about gate behavior."""
        with torch.no_grad():
            gate = self.compute_gate(homophily_scores)
            return {
                'gate_mean': gate.mean().item(),
                'gate_std': gate.std().item(),
                'gate_min': gate.min().item(),
                'gate_max': gate.max().item(),
                'threshold': self.gate_threshold.item(),
                'sharpness': self.gate_sharpness.item(),
            }


class GatedAFDLoss(nn.Module):
    """
    Complete Gated Adaptive Frequency-Decoupled Loss.
    
    Combines:
    1. Cross-entropy loss on hard labels
    2. Gated frequency fusion (adaptive blend of soft KD and AFD)
    
    L_total = L_CE + λ * L_gated
    
    where L_gated = Σ_i [g_i * L_soft_i + (1-g_i) * L_AFD_i]
    
    This is the "全能生" (all-rounder) version that works well on both
    homophilic and heterophilic nodes.
    """
    
    def __init__(self, adj, K=10, temperature=4.0, lambda_kd=1.0,
                 gate_init_threshold=0.5, gate_sharpness=5.0,
                 learnable_gate=True, device='cpu'):
        """
        Args:
            adj: scipy sparse adjacency matrix
            K: Bernstein polynomial order
            temperature: KD temperature
            lambda_kd: Weight for KD loss
            gate_init_threshold: Initial gate threshold
            gate_sharpness: Gate transition sharpness
            learnable_gate: Whether gate is learnable
            device: torch device
        """
        super().__init__()
        self.lambda_kd = lambda_kd
        self.device = device
        
        # Gated frequency fusion module
        self.gated_fusion = GatedFrequencyFusion(
            adj, K, temperature, gate_init_threshold, 
            gate_sharpness, learnable_gate, device
        )
        
        # CE loss
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits_student, logits_teacher, labels, train_mask=None,
                features_for_homophily=None):
        """
        Compute complete gated AFD loss.
        
        Args:
            logits_student: Student logits (N, C)
            logits_teacher: Teacher logits (N, C)
            labels: Ground truth labels (N,)
            train_mask: Boolean mask for training nodes
            features_for_homophily: Features for computing homophily
            
        Returns:
            loss: Total loss
            loss_dict: Component losses for logging
        """
        loss_dict = {}
        
        # 1. CE loss on training nodes
        if train_mask is not None:
            if isinstance(train_mask, np.ndarray):
                train_mask = torch.tensor(train_mask, dtype=torch.bool, device=self.device)
            loss_ce = self.ce_loss(logits_student[train_mask], labels[train_mask])
        else:
            loss_ce = self.ce_loss(logits_student, labels)
        loss_dict['loss_ce'] = loss_ce.item()
        
        # 2. Gated frequency fusion loss
        loss_per_node, gate, homophily = self.gated_fusion(
            logits_student, logits_teacher, features_for_homophily
        )
        
        # Average over all nodes (or masked nodes)
        loss_gated = loss_per_node.mean()
        loss_dict['loss_gated'] = loss_gated.item()
        
        # Gate statistics
        loss_dict['gate_mean'] = gate.mean().item()
        loss_dict['gate_std'] = gate.std().item()
        loss_dict['homophily_mean'] = homophily.mean().item()
        
        # Breakdown by homophily level
        homo_mask = homophily > 0.5
        hetero_mask = homophily <= 0.5
        if homo_mask.sum() > 0:
            loss_dict['loss_homo_nodes'] = loss_per_node[homo_mask].mean().item()
        if hetero_mask.sum() > 0:
            loss_dict['loss_hetero_nodes'] = loss_per_node[hetero_mask].mean().item()
        
        # Total loss
        loss_total = loss_ce + self.lambda_kd * loss_gated
        loss_dict['loss_total'] = loss_total.item()
        
        return loss_total, loss_dict
    
    def get_gate_analysis(self, logits_teacher, features_for_homophily=None):
        """
        Analyze gate behavior for visualization.
        
        Returns:
            dict with homophily scores, gate values, and statistics
        """
        with torch.no_grad():
            if features_for_homophily is None:
                features_for_homophily = logits_teacher
            
            homophily = self.gated_fusion.homophily_computer(features_for_homophily)
            gate = self.gated_fusion.compute_gate(homophily)
            
            return {
                'homophily': homophily.cpu().numpy(),
                'gate': gate.cpu().numpy(),
                'stats': self.gated_fusion.get_gate_stats(homophily),
                'theta': self.gated_fusion.afd_filter.get_theta_distribution(),
            }


class DualPathGatedLoss(nn.Module):
    """
    Dual-Path Gated Loss with separate low-freq and high-freq paths.
    
    Even more sophisticated version:
    - Low-frequency path: Standard soft KD (good for homophilic)
    - High-frequency path: AFD with high-pass filter (good for heterophilic)
    - Gate: Dynamically blends based on local homophily
    
    L_i = g_i * L_low_i + (1 - g_i) * L_high_i
    
    where:
        L_low = KL(softmax(Z_S/T), softmax(Z_T/T))  [standard KD]
        L_high = ||h_high(L) @ Z_T - h_high(L) @ Z_S||^2  [high-freq AFD]
    """
    
    def __init__(self, adj, K=10, temperature=4.0, lambda_kd=1.0,
                 gate_init_threshold=0.5, gate_sharpness=5.0,
                 device='cpu'):
        super().__init__()
        self.T = temperature
        self.lambda_kd = lambda_kd
        self.device = device
        
        # Homophily computer
        self.homophily_computer = NodeHomophilyComputer(adj, device)
        
        # Dual filters
        self.low_filter = LearnableBernsteinFilter(K, adj, 'low_pass', device)
        self.high_filter = LearnableBernsteinFilter(K, adj, 'high_pass', device)
        
        # Gate parameters
        self.gate_threshold = nn.Parameter(torch.tensor(gate_init_threshold))
        self.gate_sharpness = nn.Parameter(torch.tensor(gate_sharpness))
        
        # CE loss
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits_student, logits_teacher, labels, train_mask=None):
        loss_dict = {}
        
        # CE loss
        if train_mask is not None:
            if isinstance(train_mask, np.ndarray):
                train_mask = torch.tensor(train_mask, dtype=torch.bool, device=self.device)
            loss_ce = self.ce_loss(logits_student[train_mask], labels[train_mask])
        else:
            loss_ce = self.ce_loss(logits_student, labels)
        loss_dict['loss_ce'] = loss_ce.item()
        
        # Compute homophily and gate
        homophily = self.homophily_computer(logits_teacher)
        gate_input = self.gate_sharpness * (homophily - self.gate_threshold)
        gate = torch.sigmoid(gate_input)
        
        # Low-frequency path: filtered soft KD
        Z_T_low = self.low_filter(logits_teacher)
        Z_S_low = self.low_filter(logits_student)
        p_s_low = F.log_softmax(Z_S_low / self.T, dim=1)
        p_t_low = F.softmax(Z_T_low / self.T, dim=1)
        loss_low = F.kl_div(p_s_low, p_t_low, reduction='none').sum(dim=1) * (self.T ** 2)
        
        # High-frequency path: filtered MSE
        Z_T_high = self.high_filter(logits_teacher)
        Z_S_high = self.high_filter(logits_student)
        loss_high = ((Z_S_high - Z_T_high) ** 2).mean(dim=1)
        
        # Gated fusion
        loss_per_node = gate * loss_low + (1 - gate) * loss_high
        loss_gated = loss_per_node.mean()
        
        loss_dict['loss_gated'] = loss_gated.item()
        loss_dict['loss_low_mean'] = loss_low.mean().item()
        loss_dict['loss_high_mean'] = loss_high.mean().item()
        loss_dict['gate_mean'] = gate.mean().item()
        
        # Total
        loss_total = loss_ce + self.lambda_kd * loss_gated
        loss_dict['loss_total'] = loss_total.item()
        
        return loss_total, loss_dict
    
    def get_dual_path_analysis(self, logits_teacher):
        """Analyze dual path behavior."""
        with torch.no_grad():
            homophily = self.homophily_computer(logits_teacher)
            gate = torch.sigmoid(self.gate_sharpness * (homophily - self.gate_threshold))
            
            return {
                'homophily': homophily.cpu().numpy(),
                'gate': gate.cpu().numpy(),
                'low_theta': self.low_filter.get_theta_distribution(),
                'high_theta': self.high_filter.get_theta_distribution(),
                'threshold': self.gate_threshold.item(),
                'sharpness': self.gate_sharpness.item(),
            }
