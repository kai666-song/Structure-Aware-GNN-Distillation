from .st import SoftTarget
from .rkd import RKDLoss, RKDDistanceLoss, AdaptiveRKDLoss, BatchRKDLoss
from .topology_kd import (
    ContrastiveTopologyLoss, 
    SoftTopologyLoss, 
    AttentionDistillationLoss,
    HybridContrastiveLoss,
    GraphMixup,
    # Backward compatibility aliases
    TopologyConsistencyLoss,
    AdaptiveTopologyLoss, 
    HybridTopologyLoss
)
from .adaptive_kd import (
    SpectralDecomposer,
    AdaptiveSpectralKDLoss,
    HybridAdaptiveLoss
)