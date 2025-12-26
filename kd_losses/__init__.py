# Gated Adaptive Frequency-Decoupled Knowledge Distillation
# Core implementation for heterophilic graph distillation

from .adaptive_kd import (
    # Bernstein polynomial spectral filters
    BernsteinBasis,
    LearnableBernsteinFilter,
    
    # AFD Loss components
    AFDLoss,
    MultiBandAFDLoss,
    HybridAFDLoss,
    
    # Gated mechanism (main contribution)
    NodeHomophilyComputer,
    GatedFrequencyFusion,
    GatedAFDLoss,
    DualPathGatedLoss,
)
