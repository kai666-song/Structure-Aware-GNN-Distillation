"""Configuration module for standardized experiments."""

from .experiment_config import (
    RANDOM_SEEDS,
    NUM_SPLITS,
    HETEROPHILIC_DATASETS,
    DATASET_NAME_MAP,
    SPLIT_RATIO,
    GLOGNN_CONFIGS,
    GLOGNN_REPORTED_RESULTS,
    REPRODUCTION_TOLERANCE,
    GLOGNN_SPLITS_DIR,
    GLOGNN_DATA_DIR,
    DATA_DIR,
    RESULTS_DIR,
    CHECKPOINTS_DIR,
    get_split_path,
    validate_reproduction,
    print_config,
)

__all__ = [
    'RANDOM_SEEDS',
    'NUM_SPLITS', 
    'HETEROPHILIC_DATASETS',
    'DATASET_NAME_MAP',
    'SPLIT_RATIO',
    'GLOGNN_CONFIGS',
    'GLOGNN_REPORTED_RESULTS',
    'REPRODUCTION_TOLERANCE',
    'GLOGNN_SPLITS_DIR',
    'GLOGNN_DATA_DIR',
    'DATA_DIR',
    'RESULTS_DIR',
    'CHECKPOINTS_DIR',
    'get_split_path',
    'validate_reproduction',
    'print_config',
]
