"""CARA-FM: Culturally Attuned and Resource-Aware Foundation Models Evaluation Suite."""

from .indices import (
    CARAFMScorecard,
    compute_cai,
    compute_cas,
    compute_cei,
    compute_dfs,
    compute_iae,
    compute_ioe,
    compute_kpv,
    compute_mpp,
    compute_yip,
    evaluate_system_card,
)
from .schema import validate_scorecard_dict

__all__ = [
    "CARAFMScorecard",
    "compute_cai",
    "compute_cas",
    "compute_cei",
    "compute_dfs",
    "compute_iae",
    "compute_ioe",
    "compute_kpv",
    "compute_mpp",
    "compute_yip",
    "evaluate_system_card",
    "validate_scorecard_dict",
]

__version__ = "1.1.0"
