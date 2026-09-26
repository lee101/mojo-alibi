"""mojo-alibi: the compute-oriented subset of alibi with Mojo inner loops.

Names mirror `alibi` so the port drops into existing code; the numerical
contract is C-contiguous `float64` throughout.
"""

from mojo_alibi._lib import (
    abdm,
    class_metrics,
    cityblock_batch,
    exact_gather,
    feature_swap,
    gaussian_rbf,
    infer_sigma,
    kl_bernoulli,
    linearity_score,
    mvdm,
    pd_variance,
    perm_gather,
    permutation_importance_samples,
    protoselect_summarise,
    reg_metrics,
    rbf_kernel,
    row_l2,
    sort_f64,
    squared_pairwise_distance,
    superposition,
)

__all__ = [
    "abdm",
    "class_metrics",
    "cityblock_batch",
    "exact_gather",
    "feature_swap",
    "gaussian_rbf",
    "infer_sigma",
    "kl_bernoulli",
    "linearity_score",
    "mvdm",
    "pd_variance",
    "perm_gather",
    "permutation_importance_samples",
    "protoselect_summarise",
    "reg_metrics",
    "rbf_kernel",
    "row_l2",
    "sort_f64",
    "squared_pairwise_distance",
    "superposition",
]

__version__ = "0.1.0"
