"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-alibi.so"

_ADDR = ctypes.c_int64
_N = ctypes.c_int64
_F = ctypes.c_double

# Above this many fused multiply-adds the pairwise kernels are worth fanning
# out over a thread pool; below it the ctypes call overhead dominates.
THREAD_THRESHOLD = 1 << 19
MAX_WORKERS = int(os.environ.get("MOJO_ALIBI_WORKERS", "8"))


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(f"{_LIB_PATH} not found; run `bash build/build.sh` first")
    lib = ctypes.CDLL(str(_LIB_PATH))
    sig = {
        "al_sort_f64": [_N, _N],
        "al_radix_sort_f64": [_ADDR, _ADDR, _ADDR, _ADDR, _N],
        "al_kl_bernoulli": [_ADDR, _ADDR, _N, _ADDR],
        "al_cityblock_batch": [_ADDR, _ADDR, _N, _N, _ADDR],
        "al_sqdist_rows": [_ADDR, _ADDR, _N, _N, _ADDR, _N, _N],
        "al_rbf_kernel": [_ADDR, _ADDR, _N, _N, _N, _ADDR],
        "al_order_stat": [_ADDR, _N, _ADDR],
        "al_mvdm_pairwise": [_ADDR, _N, _N, _F, _ADDR],
        "al_abdm_pairwise": [_ADDR, _N, _N, _F, _ADDR],
        "al_pd_variance": [_ADDR, _N, _N, _F, _ADDR],
        "al_pd_range": [_ADDR, _N, _N, _ADDR],
        "al_protoselect_masks": [
            _ADDR, _ADDR, _ADDR, _N, _N, _N, _F, _F, _ADDR, _ADDR, _ADDR, _ADDR,
        ],
        "al_protoselect_argmax": [_ADDR, _ADDR, _N, _N, _ADDR],
        "al_protoselect_update": [_ADDR, _ADDR, _N, _N, _N, _N, _N, _ADDR],
        "al_perm_gather": [_ADDR, _ADDR, _N, _N, _ADDR],
        "al_feature_swap": [_ADDR, _N, _N, _N, _N],
        "al_class_metrics": [_ADDR, _ADDR, _ADDR, _N, _N, _ADDR],
        "al_reg_metrics": [_ADDR, _ADDR, _ADDR, _N, _ADDR],
        "al_importance": [_F, _ADDR, _N, _N, _N, _ADDR],
        "al_superpose": [_ADDR, _ADDR, _N, _N, _N, _ADDR],
        "al_row_l2": [_ADDR, _N, _N, _ADDR],
        "al_diff_row_l2": [_ADDR, _ADDR, _N, _N, _ADDR],
        "al_conditional_probs": [_ADDR, _ADDR, _N, _N, _N, _ADDR],
        "al_conditional_probs_eps": [_ADDR, _ADDR, _N, _N, _N, _F, _ADDR],
        "al_exact_gather": [_ADDR, _N, _N, _N, _N, _N, _ADDR],
    }
    for name, argtypes in sig.items():
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = argtypes
    return lib


lib = _load()


def _addr(a) -> int:
    return a.ctypes.data


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _i32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32)


def _fanout(total_work: int, n_chunks: int, fn, workers: int = MAX_WORKERS):
    """Run `fn(chunk_index)` over `n_chunks` slices, threading only when it pays."""
    if workers <= 1 or total_work < THREAD_THRESHOLD or n_chunks <= 1:
        for c in range(n_chunks):
            fn(c)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fn, range(n_chunks)))


# --------------------------------------------------------------------------
# alibi.utils.distributions
# --------------------------------------------------------------------------
def kl_bernoulli(p, q) -> np.ndarray:
    p = _f64(p)
    q = _f64(q)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    out = np.empty_like(p)
    lib.al_kl_bernoulli(_addr(p), _addr(q), p.size, _addr(out))
    return out


# --------------------------------------------------------------------------
# alibi.utils.distance
# --------------------------------------------------------------------------
def cityblock_batch(X, y) -> np.ndarray:
    X = _f64(X)
    y = _f64(y).reshape(-1)
    n, f = X.shape[0], X.shape[1]
    if y.size != f:
        raise ValueError(f"y must have {f} features, got {y.size}")
    out = np.empty(n, dtype=np.float64)
    lib.al_cityblock_batch(_addr(X), _addr(y), n, f, _addr(out))
    return out.reshape(n, -1)


def squared_pairwise_distance(x, y, a_min: float = 1e-7, a_max: float = 1e30):
    x = _f64(x).reshape(x.shape[0], -1)
    y = _f64(y).reshape(y.shape[0], -1)
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature count")
    if a_min != 1e-7 or a_max != 1e30:
        raise ValueError("only upstream's default clip bounds (1e-7, 1e30) exist")
    nx, ny, f = x.shape[0], y.shape[0], x.shape[1]
    out = np.empty((nx, ny), dtype=np.float64)
    n_chunks = min(MAX_WORKERS, max(1, nx))
    step = (nx + n_chunks - 1) // n_chunks

    def run(c):
        lo = c * step
        hi = min(lo + step, nx)
        if lo < hi:
            lib.al_sqdist_rows(_addr(x), _addr(y), ny, f, _addr(out), lo, hi)

    _fanout(nx * ny * f, n_chunks, run)
    return out


def mvdm(X, y, cat_vars: dict, alpha: float = 1.0) -> dict:
    """Pairwise category distances by the modified value difference measure."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    uniq_y = np.unique(y)
    n_y = len(uniq_y)
    out = {}
    for col, n_cat in cat_vars.items():
        if n_cat is None:
            n_cat = int(len(np.unique(X[:, col])))
            cat_vars[col] = n_cat
        col_data = _i32(X[:, col].reshape(-1))
        y_codes = _i32(np.searchsorted(uniq_y, y).reshape(-1))
        p_cond = np.zeros((n_cat, n_y), dtype=np.float64)
        lib.al_conditional_probs(
            _addr(col_data), _addr(y_codes), col_data.size, n_cat, n_y, _addr(p_cond)
        )
        d_pair = np.zeros((n_cat, n_cat), dtype=np.float64)
        lib.al_mvdm_pairwise(
            _addr(p_cond), n_cat, n_y, ctypes.c_double(alpha), _addr(d_pair)
        )
        out[col] = d_pair
    return out


def abdm(X, cat_vars: dict, cat_vars_bin: dict | None = None, eps: float = 1e-12) -> dict:
    """Pairwise category distances by the association-based distance metric."""
    X = np.asarray(X)
    combined = dict(cat_vars)
    combined.update(cat_vars_bin or {})
    out = {}
    for col, n_cat in cat_vars.items():
        if n_cat is None:
            n_cat = int(len(np.unique(X[:, col])))
            cat_vars[col] = n_cat
        group = _i32(X[:, col].reshape(-1))
        blocks = []
        for col_t, n_cat_t in combined.items():
            if col == col_t:
                continue
            # `other` must stay bound to a name: a temporary passed straight
            # into `_addr` is freed the moment the address is taken, and the
            # kernel then reads recycled memory.
            other = _i32(X[:, col_t].reshape(-1))
            p_block = np.zeros((n_cat_t, n_cat), dtype=np.float64)
            lib.al_conditional_probs_eps(
                _addr(group), _addr(other), group.size, n_cat, n_cat_t,
                ctypes.c_double(eps), _addr(p_block),
            )
            blocks.append(p_block)
        p = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, n_cat))
        d_pair = np.zeros((n_cat, n_cat), dtype=np.float64)
        lib.al_abdm_pairwise(
            _addr(_f64(p)), n_cat, p.shape[0], ctypes.c_double(eps), _addr(d_pair)
        )
        out[col] = d_pair
    return out


# --------------------------------------------------------------------------
# alibi.utils.kernel
# --------------------------------------------------------------------------
def sort_f64(a) -> np.ndarray:
    """In-place ascending sort. Uses the radix kernel: the order statistic that
    `GaussianRBF` needs is O(n) to consume once sorted, so a linear-time sort
    beats a comparison sort by a wide margin here."""
    a = _f64(a)
    if a.size <= 1:
        return a
    b0 = np.empty(a.size, dtype=np.uint64)
    b1 = np.empty(a.size, dtype=np.uint64)
    hist = np.zeros(256, dtype=np.uint32)
    lib.al_radix_sort_f64(
        _addr(a), _addr(b0), _addr(b1), _addr(hist), a.size
    )
    return a


def rbf_kernel(dist, gamma) -> np.ndarray:
    dist = _f64(dist)
    gamma = _f64(np.atleast_1d(gamma))
    ny = dist.shape[1] if dist.ndim == 2 else 1
    out = np.empty(dist.size, dtype=np.float64)
    lib.al_rbf_kernel(_addr(dist), _addr(gamma), gamma.size, ny, dist.size, _addr(out))
    return out.reshape(dist.shape)


def infer_sigma(x, y, dist) -> float:
    """`GaussianRBF`'s median-distance sigma inference, exactly as upstream."""
    nx, ny = dist.shape
    n = min(nx, ny)
    same = x.shape == y.shape and bool(np.all(x[:n] == y[:n]))
    n = n if same else 0
    n_median = n + (nx * ny - n) // 2 - 1
    # An explicit copy: the sort is in place, and `_f64` on an already
    # contiguous float64 array returns the same object, so without the copy the
    # caller's distance matrix would come back permuted. The scratch buffers
    # are bound to names for the same reason the input is: a temporary handed
    # straight to `_addr` is freed before the kernel reads it.
    flat = np.array(dist.reshape(-1), dtype=np.float64)
    b0 = np.empty(flat.size, dtype=np.uint64)
    b1 = np.empty(flat.size, dtype=np.uint64)
    hist = np.zeros(256, dtype=np.uint32)
    lib.al_radix_sort_f64(_addr(flat), _addr(b0), _addr(b1), _addr(hist), flat.size)
    stat = np.empty(1, dtype=np.float64)
    lib.al_order_stat(_addr(flat), n_median, _addr(stat))
    return float(np.sqrt(0.5 * stat[0]))


def gaussian_rbf(x, y, sigma=None, infer_sig: bool = False) -> np.ndarray:
    x = _f64(x).reshape(x.shape[0], -1)
    y = _f64(y).reshape(y.shape[0], -1)
    dist = squared_pairwise_distance(x, y)
    if infer_sig or sigma is None:
        sigma = infer_sigma(x, y, dist)
    sigmas = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    gamma = 1.0 / (2.0 * sigmas**2)
    return rbf_kernel(dist, gamma)


# --------------------------------------------------------------------------
# alibi.explainers.pd_variance
# --------------------------------------------------------------------------
def pd_variance(pd_values, categorical: bool = False) -> np.ndarray:
    pd_values = _f64(pd_values)
    t, n = pd_values.shape
    out = np.empty(t, dtype=np.float64)
    if categorical:
        lib.al_pd_range(_addr(pd_values), t, n, _addr(out))
    else:
        lib.al_pd_variance(_addr(pd_values), t, n, ctypes.c_double(1.0), _addr(out))
    return out


# --------------------------------------------------------------------------
# alibi.prototypes.protoselect
# --------------------------------------------------------------------------
def protoselect_summarise(kmatrix, y, eps, lambda_penalty, num_prototypes):
    """Greedy prototype selection, mirroring `ProtoSelect.summarise`.

    `kmatrix` is the `NZ x NX` kernel matrix, `y` the integer label of each
    `X` row. Returns `(prototypes, labels, scores)`, where `prototypes` maps a
    relabelled class to the selected indices, as upstream builds it. Selection
    walks the available prototypes in ascending index order, where upstream
    walks a `set`; the argmax is unaffected except under an exact tie, which
    this deterministic order resolves.
    """
    km = _f64(kmatrix)
    nz, nx = km.shape
    labels = np.unique(y)
    y_codes = np.searchsorted(labels, y).astype(np.int32)
    n_lab = len(labels)
    if lambda_penalty is None:
        lambda_penalty = 1.0 / nz

    b = np.zeros((nz, nx), dtype=np.int32)
    bp = np.zeros((n_lab, nx), dtype=np.int32)
    xl = np.zeros((n_lab, nx), dtype=np.int32)
    xl[y_codes, np.arange(nx)] = 1
    dxi = np.zeros((nz, n_lab, nx), dtype=np.int32)
    dnu = np.zeros((nz, n_lab, nx), dtype=np.int32)
    scores = np.zeros((nz, n_lab), dtype=np.float64)
    lib.al_protoselect_masks(
        _addr(km), _addr(bp), _addr(xl), n_lab, nz, nx,
        ctypes.c_double(eps), ctypes.c_double(lambda_penalty),
        _addr(b), _addr(dxi), _addr(dnu), _addr(scores),
    )

    protos = {l: [] for l in range(n_lab)}
    available = list(range(nz))
    j_buf = np.zeros(max(nz, 1), dtype=np.int32)
    pick = np.zeros(3, dtype=np.int32)
    for _ in range(min(num_prototypes, nz)):
        if not available:
            break
        j_buf[: len(available)] = available
        lib.al_protoselect_argmax(
            _addr(scores), _addr(j_buf), len(available), n_lab, _addr(pick)
        )
        if pick[2] == 0:
            break
        col = int(pick[1])
        i = available[int(pick[0])]
        protos[col].append(i)
        lib.al_protoselect_update(
            _addr(dxi), _addr(b), nz, n_lab, nx, col, i, _addr(scores)
        )
        bp[col] = np.maximum(bp[col], b[i])
        available.remove(i)
    return protos, labels, scores


# --------------------------------------------------------------------------
# alibi.explainers.permutation_importance
# --------------------------------------------------------------------------
def perm_gather(x, perm) -> np.ndarray:
    x = _f64(x)
    perm = _i32(perm)
    out = np.empty_like(x)
    lib.al_perm_gather(_addr(x), _addr(perm), x.shape[0], x.shape[1], _addr(out))
    return out


def feature_swap(x, f0, f1, mid) -> np.ndarray:
    x = _f64(x)
    lib.al_feature_swap(_addr(x), mid, x.shape[1], f0, f1)
    return x


def exact_gather(x, f0, f1, skip) -> np.ndarray:
    x = _f64(x)
    n = x.shape[0]
    out = np.empty((n - 1, f1 - f0), dtype=np.float64)
    lib.al_exact_gather(_addr(x), n, x.shape[1], f0, f1, skip, _addr(out))
    return out


def class_metrics(y, proba, sample_weight=None) -> dict:
    y = _i32(y).reshape(-1)
    proba = _f64(proba)
    n, n_class = proba.shape
    w = _f64(sample_weight).reshape(-1) if sample_weight is not None else None
    out = np.zeros(4, dtype=np.float64)
    lib.al_class_metrics(
        _addr(y), _addr(proba), _addr(w) if w is not None else 0, n, n_class, _addr(out)
    )
    return {"log_loss": out[0], "brier": out[1], "accuracy": out[2], "zero_one": out[3]}


def reg_metrics(y, yhat, sample_weight=None) -> dict:
    y = _f64(y).reshape(-1)
    yhat = _f64(yhat).reshape(-1)
    w = _f64(sample_weight).reshape(-1) if sample_weight is not None else None
    out = np.zeros(4, dtype=np.float64)
    lib.al_reg_metrics(
        _addr(y), _addr(yhat), _addr(w) if w is not None else 0, y.size, _addr(out)
    )
    return {"mae": out[0], "mse": out[1], "rmse": out[2], "r2": out[3]}


def permutation_importance_samples(metric_orig, permuted, kind="difference",
                                   lower_is_better=True) -> dict:
    permuted = _f64(permuted).reshape(-1)
    n = permuted.size
    out = np.zeros(n + 2, dtype=np.float64)
    lib.al_importance(
        ctypes.c_double(metric_orig), _addr(permuted), n,
        1 if kind == "ratio" else 0, 1 if lower_is_better else 0, _addr(out),
    )
    return {"samples": out[:n].copy(), "mean": out[n], "std": out[n + 1]}


# --------------------------------------------------------------------------
# alibi.confidence.model_linearity
# --------------------------------------------------------------------------
def superposition(alphas, vecs, shape=()) -> np.ndarray:
    """`alphas @ vecs` over the sample axis: `out[b] = sum_a alphas[a] v[b, a]`."""
    alphas = _f64(alphas).reshape(-1)
    vecs = _f64(vecs)
    na, nb = alphas.size, vecs.shape[0]
    cols = int(np.prod(shape)) if len(shape) else vecs.shape[2] if vecs.ndim > 2 else 1
    out = np.empty((nb, cols), dtype=np.float64)
    lib.al_superpose(
        _addr(alphas), _addr(vecs.reshape(nb, na, cols)), nb, na, cols, _addr(out)
    )
    return out


def row_l2(a) -> np.ndarray:
    a = _f64(a)
    nb, cols = a.shape
    out = np.empty(nb, dtype=np.float64)
    lib.al_row_l2(_addr(a), nb, cols, _addr(out))
    return out


def linearity_score(a, b) -> np.ndarray:
    """Row-wise ``||a - b||_2``, the residual the linearity measure reports."""
    a = _f64(a)
    b = _f64(b)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    nb, cols = a.shape
    out = np.empty(nb, dtype=np.float64)
    lib.al_diff_row_l2(_addr(a), _addr(b), nb, cols, _addr(out))
    return out
