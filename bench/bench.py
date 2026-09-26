"""Correctness-gated benchmark for mojo-alibi.

Every case checks numerical agreement with the reference formulation before it
times anything, so a regression in the Mojo kernels surfaces as a correctness
failure rather than as a suspiciously good number.

The references are the fastest reasonable NumPy formulations, not Python loops
that NumPy would never use: the expansion form for the pairwise distance, a
single `exp` over the stacked blocks for the RBF kernel, and a broadcast
reduction for the metrics.
"""

from __future__ import annotations

import ctypes
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_alibi  # noqa: E402
from mojo_alibi import _lib  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_kl_bernoulli(n: int = 1 << 22):
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, n)
    q = rng.uniform(0.0, 1.0, n)

    def ref():
        m = np.clip(p, 0.0000001, 0.9999999999999999).astype(float)
        k = np.clip(q, 0.0000001, 0.9999999999999999).astype(float)
        return m * np.log(m / k) + (1.0 - m) * np.log((1.0 - m) / (1.0 - k))

    np.testing.assert_allclose(mojo_alibi.kl_bernoulli(p, q), ref(), rtol=1e-13,
                               atol=1e-15)
    return f"kl_bernoulli n={n}", _time(ref, 3), _time(
        lambda: mojo_alibi.kl_bernoulli(p, q), 3
    )


def bench_sqdist(nx: int = 2048, ny: int = 2048, f: int = 32):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((nx, f))
    y = rng.standard_normal((ny, f))

    def ref():
        x2 = np.sum(x**2, axis=-1, keepdims=True)
        y2 = np.sum(y**2, axis=-1, keepdims=True)
        return np.clip(x2 + y2.T - 2.0 * x @ y.T, 1e-7, 1e30)

    got = mojo_alibi.squared_pairwise_distance(x, y)
    np.testing.assert_allclose(got, ref(), rtol=1e-9, atol=1e-7)
    label = f"sqdist {nx}x{ny}x{f}"
    return label, _time(ref, 3), _time(
        lambda: mojo_alibi.squared_pairwise_distance(x, y), 3
    )


def bench_rbf(n: int = 1024, s: int = 8):
    rng = np.random.default_rng(2)
    dist = rng.random((n, n)) * 4.0
    gamma = rng.random(s) * 2.0

    def ref():
        return np.mean(
            np.exp(-np.concatenate([(g * dist)[None] for g in gamma], axis=0)), axis=0
        )

    np.testing.assert_allclose(mojo_alibi.rbf_kernel(dist, gamma), ref(), rtol=1e-10,
                               atol=1e-13)
    return f"rbf {n}x{n} s={s}", _time(ref, 3), _time(
        lambda: mojo_alibi.rbf_kernel(dist, gamma), 3
    )


def bench_rbf_inferred_sigma(n: int = 768, f: int = 16):
    """End to end: the pairwise distance, the median-sigma inference (which
    needs a full sort) and the kernel matrix. This is the case a caller of
    `GaussianRBF(sigma=None)` actually pays for."""
    rng = np.random.default_rng(9)
    x = rng.standard_normal((n, f))

    def ref():
        x2 = np.sum(x**2, axis=-1, keepdims=True)
        dist = np.clip(x2 + x2.T - 2.0 * x @ x.T, 1e-7, 1e30)
        nn = n if np.all(x[:n] == x[:n]) and x.shape == x.shape else 0
        median = nn + (dist.size - nn) // 2 - 1
        sigma = np.sqrt(0.5 * np.sort(dist.reshape(-1))[median])
        return np.exp(-(1.0 / (2 * sigma**2)) * dist)

    got = mojo_alibi.gaussian_rbf(x, x)
    np.testing.assert_allclose(got, ref(), rtol=1e-10, atol=1e-13)
    return f"rbf + inferred sigma n={n}", _time(ref, 3), _time(
        lambda: mojo_alibi.gaussian_rbf(x, x), 3
    )


def bench_sort(n: int = 1 << 21):
    rng = np.random.default_rng(3)
    a = rng.standard_normal(n)
    np.testing.assert_array_equal(mojo_alibi.sort_f64(a.copy()), np.sort(a))
    return f"sort n={n}", _time(lambda: np.sort(a), 3), _time(
        lambda: mojo_alibi.sort_f64(a.copy()), 3
    )


def bench_cityblock(n: int = 1 << 20, f: int = 16):
    rng = np.random.default_rng(4)
    X = rng.standard_normal((n, f))
    y = rng.standard_normal(f)
    np.testing.assert_allclose(
        mojo_alibi.cityblock_batch(X, y).ravel(), np.abs(X - y).sum(axis=1), rtol=1e-13
    )
    return f"cityblock n={n} f={f}", _time(lambda: np.abs(X - y).sum(axis=1), 3), _time(
        lambda: mojo_alibi.cityblock_batch(X, y), 3
    )


def bench_class_metrics(n: int = 1 << 20, n_class: int = 5):
    from sklearn.metrics import accuracy_score, log_loss

    rng = np.random.default_rng(5)
    proba = rng.random((n, n_class)) + 0.05
    proba /= proba.sum(axis=1, keepdims=True)
    y = rng.integers(0, n_class, n)
    got = mojo_alibi.class_metrics(y, proba)
    assert abs(got["accuracy"] - accuracy_score(y, np.argmax(proba, axis=1))) < 1e-12
    assert abs(got["log_loss"] - log_loss(y, proba, labels=list(range(n_class)))) < 1e-9

    def ref():
        ll = -np.log(proba[np.arange(n), y])
        b = ((proba - np.eye(n_class)[y]) ** 2).sum(axis=1)
        acc = (np.argmax(proba, axis=1) == y).astype(float)
        return ll.mean(), b.mean(), acc.mean()

    return f"class_metrics n={n} c={n_class}", _time(ref, 3), _time(
        lambda: mojo_alibi.class_metrics(y, proba), 3
    )


def _ref_mvdm(X, y, n_cat, n_y, alpha=1):
    """Upstream's `mvdm` end to end, including the per-category scan."""
    p_cond = np.zeros((n_cat, n_y))
    for i in range(n_cat):
        idx = np.where(X[:, 0] == i)[0]
        for k in range(n_y):
            p_cond[i, k] = np.sum(y[idx] == k) / (y[idx].shape[0] + 1e-12)
    d = np.zeros((n_cat, n_cat))
    for i in range(n_cat):
        for j in range(i):
            d[i, j] = np.sum(np.abs(p_cond[i] - p_cond[j]) ** alpha)
    return d + d.T


def bench_mvdm(n: int = 200000, n_cat: int = 12, n_y: int = 6):
    rng = np.random.default_rng(6)
    X = rng.integers(0, n_cat, (n, 1))
    y = rng.integers(0, n_y, n)
    np.testing.assert_allclose(
        mojo_alibi.mvdm(X, y, {0: n_cat})[0], _ref_mvdm(X, y, n_cat, n_y),
        rtol=1e-12, atol=1e-14,
    )
    return f"mvdm n={n} cat={n_cat}", _time(lambda: _ref_mvdm(X, y, n_cat, n_y), 3), _time(
        lambda: mojo_alibi.mvdm(X, y, {0: n_cat}), 3
    )


def bench_protoselect(nz: int = 400, nx: int = 2000, n_lab: int = 4):
    """Clustered data, because an epsilon ball over uniform noise covers every
    class equally and no prototype ever scores positive -- which is upstream's
    stopping criterion, not a kernel bug."""
    rng = np.random.default_rng(7)
    centres = rng.standard_normal((n_lab, 3)) * 8.0
    y = rng.integers(0, n_lab, nx)
    X = centres[y] + rng.standard_normal((nx, 3))
    Z = centres[rng.integers(0, n_lab, nz)] + rng.standard_normal((nz, 3))
    km = ((Z**2).sum(1)[:, None] + (X**2).sum(1)[None, :] - 2 * Z @ X.T) ** 2
    eps, lam = 20.0, 1.0 / nz
    protos, _, _ = mojo_alibi.protoselect_summarise(km, y, eps, lam, 12)
    assert sum(len(v) for v in protos.values()) > 0

    def ref():
        B = (km <= eps).astype(np.int32)
        B_P = np.zeros((n_lab, nx), dtype=np.int32)
        Xl = np.concatenate(
            [(y == l).reshape(1, -1) for l in range(n_lab)], axis=0
        ).astype(np.int32)
        delta_xi = B[:, None, :] - B_P[None, :, :] + Xl[None, ...] >= 2
        delta_nu = B[:, None, :] + (1 - Xl[None, ...]) >= 2
        return delta_xi.sum(-1) - delta_nu.sum(-1) - lam

    mine = mojo_alibi.protoselect_summarise(km, y, eps, lam, 0)[2]
    np.testing.assert_allclose(ref(), mine, rtol=0, atol=0)
    return f"protoselect mask {nz}x{nx}x{n_lab}", _time(ref, 3), _time(
        lambda: mojo_alibi.protoselect_summarise(km, y, eps, lam, 0), 3
    )


def bench_linearity(nb: int = 4096, na: int = 32, cols: int = 16):
    rng = np.random.default_rng(8)
    alphas = np.full(na, 1.0 / na)
    vecs = rng.standard_normal((nb, na, cols))
    a = rng.standard_normal((nb, cols))
    b = rng.standard_normal((nb, cols))
    got = mojo_alibi.linearity_score(a, b)
    np.testing.assert_allclose(got, np.linalg.norm(a - b, axis=1), rtol=1e-13)
    np.testing.assert_allclose(
        mojo_alibi.superposition(alphas, vecs, (cols,)),
        np.einsum("a,bac->bc", alphas, vecs), rtol=1e-12, atol=1e-14,
    )
    return f"linearity {nb}x{cols}", _time(lambda: np.linalg.norm(a - b, axis=1), 3), _time(
        lambda: mojo_alibi.linearity_score(a, b), 3
    )


def main():
    print(f"{'case':<34}{'reference':>12}{'mojo-alibi':>14}{'ratio':>10}")
    print("-" * 70)
    for fn in (
        bench_kl_bernoulli,
        bench_cityblock,
        bench_sqdist,
        bench_rbf,
        bench_rbf_inferred_sigma,
        bench_sort,
        bench_class_metrics,
        bench_mvdm,
        bench_protoselect,
        bench_linearity,
    ):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<34}{ref*1e3:>10.2f}ms{got*1e3:>12.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
