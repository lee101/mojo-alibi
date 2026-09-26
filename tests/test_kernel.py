"""Parity for `alibi.utils.kernel.GaussianRBF`: the distance matrix, the
median-sigma inference and the mean-over-sigmas RBF kernel matrix.

`alibi` is not in the parity test venv, so the references are transcriptions of
the upstream expressions; the sort is compared against `np.sort` exactly,
because a comparison sort does no arithmetic and must be bit-identical.
"""

import numpy as np
import pytest

import mojo_alibi


def _ref_rbf(dist, gamma):
    """Upstream: `exp(-concat([(g * dist) for g in gamma])` then mean over axis 0."""
    return np.mean(
        np.exp(-np.concatenate([(g * dist)[None, :, :] for g in gamma], axis=0)), axis=0
    )


def _ref_sigma(x, y, dist):
    n = min(x.shape[0], y.shape[0])
    n = n if np.all(x[:n] == y[:n]) and x.shape == y.shape else 0
    n_median = n + (np.prod(dist.shape) - n) // 2 - 1
    return np.sqrt(0.5 * np.sort(dist.reshape(-1))[n_median])


# ------------------------------------------------------------------- sort
@pytest.mark.parametrize("n", [0, 1, 2, 3, 17, 64, 65, 1000, 4097])
def test_sort_is_bit_exact_against_numpy(n):
    rng = np.random.default_rng(20 + n)
    a = rng.standard_normal(n) if n else np.zeros(0)
    np.testing.assert_array_equal(mojo_alibi.sort_f64(a), np.sort(a))


def test_sort_handles_duplicates_and_extremes():
    a = np.array([3.0, -1.0, 3.0, 0.0, 1e308, -1e308, 3.0, 5e-324, 1e-320])
    np.testing.assert_array_equal(mojo_alibi.sort_f64(a), np.sort(a))


def test_sort_orders_signed_zero_and_infinities():
    """The radix key is the order-preserving bit transform, so -0.0 sorts
    strictly before +0.0; the values are otherwise in exact float order."""
    a = np.array([np.inf, 0.0, -np.inf, -0.0, 1.0, -1.0])
    got = mojo_alibi.sort_f64(a)
    np.testing.assert_array_equal(got[:-1], [-np.inf, -1.0, -0.0, 0.0, 1.0])
    assert got[-1] == np.inf


def test_sort_moves_the_reference_array_in_place():
    a = np.array([5.0, 1.0, 3.0])
    out = mojo_alibi.sort_f64(a)
    np.testing.assert_array_equal(a, [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(out, a)


# ------------------------------------------------------- sigma inference
def test_sigma_inference_matches_upstream_self_comparison():
    """With `x is y`, upstream skips the first `n` entries of the distance
    matrix, which are the clipped self-distances, so the median is taken over
    the off-diagonal part."""
    rng = np.random.default_rng(21)
    x = rng.standard_normal((40, 5))
    dist = mojo_alibi.squared_pairwise_distance(x, x)
    assert mojo_alibi.infer_sigma(x, x, dist) == pytest.approx(
        _ref_sigma(x, x, dist), rel=1e-15
    )


def test_sigma_inference_for_a_cross_comparison():
    """Different shapes mean `n = 0`, so the median is over the whole matrix.
    Getting the `n` term wrong moves the index and changes the answer."""
    rng = np.random.default_rng(22)
    x = rng.standard_normal((30, 4))
    y = rng.standard_normal((30, 4))
    dist = mojo_alibi.squared_pairwise_distance(x, y)
    assert mojo_alibi.infer_sigma(x, y, dist) == pytest.approx(
        _ref_sigma(x, y, dist), rel=1e-15
    )
    # Same point set, different object identity but equal values: upstream's
    # `x.shape == y.shape and np.all(x[:n] == y[:n])` still holds.
    y2 = x.copy()
    d2 = mojo_alibi.squared_pairwise_distance(x, y2)
    assert mojo_alibi.infer_sigma(x, y2, d2) == pytest.approx(
        _ref_sigma(x, y2, d2), rel=1e-15
    )


def test_sigma_inference_depends_on_the_shape_test():
    """`n` is non-zero only when the two arrays agree elementwise over the
    overlap; feeding a shifted array must take the `n = 0` branch."""
    rng = np.random.default_rng(23)
    x = rng.standard_normal((20, 3))
    y = x.copy()
    y[3] += 0.5
    d_same = mojo_alibi.squared_pairwise_distance(x, x)
    d_diff = mojo_alibi.squared_pairwise_distance(x, y)
    assert mojo_alibi.infer_sigma(x, y, d_diff) == pytest.approx(
        _ref_sigma(x, y, d_diff), rel=1e-15
    )
    assert mojo_alibi.infer_sigma(x, x, d_same) == pytest.approx(
        _ref_sigma(x, x, d_same), rel=1e-15
    )


# --------------------------------------------------------- RBF kernel
@pytest.mark.parametrize("shape,sigma", [((12, 9), 1.0), ((33, 33), 2.5), ((5, 41), 0.25)])
def test_rbf_kernel_matches_upstream_expression(shape, sigma):
    nx, ny = shape
    rng = np.random.default_rng(24)
    x = rng.standard_normal((nx, 4))
    y = rng.standard_normal((ny, 4))
    got = mojo_alibi.gaussian_rbf(x, y, sigma=sigma)
    dist = mojo_alibi.squared_pairwise_distance(x, y)
    np.testing.assert_allclose(got, _ref_rbf(dist, [1.0 / (2 * sigma**2)]), rtol=1e-10,
                               atol=1e-13)


def test_rbf_averages_over_several_sigmas():
    rng = np.random.default_rng(25)
    x = rng.standard_normal((16, 3))
    y = rng.standard_normal((11, 3))
    sigmas = np.array([0.5, 1.0, 3.0])
    got = mojo_alibi.gaussian_rbf(x, y, sigma=sigmas)
    dist = mojo_alibi.squared_pairwise_distance(x, y)
    gamma = 1.0 / (2 * sigmas**2)
    np.testing.assert_allclose(got, _ref_rbf(dist, gamma), rtol=1e-10, atol=1e-13)
    # A mean over sigmas must differ from any single one, otherwise the
    # averaging loop is not running.
    for s in sigmas:
        assert not np.allclose(got, _ref_rbf(dist, [1.0 / (2 * s**2)]))


def test_rbf_is_bounded_and_peaks_on_the_diagonal():
    rng = np.random.default_rng(26)
    x = rng.standard_normal((25, 4))
    got = mojo_alibi.gaussian_rbf(x, x, sigma=1.0)
    assert got.max() <= 1.0 + 1e-12
    assert got.min() > 0.0
    off = got[~np.eye(25, dtype=bool)]
    assert off.max() < got[np.diag_indices(25)].min()


def test_rbf_with_inferred_sigma_matches_upstream():
    rng = np.random.default_rng(27)
    x = rng.standard_normal((30, 4))
    y = rng.standard_normal((18, 4))
    got = mojo_alibi.gaussian_rbf(x, y, infer_sig=True)
    dist = mojo_alibi.squared_pairwise_distance(x, y)
    sigma = _ref_sigma(x, y, dist)
    np.testing.assert_allclose(got, _ref_rbf(dist, [1.0 / (2 * sigma**2)]),
                               rtol=1e-10, atol=1e-13)


def test_rbf_infers_sigma_when_none_is_given():
    """`GaussianRBF(sigma=None)` sets `init_required`, so the first call infers
    it; a shim that skipped the inference would return a different matrix."""
    rng = np.random.default_rng(28)
    x = rng.standard_normal((20, 3))
    got = mojo_alibi.gaussian_rbf(x, x)
    dist = mojo_alibi.squared_pairwise_distance(x, x)
    sigma = _ref_sigma(x, x, dist)
    np.testing.assert_allclose(got, _ref_rbf(dist, [1.0 / (2 * sigma**2)]),
                               rtol=1e-10, atol=1e-13)


def test_infer_sigma_does_not_disturb_the_callers_distance_matrix():
    """The median is taken from a sorted copy. Sorting the caller's matrix in
    place would silently corrupt every downstream use of it, and the kernel
    matrix built from it would be wrong."""
    rng = np.random.default_rng(29)
    x = rng.standard_normal((20, 3))
    y = rng.standard_normal((20, 3))
    dist = mojo_alibi.squared_pairwise_distance(x, y)
    keep = dist.copy()
    mojo_alibi.infer_sigma(x, y, dist)
    np.testing.assert_array_equal(dist, keep)
    # and the kernel matrix still matches the reference afterwards
    sigma = mojo_alibi.infer_sigma(x, y, dist)
    np.testing.assert_allclose(
        mojo_alibi.rbf_kernel(dist, [1.0 / (2 * sigma**2)]),
        _ref_rbf(dist, [1.0 / (2 * sigma**2)]),
        rtol=1e-10,
        atol=1e-13,
    )
