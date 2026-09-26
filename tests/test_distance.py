"""Parity for `alibi.utils.distance`: cityblock_batch, squared_pairwise_distance,
and the MVDM / ABDM pairwise category distance matrices.

`alibi` is not installed in the parity test venv. Where upstream's own test
suite compares against a third party, that is what is used here
(`scipy.spatial.distance.cityblock`); elsewhere the reference is a direct
transcription of the upstream loop nest, which is the same code path a wrong
stride, transposed index or doubled value would break.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cityblock as scipy_cityblock

import mojo_alibi


# ---------------------------------------------------------------- cityblock
@pytest.mark.parametrize("shape", [(1, 1), (5, 3), (64, 17), (257, 1)])
def test_cityblock_matches_scipy(shape):
    """This is exactly upstream's own test: scipy's L1 against the batch form."""
    rng = np.random.default_rng(3)
    X = rng.uniform(-2, 2, shape)
    y = X[rng.integers(shape[0])]
    got = mojo_alibi.cityblock_batch(X, y)
    expect = np.array([scipy_cityblock(x, y) for x in X]).reshape(X.shape[0], -1)
    assert got.shape == expect.shape
    np.testing.assert_allclose(got, expect, rtol=1e-13, atol=1e-14)


def test_cityblock_is_zero_only_for_the_query_row():
    """A kernel with an off-by-one in the row base would zero a different row,
    or leak the neighbouring row's distance into this one."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    for row in range(3):
        got = mojo_alibi.cityblock_batch(X, X[row]).ravel()
        expect = np.array([np.abs(X[k] - X[row]).sum() for k in range(3)])
        assert got[row] == 0.0
        np.testing.assert_allclose(got, expect, rtol=0, atol=0)


def test_cityblock_keeps_upstream_column_shape():
    """Upstream reshapes to `(N, -1)`, so a 2-D input yields an (N, 1) result;
    flattening it in the shim would break every downstream index."""
    X = np.zeros((3, 2))
    assert mojo_alibi.cityblock_batch(X, np.zeros(2)).shape == (3, 1)


def test_cityblock_rejects_feature_mismatch():
    with pytest.raises(ValueError):
        mojo_alibi.cityblock_batch(np.zeros((3, 4)), np.zeros(5))


# ------------------------------------------------------- pairwise distance
def _ref_sqdist(x, y, a_min=1e-7, a_max=1e30):
    x2 = np.sum(x**2, axis=-1, keepdims=True)
    y2 = np.sum(y**2, axis=-1, keepdims=True)
    dist = x2 + np.transpose(y2, (1, 0)) - 2.0 * x @ np.transpose(y, (1, 0))
    return np.clip(dist, a_min=a_min, a_max=a_max)


@pytest.mark.parametrize("shape", [(1, 1, 3), (7, 5, 4), (40, 33, 16), (129, 130, 8)])
def test_sqdist_matches_expansion_form(shape):
    nx, ny, f = shape
    rng = np.random.default_rng(4)
    x = rng.standard_normal((nx, f))
    y = rng.standard_normal((ny, f))
    got = mojo_alibi.squared_pairwise_distance(x, y)
    np.testing.assert_allclose(got, _ref_sqdist(x, y), rtol=1e-9, atol=1e-7)


def test_sqdist_is_symmetric_for_a_self_comparison():
    """The block/chunk fan-out splits rows; a wrong chunk boundary or a
    transposed write breaks symmetry."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((100, 6))
    got = mojo_alibi.squared_pairwise_distance(x, x)
    np.testing.assert_allclose(got, got.T, rtol=0, atol=0)


def test_sqdist_self_distance_is_clipped_to_the_lower_bound():
    """Upstream's `a_min=1e-7` exists precisely so a point's distance to itself
    is not zero. Forgetting the clip shows up only here."""
    x = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.5]])
    got = mojo_alibi.squared_pairwise_distance(x, x)
    assert got[0, 0] == pytest.approx(1e-7, rel=1e-6)
    assert got[1, 1] == pytest.approx(1e-7, rel=1e-6)
    assert got[0, 2] == pytest.approx(0.25, rel=1e-9)


def test_sqdist_known_value():
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    y = np.array([[0.0, 0.0], [1.0, 0.0]])
    got = mojo_alibi.squared_pairwise_distance(x, y)
    np.testing.assert_allclose(got, [[1e-7, 1.0], [25.0, 20.0]], rtol=1e-12, atol=0)


def test_sqdist_chunking_is_transparent():
    """The shim fans out over row blocks; the result must not depend on how the
    rows were split, which a wrong `row1` or an off-by-one step breaks."""
    rng = np.random.default_rng(12)
    x = rng.standard_normal((97, 5))
    y = rng.standard_normal((31, 5))
    a = mojo_alibi.squared_pairwise_distance(x, y)
    import mojo_alibi._lib as _m

    old = _m.THREAD_THRESHOLD
    try:
        _m.THREAD_THRESHOLD = 1 << 40
        b = mojo_alibi.squared_pairwise_distance(x, y)
        _m.THREAD_THRESHOLD = 0
        c = mojo_alibi.squared_pairwise_distance(x, y)
    finally:
        _m.THREAD_THRESHOLD = old
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_sqdist_rejects_non_default_clip():
    with pytest.raises(ValueError):
        mojo_alibi.squared_pairwise_distance(np.zeros((2, 2)), np.zeros((2, 2)), a_min=0.0)


# -------------------------------------------------------------------- MVDM
def _ref_mvdm(col, y, n_cat, n_y, alpha=1):
    p_cond = np.zeros((n_cat, n_y))
    for i in range(n_cat):
        idx = np.where(col == i)[0]
        for i_y in range(n_y):
            p_cond[i, i_y] = np.sum(y[idx] == i_y) / (y[idx].shape[0] + 1e-12)
    d = np.zeros((n_cat, n_cat))
    for i in range(n_cat):
        j = 0
        while j < i:
            d[i, j] = np.sum(np.abs(p_cond[i, :] - p_cond[j, :]) ** alpha)
            j += 1
    return d + d.T


def test_mvdm_matches_upstream_loop():
    rng = np.random.default_rng(6)
    n, n_cat = 400, 7
    X = rng.integers(0, n_cat, (n, 2))
    y = rng.integers(0, 3, n)
    got = mojo_alibi.mvdm(X, y, {0: None})
    np.testing.assert_allclose(got[0], _ref_mvdm(X[:, 0], y, n_cat, 3), rtol=1e-13, atol=1e-15)


def test_mvdm_is_symmetric_with_zero_diagonal():
    """Upstream fills the lower triangle then adds the transpose, so the matrix
    is symmetric, not doubled. A kernel that doubled the value would still be
    symmetric, which is why the magnitude check matters."""
    rng = np.random.default_rng(7)
    n, n_cat = 300, 5
    X = rng.integers(0, n_cat, (n, 1))
    y = rng.integers(0, 4, n)
    d = mojo_alibi.mvdm(X, y, {0: n_cat})[0]
    np.testing.assert_allclose(d, d.T, rtol=0, atol=0)
    np.testing.assert_array_equal(np.diag(d), np.zeros(n_cat))
    single = _ref_mvdm(X[:, 0], y, n_cat, 4)
    assert np.max(d) == pytest.approx(np.max(single), rel=1e-12)
    assert np.max(d) != pytest.approx(2 * np.max(single), rel=1e-6)


def test_mvdm_alpha_two():
    rng = np.random.default_rng(8)
    n, n_cat = 250, 4
    X = rng.integers(0, n_cat, (n, 1))
    y = rng.integers(0, 2, n)
    got = mojo_alibi.mvdm(X, y, {0: n_cat}, alpha=2.0)[0]
    np.testing.assert_allclose(
        got, _ref_mvdm(X[:, 0], y, n_cat, 2, alpha=2.0), rtol=1e-12, atol=1e-14
    )


def test_mvdm_catches_a_duplicate_column():
    """Two columns that encode the same categories must give the same distance
    matrix; a kernel that mixed the category and class axes would not."""
    rng = np.random.default_rng(13)
    n, n_cat = 200, 3
    col = rng.integers(0, n_cat, n)
    y = rng.integers(0, 2, n)
    X = np.column_stack([col, col])
    got = mojo_alibi.mvdm(X, y, {0: n_cat, 1: n_cat})
    np.testing.assert_allclose(got[0], got[1], rtol=1e-14, atol=0)


# -------------------------------------------------------------------- ABDM
def _ref_abdm(X, col, cat_vars, cat_vars_bin, eps=1e-12):
    combined = {**cat_vars, **cat_vars_bin}
    n_cat = cat_vars[col]
    X_cat_eq = []
    for i in range(n_cat):
        X_cat_eq.append(X[np.where(X[:, col] == i)[0], :])
    p_cond = []
    for col_t, n_cat_t in combined.items():
        if col == col_t:
            continue
        p_cond_t = np.zeros([n_cat_t, n_cat])
        for i in range(n_cat_t):
            for j, X_cat_j in enumerate(X_cat_eq):
                idx = np.where(X_cat_j[:, col_t] == i)[0]
                p_cond_t[i, j] = len(idx) / (X_cat_j.shape[0] + eps)
        p_cond.append(p_cond_t)
    d = np.zeros([n_cat, n_cat])
    for i in range(n_cat):
        j = 0
        while j < i:
            acc = 0
            for p in p_cond:
                for t in range(p.shape[0]):
                    a, b = p[t, i], p[t, j]
                    acc += a * np.log((a + eps) / (b + eps)) + b * np.log(
                        (b + eps) / (a + eps)
                    )
            d[i, j] = acc
            j += 1
    return d + d.T


def test_abdm_matches_upstream_loop():
    rng = np.random.default_rng(9)
    n, n_cat = 500, 4
    X = np.column_stack([rng.integers(0, n_cat, n), rng.integers(0, 3, n)])
    cat_vars = {0: n_cat, 1: 3}
    got = mojo_alibi.abdm(X, cat_vars)
    for col in (0, 1):
        np.testing.assert_allclose(
            got[col], _ref_abdm(X, col, cat_vars, {}), rtol=1e-11, atol=1e-13
        )


def test_abdm_with_binned_numeric_column():
    """The two columns have different category counts, which is the case that
    catches a kernel assuming a square block."""
    rng = np.random.default_rng(10)
    n, n_cat, n_bin = 600, 3, 5
    X = np.column_stack([rng.integers(0, n_cat, n), rng.integers(0, n_bin, n)])
    cat_vars = {0: n_cat}
    cat_vars_bin = {1: n_bin}
    got = mojo_alibi.abdm(X, cat_vars, cat_vars_bin)
    for col in cat_vars:
        np.testing.assert_allclose(
            got[col], _ref_abdm(X, col, cat_vars, cat_vars_bin), rtol=1e-11, atol=1e-13
        )


def test_abdm_is_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(11)
    n, n_cat = 300, 4
    X = np.column_stack([rng.integers(0, n_cat, n), rng.integers(0, 3, n)])
    d = mojo_alibi.abdm(X, {0: n_cat, 1: 3})[0]
    np.testing.assert_allclose(d, d.T, rtol=0, atol=0)
    np.testing.assert_array_equal(np.diag(d), np.zeros(n_cat))


def test_abdm_uninformative_column_gives_zero():
    """When the other column is constant its conditional distribution is the
    same for every category, so every KL term is a log of one and the matrix is
    exactly zero. Reading the block with transposed indices would not be."""
    rng = np.random.default_rng(14)
    n, n_cat = 200, 3
    col = rng.integers(0, n_cat, n)
    X = np.column_stack([col, np.zeros(n, dtype=np.int64)])
    d = mojo_alibi.abdm(X, {0: n_cat, 1: 2})
    # Identical up to the 1e-12 denominator guard, so every KL term is a log
    # of one to within ~eps^2; a transposed block read would not be this small.
    assert np.max(d[0]) < 1e-20
    np.testing.assert_array_equal(d[0], d[0].T)
    # The constant column has no category left to grow into, so its own matrix
    # is dominated by the eps-regularised divergence, not by zero.
    assert d[1][0, 1] > 20.0
    np.testing.assert_allclose(
        d[1], _ref_abdm(X, 1, {0: n_cat, 1: 2}, {}), rtol=1e-11, atol=1e-13
    )


def test_abdm_is_exactly_symmetric():
    """Only the lower triangle is accumulated and then mirrored, so symmetry
    holds bit-exactly rather than to a tolerance."""
    rng = np.random.default_rng(15)
    n, n_cat = 400, 5
    X = np.column_stack([rng.integers(0, n_cat, n), rng.integers(0, 3, n)])
    d = mojo_alibi.abdm(X, {0: n_cat, 1: 3})[0]
    np.testing.assert_array_equal(d, d.T)
