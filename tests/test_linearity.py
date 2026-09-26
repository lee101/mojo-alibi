"""Parity for the `alibi.confidence.model_linearity` reductions: the linear
superposition and the row-wise L2 norms that produce the linearity score.

`alibi` is not in the parity test venv, so the references are transcriptions of
upstream's `_linear_superposition` einsum and `norm(..., axis=1)`.
"""

import numpy as np
import pytest

import mojo_alibi


def test_superposition_matches_einsum():
    rng = np.random.default_rng(60)
    nb, na, cols = 11, 7, 5
    alphas = rng.random(na)
    vecs = rng.standard_normal((nb, na, cols))
    got = mojo_alibi.superposition(alphas, vecs, (cols,))
    expect = np.einsum("a,bac->bc", alphas, vecs)
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-14)


def test_superposition_with_equal_weights_is_a_mean():
    """`_linearity_measure` defaults to `alphas = 1/nb_samples`; a kernel that
    ignored `alphas` entirely and summed instead of averaging would be off by
    exactly that factor."""
    rng = np.random.default_rng(61)
    nb, na, cols = 4, 4, 3
    alphas = np.full(na, 1.0 / na)
    vecs = rng.standard_normal((nb, na, cols))
    got = mojo_alibi.superposition(alphas, vecs, (cols,))
    np.testing.assert_allclose(got, vecs.mean(axis=1), rtol=1e-12, atol=1e-14)


def test_superposition_of_a_single_sample_is_the_identity():
    """With one sample the superposition coefficient is 1, so the output is the
    input row. A wrong `na` stride breaks this immediately."""
    rng = np.random.default_rng(62)
    vecs = rng.standard_normal((3, 1, 4))
    got = mojo_alibi.superposition(np.array([1.0]), vecs, (4,))
    np.testing.assert_array_equal(got, vecs[:, 0, :])


def test_superposition_with_zero_weights_is_zero():
    rng = np.random.default_rng(63)
    vecs = rng.standard_normal((3, 5, 2))
    got = mojo_alibi.superposition(np.zeros(5), vecs, (2,))
    np.testing.assert_array_equal(got, np.zeros((3, 2)))


def test_row_l2_matches_numpy_norm():
    rng = np.random.default_rng(64)
    a = rng.standard_normal((9, 6))
    np.testing.assert_allclose(mojo_alibi.row_l2(a), np.linalg.norm(a, axis=1),
                               rtol=1e-13, atol=1e-15)


def test_row_l2_of_the_zero_row_is_zero():
    a = np.zeros((3, 4))
    a[1] = 1.0
    np.testing.assert_array_equal(mojo_alibi.row_l2(a), [0.0, 2.0, 0.0])


def test_linearity_score_matches_norm_of_the_difference():
    rng = np.random.default_rng(65)
    a = rng.standard_normal((13, 8))
    b = rng.standard_normal((13, 8))
    np.testing.assert_allclose(mojo_alibi.linearity_score(a, b),
                               np.linalg.norm(a - b, axis=1), rtol=1e-13, atol=1e-15)


def test_linearity_score_is_zero_for_a_linear_model():
    """The defining property of the measure: a model that is genuinely linear
    in its inputs has superposition error zero. This exercises superposition,
    the norms and the difference together, and any sign or axis error in any of
    the three makes it non-zero."""
    rng = np.random.default_rng(66)
    nb, na, f, out_dim = 6, 9, 4, 3
    W = rng.standard_normal((f, out_dim))
    samples = rng.standard_normal((nb, na, f))
    alphas = np.full(na, 1.0 / na)
    # outputs of each individual sample, then of the superposition
    outs = samples @ W
    sum_out = mojo_alibi.superposition(alphas, outs, (out_dim,))
    summ = mojo_alibi.superposition(alphas, samples, (f,))
    out_sum = summ @ W
    score = mojo_alibi.linearity_score(out_sum, sum_out)
    np.testing.assert_allclose(score, np.zeros(nb), rtol=1e-11, atol=1e-13)


def test_linearity_score_is_positive_for_a_non_linear_model():
    """The complementary check: a model of the mean square, which is not linear
    in the inputs, must not score zero. Without this the previous test would
    pass for a kernel that always returned zero."""
    rng = np.random.default_rng(67)
    nb, na, f = 5, 8, 3
    samples = rng.standard_normal((nb, na, f))
    alphas = np.full(na, 1.0 / na)
    outs = np.stack(
        [(samples**2).mean(axis=2), np.exp(samples).mean(axis=2)], axis=2
    )
    sum_out = mojo_alibi.superposition(alphas, outs, (2,))
    summ = mojo_alibi.superposition(alphas, samples, (f,))
    out_sum = np.stack([(summ**2).sum(axis=1), np.exp(summ).sum(axis=1)], axis=1)
    score = mojo_alibi.linearity_score(out_sum, sum_out)
    assert np.all(score > 0.0)


def test_linearity_score_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        mojo_alibi.linearity_score(np.zeros((3, 4)), np.zeros((3, 5)))
