"""Parity for the `alibi.explainers.pd_variance` variance reductions.

Upstream's numerical features are `np.std(pd_values, axis=-1, ddof=1)` for a
numerical feature and `(max - min) / 4` for a categorical one; both are
compared against numpy directly, which is the same expression.
"""

import numpy as np
import pytest

import mojo_alibi


@pytest.mark.parametrize("shape", [(1, 5), (3, 7), (4, 100), (17, 64)])
def test_pd_variance_matches_numpy_std_ddof1(shape):
    t, n = shape
    rng = np.random.default_rng(30)
    pd = rng.standard_normal(shape) * 3.0 + 1.0
    np.testing.assert_allclose(mojo_alibi.pd_variance(pd), np.std(pd, axis=-1, ddof=1),
                               rtol=1e-12, atol=1e-13)


def test_pd_variance_is_zero_for_a_flat_curve():
    """A constant partial dependence curve has no variance. A wrong row stride
    would pick up a neighbouring row and give a non-zero spread here."""
    pd = np.tile(np.array([2.5] * 12), (5, 1))
    assert np.all(mojo_alibi.pd_variance(pd) == 0.0)
    # a ramp is not flat: it must come out at the sample standard deviation
    ramp = np.tile(np.linspace(0.0, 1.0, 12), (5, 1))
    np.testing.assert_allclose(
        mojo_alibi.pd_variance(ramp), np.std(ramp, axis=-1, ddof=1), rtol=1e-13
    )


def test_pd_variance_is_shift_invariant():
    """Standard deviation does not depend on the offset; a kernel that forgot to
    subtract the mean would return the raw second moment and blow up here."""
    rng = np.random.default_rng(31)
    pd = rng.standard_normal((4, 40))
    base = mojo_alibi.pd_variance(pd)
    shifted = mojo_alibi.pd_variance(pd + 1000.0)
    np.testing.assert_allclose(shifted, base, rtol=1e-10, atol=1e-12)


def test_pd_variance_matches_a_hand_computed_row():
    pd = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 2.0]])
    mean = np.array([2.5, 0.5])
    expect = np.sqrt(((pd - mean[:, None]) ** 2).sum(-1) / 3)
    np.testing.assert_allclose(mojo_alibi.pd_variance(pd), expect, rtol=1e-13,
                               atol=1e-15)


@pytest.mark.parametrize("shape", [(1, 4), (3, 9), (6, 50)])
def test_pd_range_matches_numpy_max_min_over_four(shape):
    t, n = shape
    rng = np.random.default_rng(32)
    pd = rng.standard_normal(shape)
    expect = (pd.max(axis=-1) - pd.min(axis=-1)) / 4
    np.testing.assert_array_equal(mojo_alibi.pd_variance(pd, categorical=True), expect)


def test_pd_range_is_zero_for_a_flat_curve():
    pd = np.full((3, 8), 2.5)
    np.testing.assert_array_equal(mojo_alibi.pd_variance(pd, categorical=True),
                                  np.zeros(3))


def test_pd_range_single_point_grid_is_zero():
    """One grid point means max == min, so the range and therefore the
    categorical importance is zero."""
    pd = np.array([[7.0], [-1.0], [0.0]])
    np.testing.assert_array_equal(mojo_alibi.pd_variance(pd, categorical=True),
                                  np.zeros(3))


def test_pd_range_is_monotone_in_spread():
    """Widening every row must not decrease the reported range, and a wrong
    division by `t` instead of `4` would break the ordering across rows."""
    rng = np.random.default_rng(33)
    pd = rng.standard_normal((4, 30))
    wide = mojo_alibi.pd_variance(pd * 3.0, categorical=True)
    narrow = mojo_alibi.pd_variance(pd, categorical=True)
    assert np.all(wide > narrow)
    np.testing.assert_allclose(wide, 3.0 * narrow, rtol=1e-12)
