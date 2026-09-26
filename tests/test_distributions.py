"""Parity for `alibi.utils.distributions.kl_bernoulli`.

`alibi` itself is not in the parity test venv, so the reference here is a
direct transcription of the upstream expression plus the closed-form property
that defines the divergence. The clipping bounds are upstream's, so the
boundary behaviour is checked explicitly.

The kernel routes `log` to libm through `external_call`, because
`std.math.log` on float64 is only good to 3.4e-9 relative; with libm the
tolerances below are tight enough to catch a wrong clip or a swapped operand.
"""

import numpy as np
import pytest

import mojo_alibi


def _ref(p, q):
    m = np.clip(p, 0.0000001, 0.9999999999999999).astype(float)
    n = np.clip(q, 0.0000001, 0.9999999999999999).astype(float)
    return m * np.log(m / n) + (1.0 - m) * np.log((1.0 - m) / (1.0 - n))


def test_matches_upstream_expression():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 4096)
    q = rng.uniform(0.01, 0.99, 4096)
    np.testing.assert_allclose(mojo_alibi.kl_bernoulli(p, q), _ref(p, q), rtol=1e-14, atol=1e-16)


def test_zero_when_identical():
    """A plausible bug here -- reading the wrong buffer, or a wrong clip -- would
    make this non-zero for some element."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.0, 1.0, 1024)
    got = mojo_alibi.kl_bernoulli(p, p)
    assert np.all(got == 0.0)


def test_non_negative_and_zero_only_on_equality():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.0, 1.0, 2048)
    q = rng.uniform(0.0, 1.0, 2048)
    got = mojo_alibi.kl_bernoulli(p, q)
    assert np.all(got >= 0.0)
    assert np.all(got[p != q] > 0.0)


def test_lower_clip_bound_is_applied():
    """p = 0 must not produce inf: upstream clips to 1e-7, and the resulting
    value is finite and matches the hand-computed clipped form."""
    q = np.full(1, 0.5)
    got = mojo_alibi.kl_bernoulli(np.array([0.0]), q)
    m = 1e-7
    expect = m * np.log(m / 0.5) + (1 - m) * np.log((1 - m) / 0.5)
    assert np.isfinite(got[0])
    np.testing.assert_allclose(got, [expect], rtol=1e-14, atol=0.0)


def test_upper_clip_bound_is_applied():
    """p = 1 must be clipped to 0.9999999999999999, not left at 1.0; the two
    differ in the last ulps of the log, which a dropped clip makes visible."""
    q = np.full(1, 0.25)
    got = mojo_alibi.kl_bernoulli(np.array([1.0]), q)
    m = 0.9999999999999999
    expect = m * np.log(m / 0.25) + (1 - m) * np.log((1 - m) / 0.25)
    np.testing.assert_allclose(got, [expect], rtol=1e-15, atol=0.0)
    unclipped = np.log(1 / 0.25)
    assert got[0] != unclipped


def test_asymmetry_is_preserved():
    """KL(p||q) != KL(q||p). A kernel that swapped the operands, or averaged
    them, would pass every other test in this file."""
    p = np.array([0.2, 0.7, 0.05])
    q = np.array([0.6, 0.3, 0.4])
    assert not np.allclose(mojo_alibi.kl_bernoulli(p, q), mojo_alibi.kl_bernoulli(q, p))


def test_non_contiguous_input_matches_contiguous():
    """The shim owns the arrays and must copy a strided view before handing an
    address across the ABI; skipping that read the wrong elements."""
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.95, (16, 8))
    q = rng.uniform(0.05, 0.95, (16, 8))
    strided = mojo_alibi.kl_bernoulli(p[:, ::2], q[:, ::2])
    dense = mojo_alibi.kl_bernoulli(
        np.ascontiguousarray(p[:, ::2]), np.ascontiguousarray(q[:, ::2])
    )
    np.testing.assert_array_equal(strided, dense)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        mojo_alibi.kl_bernoulli(np.zeros(3), np.zeros(4))
