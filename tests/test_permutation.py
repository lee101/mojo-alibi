"""Parity for the `alibi.explainers.permutation_importance` numeric surface.

`alibi` delegates its metrics to scikit-learn, so the reductions are compared
against the very sklearn functions it calls: `log_loss`, `brier_score_loss`,
`accuracy_score`, `mean_absolute_error`, `mean_squared_error` and `r2_score`.
The data movement -- the permuted gather, the half swap and the leave-one-out
construction -- is compared against the numpy expressions upstream writes.
"""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import mojo_alibi


def _proba(rng, n, n_class):
    p = rng.random((n, n_class)) + 0.05
    return p / p.sum(axis=1, keepdims=True)


# ------------------------------------------------------- classification
@pytest.mark.parametrize("n,n_class", [(1, 2), (5, 2), (200, 3), (64, 5)])
def test_class_metrics_log_loss_and_accuracy_match_sklearn(n, n_class):
    rng = np.random.default_rng(50 + n)
    y = rng.integers(0, n_class, n)
    proba = _proba(rng, n, n_class)
    got = mojo_alibi.class_metrics(y, proba)
    assert got["log_loss"] == pytest.approx(
        log_loss(y, proba, labels=list(range(n_class))), rel=1e-11
    )
    assert got["accuracy"] == pytest.approx(
        accuracy_score(y, np.argmax(proba, axis=1)), rel=0, abs=0
    )
    assert got["zero_one"] == pytest.approx(1.0 - got["accuracy"], rel=0, abs=1e-15)
    assert 0.0 <= got["accuracy"] <= 1.0


def test_brier_is_twice_the_sklearn_binary_score():
    """sklearn's `brier_score_loss` is `mean((p_pos - y)^2)`, which is half the
    multiclass sum over classes. The kernel reports the general form, so the two
    differ by exactly the number of classes; a kernel that forgot to sum over
    the class axis would land on sklearn's value by accident."""
    rng = np.random.default_rng(67)
    n, n_class = 40, 2
    y = rng.integers(0, n_class, n)
    proba = _proba(rng, n, n_class)
    got = mojo_alibi.class_metrics(y, proba)["brier"]
    sklearn_brier = brier_score_loss(y, proba[:, 1])
    assert got == pytest.approx(2.0 * sklearn_brier, rel=1e-12)
    onehot = np.eye(n_class)[y]
    expect = np.mean(np.sum((proba - onehot) ** 2, axis=1))
    assert got == pytest.approx(expect, rel=1e-13)


def test_brier_uses_all_classes_not_just_the_true_one():
    """The Brier term sums over every class; a kernel that only looked at column
    `c` would give a different number."""
    rng = np.random.default_rng(52)
    y = rng.integers(0, 3, 30)
    proba = _proba(rng, 30, 3)
    got = mojo_alibi.class_metrics(y, proba)["brier"]
    true_col_only = np.mean((proba[np.arange(30), y] - 1.0) ** 2)
    assert got != pytest.approx(true_col_only, rel=1e-6)
    assert got > true_col_only


def test_class_metrics_on_a_perfect_classifier():
    y = np.array([0, 1, 0, 1, 1, 0])
    proba = np.eye(2)[y] * 0.99 + 0.005
    got = mojo_alibi.class_metrics(y, proba)
    assert got["accuracy"] == 1.0
    assert got["zero_one"] == 0.0
    assert got["log_loss"] < 0.1
    assert got["brier"] < 0.01


def test_class_metrics_log_loss_diverges_on_a_wrong_confident_call():
    """Predicting the wrong class at probability 1e-30 must blow the log loss up
    to about -log(1e-15), the kernel's floor. A kernel that clamped the wrong
    way, or skipped the floor, would not."""
    y = np.zeros(4, dtype=np.int32)
    proba = np.tile([1e-30, 1.0], (4, 1))
    got = mojo_alibi.class_metrics(y, proba)["log_loss"]
    assert got > 30.0
    assert got == pytest.approx(-np.log(1e-15), rel=1e-12)


def test_class_metrics_honour_sample_weights():
    rng = np.random.default_rng(51)
    n, n_class = 40, 3
    y = rng.integers(0, n_class, n)
    proba = _proba(rng, n, n_class)
    w = rng.random(n) + 0.1
    got = mojo_alibi.class_metrics(y, proba, w)
    assert got["accuracy"] == pytest.approx(
        accuracy_score(y, np.argmax(proba, axis=1), sample_weight=w), rel=0, abs=1e-15
    )
    assert got["zero_one"] == pytest.approx(
        1.0 - accuracy_score(y, np.argmax(proba, axis=1), sample_weight=w), rel=0, abs=1e-15
    )
    plain = mojo_alibi.class_metrics(y, proba)
    assert got["accuracy"] != pytest.approx(plain["accuracy"])


# ---------------------------------------------------------- regression
@pytest.mark.parametrize("n", [1, 4, 100, 257])
def test_reg_metrics_match_sklearn(n):
    rng = np.random.default_rng(53 + n)
    y = rng.standard_normal(n)
    yhat = y + rng.standard_normal(n) * 0.3
    got = mojo_alibi.reg_metrics(y, yhat)
    assert got["mae"] == pytest.approx(mean_absolute_error(y, yhat), rel=1e-13)
    assert got["mse"] == pytest.approx(mean_squared_error(y, yhat), rel=1e-13)
    assert got["rmse"] == pytest.approx(np.sqrt(got["mse"]), rel=1e-15)
    if n >= 2:
        assert got["r2"] == pytest.approx(r2_score(y, yhat), rel=1e-11, abs=1e-13)
    else:
        # A single sample has no variance to explain; upstream's r2_score is
        # undefined there and the kernel reports 0.
        assert got["r2"] == 0.0


def test_reg_metrics_of_a_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    got = mojo_alibi.reg_metrics(y, y)
    assert got["mae"] == 0.0
    assert got["mse"] == 0.0
    assert got["r2"] == pytest.approx(1.0, rel=0, abs=1e-15)


def test_reg_metrics_r2_is_zero_at_the_mean():
    """R^2 of a prediction equal to the sample mean is 0 by definition; swapping
    the residual and total sums would make it 1."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    got = mojo_alibi.reg_metrics(y, np.full(5, y.mean()))
    assert got["r2"] == pytest.approx(0.0, abs=1e-15)


def test_reg_metrics_honour_sample_weights():
    rng = np.random.default_rng(54)
    y = rng.standard_normal(50)
    yhat = y + rng.standard_normal(50) * 0.2
    w = rng.random(50) + 0.1
    got = mojo_alibi.reg_metrics(y, yhat, w)
    assert got["mae"] == pytest.approx(
        mean_absolute_error(y, yhat, sample_weight=w), rel=1e-13
    )
    assert got["r2"] == pytest.approx(r2_score(y, yhat, sample_weight=w), rel=1e-11)


# ------------------------------------------------------ data movement
def test_perm_gather_matches_fancy_indexing():
    rng = np.random.default_rng(55)
    x = rng.standard_normal((37, 5))
    perm = rng.permutation(37)
    np.testing.assert_array_equal(mojo_alibi.perm_gather(x, perm), x[perm])


def test_perm_gather_identity_is_a_copy():
    x = np.arange(12, dtype=np.float64).reshape(4, 3)
    out = mojo_alibi.perm_gather(x, np.arange(4))
    np.testing.assert_array_equal(out, x)
    out[0, 0] = -1.0
    assert x[0, 0] == 0.0


def test_feature_swap_matches_the_numpy_half_swap():
    """Upstream copies the chosen feature columns out of the second half, writes
    them into the first half, and writes the first half's values back. A wrong
    `mid`, a wrong column width or a transposed index breaks this."""
    rng = np.random.default_rng(56)
    n, f = 40, 6
    x = rng.standard_normal((n, f))
    mid = n // 2
    got = mojo_alibi.feature_swap(x.copy(), 2, 4, mid)
    expect = x.copy()
    expect[0:mid, 2:4], expect[mid : 2 * mid, 2:4] = (
        x[mid : 2 * mid, 2:4].copy(),
        x[0:mid, 2:4].copy(),
    )
    np.testing.assert_array_equal(got, expect)


def test_feature_swap_leaves_other_columns_alone():
    rng = np.random.default_rng(57)
    x = rng.standard_normal((20, 5))
    got = mojo_alibi.feature_swap(x.copy(), 1, 3, 10)
    np.testing.assert_array_equal(got[:, [0, 3, 4]], x[:, [0, 3, 4]])
    assert not np.array_equal(got[:, 1:3], x[:, 1:3])


def test_feature_swap_of_the_full_range_is_a_plain_half_swap():
    rng = np.random.default_rng(68)
    n, f = 20, 4
    x = rng.standard_normal((n, f))
    got = mojo_alibi.feature_swap(x.copy(), 0, f, n // 2)
    expect = np.concatenate([x[n // 2 :], x[: n // 2]])
    np.testing.assert_array_equal(got, expect)


@pytest.mark.parametrize("skip", [0, 1, 5, 8])
def test_exact_gather_matches_the_leave_one_out_construction(skip):
    rng = np.random.default_rng(58)
    n, f = 9, 4
    x = rng.standard_normal((n, f))
    got = mojo_alibi.exact_gather(x, 1, 3, skip)
    keep = [i for i in range(n) if i != skip]
    expect = x[keep][:, 1:3].copy()
    if skip <= n - 2:
        expect[skip] = x[skip, 1:3]
    assert got.shape == (n - 1, 2)
    np.testing.assert_array_equal(got, expect)


def test_exact_gather_selects_the_requested_columns():
    x = np.arange(30, dtype=np.float64).reshape(6, 5)
    got = mojo_alibi.exact_gather(x, 0, 5, 2)
    expect = np.delete(x, 2, axis=0)
    expect[2] = x[2]
    np.testing.assert_array_equal(got, expect)


def test_exact_gather_tiles_only_the_skipped_row():
    """Only output row `skip` is the repeated row; every other row is a distinct
    source row. A kernel that tiled the wrong row would duplicate data."""
    x = np.arange(40, dtype=np.float64).reshape(8, 5)
    got = mojo_alibi.exact_gather(x, 1, 4, 3)
    np.testing.assert_array_equal(got[3], x[3, 1:4])
    np.testing.assert_array_equal(got[0], x[0, 1:4])
    np.testing.assert_array_equal(got[4], x[5, 1:4])


# --------------------------------------------------------- aggregation
def test_importance_difference_for_a_loss():
    orig, permuted = 0.4, np.array([0.5, 0.7, 0.3])
    got = mojo_alibi.permutation_importance_samples(orig, permuted, "difference", True)
    np.testing.assert_allclose(got["samples"], permuted - orig, rtol=0, atol=0)
    assert got["mean"] == pytest.approx((permuted - orig).mean(), rel=1e-14)
    assert got["std"] == pytest.approx((permuted - orig).std(), rel=1e-13)


def test_importance_ratio_for_a_loss():
    orig, permuted = 0.4, np.array([0.5, 0.7, 0.3])
    got = mojo_alibi.permutation_importance_samples(orig, permuted, "ratio", True)
    np.testing.assert_allclose(got["samples"], permuted / orig, rtol=1e-15)


def test_importance_flips_sign_for_a_higher_is_better_score():
    """A score inverts the sign of the difference: a kernel that ignored
    `lower_is_better` would return the loss direction for a score."""
    orig, permuted = 0.9, np.array([0.8, 0.7])
    loss = mojo_alibi.permutation_importance_samples(orig, permuted, "difference", True)
    score = mojo_alibi.permutation_importance_samples(
        orig, permuted, "difference", False
    )
    np.testing.assert_allclose(score["samples"], -loss["samples"], rtol=0, atol=0)
    assert loss["mean"] == pytest.approx(-0.15, rel=1e-14)
    assert score["mean"] == pytest.approx(0.15, rel=1e-14)
    np.testing.assert_allclose(score["std"], loss["std"], rtol=0, atol=0)


def test_importance_ratio_inverts_for_a_score():
    orig, permuted = 0.9, np.array([0.45, 0.3])
    loss = mojo_alibi.permutation_importance_samples(orig, permuted, "ratio", True)
    score = mojo_alibi.permutation_importance_samples(orig, permuted, "ratio", False)
    np.testing.assert_allclose(score["samples"], 1.0 / loss["samples"], rtol=1e-15)


def test_importance_mean_and_std_are_population_statistics():
    """Upstream aggregates with `np.mean` / `np.std`, i.e. ddof=0. A kernel using
    the sample standard deviation would be sqrt(n/(n-1)) too large."""
    rng = np.random.default_rng(59)
    permuted = rng.random(8) + 1.0
    got = mojo_alibi.permutation_importance_samples(1.0, permuted, "difference", True)
    assert got["std"] == pytest.approx(np.std(permuted - 1.0), rel=1e-13)
    assert got["std"] != pytest.approx(np.std(permuted - 1.0, ddof=1), rel=1e-6)


def test_importance_of_a_constant_metric_has_no_spread():
    got = mojo_alibi.permutation_importance_samples(0.5, np.full(5, 0.5))
    assert got["std"] == 0.0
    assert got["mean"] == 0.0
    np.testing.assert_array_equal(got["samples"], np.zeros(5))
