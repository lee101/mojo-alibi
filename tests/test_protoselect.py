"""Parity for the `alibi.prototypes.protoselect.ProtoSelect.summarise` vectorised
scoring and greedy selection.

`alibi` is not in the parity test venv, so the score matrix is compared against
a transcription of upstream's page-8 vectorisation, and the greedy loop is
checked against the invariants that define it: every pick is the argmax of the
current score matrix, the score decrement equals the number of newly covered
instances, and the loop stops exactly when no score is non-negative.
"""

import numpy as np

import mojo_alibi


def _ref_masks(km, y_codes, n_lab, eps, lambda_penalty):
    """Upstream's `delta_xi_all` / `delta_xi_summed` / `scores_all`, in numpy."""
    nz, nx = km.shape
    B = (km <= eps).astype(np.int32)
    B_P = np.zeros((n_lab, nx), dtype=np.int32)
    Xl = np.concatenate(
        [(y_codes == l).reshape(1, -1) for l in range(n_lab)], axis=0
    ).astype(np.int32)
    delta_xi_all = B[:, np.newaxis, :] - B_P[np.newaxis, :, :] + Xl[np.newaxis, ...] >= 2
    delta_xi_summed = np.sum(delta_xi_all, axis=-1)
    delta_nu_summed = np.sum(B[:, np.newaxis, :] + (1 - Xl[np.newaxis, ...]) >= 2, axis=-1)
    scores_all = delta_xi_summed - delta_nu_summed - lambda_penalty
    return B, delta_xi_all, delta_xi_summed, scores_all


def _setup(nz=24, nx=60, n_lab=3, seed=40):
    rng = np.random.default_rng(seed)
    km = rng.random((nz, nx))
    y = rng.integers(0, n_lab, nx)
    return km, y, n_lab


def test_initial_scores_match_upstream_vectorisation():
    km, y, n_lab = _setup()
    eps, lam = 0.3, 1.0 / len(km)
    labels = np.unique(y)
    y_codes = np.searchsorted(labels, y).astype(np.int32)
    *_, scores = _ref_masks(km, y_codes, n_lab, eps, lam)
    *_, mine = mojo_alibi.protoselect_summarise(km, y, eps, lam, 0)
    np.testing.assert_allclose(mine, scores, rtol=0, atol=0)


def test_scores_are_integer_counts_minus_the_penalty():
    """The running sums are integer instance counts, so the fractional part of
    every score is exactly `-lambda_penalty`."""
    km, y, n_lab = _setup()
    eps, lam = 0.3, 0.25
    *_, mine = mojo_alibi.protoselect_summarise(km, y, eps, lam, 0)
    np.testing.assert_allclose(mine + lam, np.round(mine + lam), rtol=0, atol=1e-12)


def test_a_wider_epsilon_ball_covers_more():
    """With a single class there is nothing for `delta_nu` to avoid, so the
    score is exactly the coverage count minus the penalty and must be
    non-decreasing in eps. An inverted mask comparison or a `>` instead of `<=`
    breaks the ordering."""
    rng = np.random.default_rng(48)
    km = rng.random((20, 50))
    y = np.zeros(50, dtype=np.int64)
    *_, small = mojo_alibi.protoselect_summarise(km, y, 0.1, 0.0, 0)
    *_, large = mojo_alibi.protoselect_summarise(km, y, 0.9, 0.0, 0)
    assert np.all(large >= small)
    assert np.max(large) > np.max(small)


def test_penalty_shifts_every_score_by_the_same_amount():
    km, y, n_lab = _setup()
    *_, a = mojo_alibi.protoselect_summarise(km, y, 0.3, 0.0, 0)
    *_, b = mojo_alibi.protoselect_summarise(km, y, 0.3, 0.1, 0)
    np.testing.assert_allclose(a - b, 0.1, rtol=0, atol=1e-12)


def _replay(km, y_codes, n_lab, eps, lam, k):
    """The upstream greedy loop, in numpy, returning the picks in order."""
    B, delta_xi_all, delta_xi_summed, scores = _ref_masks(km, y_codes, n_lab, eps, lam)
    available = list(range(km.shape[0]))
    picks = []
    for _ in range(k):
        if not available:
            break
        live = scores[available]
        if np.all(live < 0):
            break
        row, col = np.unravel_index(np.argmax(live), live.shape)
        i = available[int(row)]
        picks.append((i, int(col)))
        covered = np.sum(delta_xi_all[:, int(col), B[i].astype(bool)], axis=-1)
        delta_xi_all[:, int(col), B[i].astype(bool)] = 0
        delta_xi_summed[:, int(col)] -= covered
        scores[:, int(col)] -= covered
        available.remove(i)
    return picks


def test_greedy_picks_match_the_upstream_loop():
    """Replaying the whole selection in numpy pins the score updates, the label
    bookkeeping and the argmax order; a stale score, a wrong label or an
    off-by-one in the update all change the pick list."""
    km, y, n_lab = _setup(nz=20, nx=50, n_lab=2, seed=41)
    eps, lam, k = 0.25, 0.05, 5
    protos, labels, _ = mojo_alibi.protoselect_summarise(km, y, eps, lam, k)
    y_codes = np.searchsorted(labels, y).astype(np.int32)
    picks = _replay(km, y_codes, n_lab, eps, lam, k)
    flat = [(i, l) for l, v in protos.items() for i in v]
    assert sorted(flat) == sorted(picks)
    assert len(flat) == len(picks)


def test_every_pick_adds_new_coverage():
    """A pick is only made when `new_xi - new_nu - lambda >= 0`, so with a
    penalty above zero it must bring at least one new instance of its class.
    An update that failed to retire the covered entries would let the same
    instance be counted over and over and inflate every later score."""
    km, y, n_lab = _setup(nz=22, nx=64, n_lab=2, seed=46)
    eps, lam, k = 0.2, 0.5, 6
    protos, labels, _ = mojo_alibi.protoselect_summarise(km, y, eps, lam, k)
    y_codes = np.searchsorted(labels, y).astype(np.int32)
    B = (km <= eps).astype(np.int32)
    covered = set()
    for l, chosen in sorted(protos.items()):
        for i in chosen:
            new = {k2 for k2 in range(km.shape[1])
                   if B[i, k2] and y_codes[k2] == l and k2 not in covered}
            assert new, f"prototype {i} for class {l} added no new coverage"
            covered |= new
    assert covered


def test_each_prototype_is_selected_at_most_once():
    km, y, n_lab = _setup(nz=18, nx=44, n_lab=3, seed=42)
    protos, _, _ = mojo_alibi.protoselect_summarise(km, y, 0.3, 0.05, 12)
    flat = [i for v in protos.values() for i in v]
    assert len(flat) == len(set(flat))
    assert len(flat) <= 12


def test_selection_stops_when_nothing_scores_positive():
    """A huge penalty makes every score negative, which is upstream's stopping
    criterion; a kernel that ignored the sign would keep selecting."""
    km, y, n_lab = _setup(nz=15, nx=40, n_lab=2, seed=43)
    protos, _, scores = mojo_alibi.protoselect_summarise(km, y, 0.3, 1e6, 5)
    assert sum(len(v) for v in protos.values()) == 0
    assert np.all(scores < 0)


def test_selection_is_deterministic():
    km, y, n_lab = _setup(nz=19, nx=55, n_lab=3, seed=47)
    a = mojo_alibi.protoselect_summarise(km, y, 0.3, 0.05, 6)
    b = mojo_alibi.protoselect_summarise(km, y, 0.3, 0.05, 6)
    assert a[0] == b[0]


def test_full_coverage_prototype_is_selected_first():
    """A prototype whose epsilon ball covers every instance of its class scores
    strictly above any alternative, so the greedy search must start there."""
    km = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    y = np.array([0, 0, 0, 1])
    protos, _, _ = mojo_alibi.protoselect_summarise(km, y, 0.0, 0.0, 1)
    assert protos[0] == [0]


def test_prototype_ball_never_covers_a_whole_second_class():
    """`delta_nu` exists precisely to stop a prototype's ball from swallowing
    another class. Rows 0-2 cover all eight instances (4 of each class, score
    4 - 4 = 0); rows 3-5 cover only the four class-0 instances (score 4). The
    greedy search must pick a row-3 ball, not a row-0 one."""
    km = np.ones((9, 8))
    km[0:3, :] = 0.0
    km[3:6, 0:4] = 0.0
    km[6:9, 4:6] = 0.0
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    protos, _, _ = mojo_alibi.protoselect_summarise(km, y, 0.0, 0.0, 1)
    assert sum(len(v) for v in protos.values()) == 1
    chosen = [i for v in protos.values() for i in v][0]
    assert chosen in (3, 4, 5)
    assert 0 in protos


def test_label_relabelling_is_reported():
    """Upstream relabels to `[0, L-1]` and returns the mapping, so a caller can
    recover the original label values."""
    km = np.random.default_rng(45).random((8, 20))
    y = np.array([40] * 10 + [51] * 10)
    protos, labels, _ = mojo_alibi.protoselect_summarise(km, y, 0.3, 0.0, 4)
    np.testing.assert_array_equal(labels, [40, 51])
    assert set(protos) == {0, 1}
