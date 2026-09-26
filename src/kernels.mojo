"""Compiled numeric cores for the Mojo port of alibi.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

The ported surfaces map one-to-one onto numeric modules of the upstream
`alibi` package:

* `alibi.utils.distributions.kl_bernoulli`
* `alibi.utils.distance.cityblock_batch` / `squared_pairwise_distance` /
  `mvdm` / `abdm` pairwise matrices
* `alibi.utils.kernel.GaussianRBF` including the median-sigma inference
* `alibi.explainers.pd_variance` variance reductions
* `alibi.prototypes.protoselect.ProtoSelect.summarise` vectorised scoring
* `alibi.explainers.permutation_importance` permutation and metric reductions
"""

from std.ffi import external_call
from std.memory import bitcast
from std.math import exp, fma, pow, sqrt
from std.sys.info import simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]

# Clip bounds used by `alibi.utils.distance.squared_pairwise_distance`.
comptime DIST_MIN = 1e-7
comptime DIST_MAX = 1e30
# Clip bounds used by `alibi.utils.distributions.kl_bernoulli`.
comptime KL_MIN = 0.0000001
comptime KL_MAX = 0.9999999999999999


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)



@always_inline
def _log(x: Float64) -> Float64:
    """libm `log`, not `std.math.log`.

    `std.math.log` on float64 is a fast polynomial: measured against numpy's
    `log` it is only good to 3.4e-9 relative, which would show up directly in
    the KL and ABDM results. Routing to libm is bit-exact and costs a call per
    element, which is nothing next to the surrounding vector traffic.
    """
    return external_call["log", Float64](x)

@always_inline
def _fswap(p: FPtr, i: Int, j: Int):
    var t = p[unsafe_offset=i]
    p[unsafe_offset=i] = p[unsafe_offset=j]
    p[unsafe_offset=j] = t


@always_inline
def _sift_down(a: FPtr, root: Int, size: Int):
    var r = root
    while True:
        var child = 2 * r + 1
        if child >= size:
            return
        if child + 1 < size and a[unsafe_offset=child] < a[unsafe_offset=child + 1]:
            child += 1
        if a[unsafe_offset=r] < a[unsafe_offset=child]:
            _fswap(a, r, child)
            r = child
        else:
            return


@export("al_sort_f64")
def al_sort_f64(a_addr: Int, n: Int) abi("C"):
    """In-place ascending heapsort of `n` float64 values.

    `GaussianRBF` infers `sigma` from the median of the distance matrix, which
    needs a full order statistic. A comparison sort does no arithmetic, so this
    one is bit-exact.
    """
    var a = fp(a_addr)
    if n <= 1:
        return
    var i = n // 2 - 1
    while i >= 0:
        _sift_down(a, i, n)
        i -= 1
    var last = n - 1
    while last > 0:
        _fswap(a, 0, last)
        _sift_down(a, 0, last)


comptime UPtr = Pointer[UInt64, AnyOrigin[mut=True]]
comptime HPtr = Pointer[UInt32, AnyOrigin[mut=True]]
comptime SIGN_BIT = UInt64(0x8000000000000000)


def up(addr: Int) -> UPtr:
    return UPtr(unsafe_from_address=addr)


@always_inline
def _order_key(u: UInt64) -> UInt64:
    """Map the float64 bit pattern to a UInt64 whose unsigned order is the
    float's signed order: flip every bit of a negative, set the top bit of a
    positive."""
    if (u & SIGN_BIT) != UInt64(0):
        return ~u
    return u | SIGN_BIT


@export("al_radix_sort_f64")
def al_radix_sort_f64(
    a_addr: Int, b0_addr: Int, b1_addr: Int, hist_addr: Int, n: Int
) abi("C"):
    """Ascending LSD radix sort of `n` float64 values.

    `b0` and `b1` are `n`-element `UInt64` scratch buffers and `hist_addr` a
    256-entry `UInt32` histogram, all owned by the caller. Eight stable
    counting passes over the order-preserving 64-bit key make the work linear
    in `n`, which is the right shape for the order statistic `GaussianRBF`
    needs: it reads one element of the result and throws the rest away.
    Eight bits per pass rather than sixteen is deliberate: a 1 KiB histogram
    stays in L1, and measured on this box the 16-bit variant is 1.6x slower
    because every scatter then misses L2.

    Ordering is exact, including for infinities and denormals. NaN payloads are
    not ordered (a negative NaN lands first, where numpy puts every NaN last)
    and `-0.0` sorts before `+0.0`.
    """
    if n <= 1:
        return
    var a = fp(a_addr)
    var b0 = up(b0_addr)
    var b1 = up(b1_addr)
    var hist = HPtr(unsafe_from_address=hist_addr)
    for i in range(n):
        b0[unsafe_offset=i] = bitcast[DType.uint64](a[unsafe_offset=i])
    var src = b0
    var dst = b1
    for k in range(8):
        var sh = UInt64(8 * k)
        for d in range(256):
            hist[unsafe_offset=d] = UInt32(0)
        for i in range(n):
            var d = Int(UInt32((_order_key(src[unsafe_offset=i]) >> sh) & UInt64(0xFF)))
            hist[unsafe_offset=d] += UInt32(1)
        var total = UInt32(0)
        for d in range(256):
            var c = hist[unsafe_offset=d]
            hist[unsafe_offset=d] = total
            total += c
        for i in range(n):
            var v = src[unsafe_offset=i]
            var d = Int(UInt32((_order_key(v) >> sh) & UInt64(0xFF)))
            var pos = Int(hist[unsafe_offset=d])
            dst[unsafe_offset=pos] = v
            hist[unsafe_offset=d] = UInt32(pos + 1)
        var t = src
        src = dst
        dst = t
    for i in range(n):
        a[unsafe_offset=i] = bitcast[DType.float64](src[unsafe_offset=i])



@export("al_kl_bernoulli")
def al_kl_bernoulli(p_addr: Int, q_addr: Int, n: Int, out_addr: Int) abi("C"):
    """Elementwise KL(p || q) for Bernoulli outcomes, with upstream clipping."""
    var p = fp(p_addr)
    var q = fp(q_addr)
    var out = fp(out_addr)
    for i in range(n):
        var m = p[unsafe_offset=i]
        if m < KL_MIN:
            m = KL_MIN
        elif m > KL_MAX:
            m = KL_MAX
        var nq = q[unsafe_offset=i]
        if nq < KL_MIN:
            nq = KL_MIN
        elif nq > KL_MAX:
            nq = KL_MAX
        out[unsafe_offset=i] = m * _log(m / nq) + (1.0 - m) * _log(
            (1.0 - m) / (1.0 - nq)
        )


@export("al_cityblock_batch")
def al_cityblock_batch(
    x_addr: Int, y_addr: Int, n: Int, f: Int, out_addr: Int
) abi("C"):
    """Batch L1 distance: out[i] = sum_k |x[i, k] - y[k]|."""
    var x = fp(x_addr)
    var y = fp(y_addr)
    var out = fp(out_addr)
    for i in range(n):
        var acc = Float64(0.0)
        var base = i * f
        for k in range(f):
            var d = x[unsafe_offset=base + k] - y[unsafe_offset=k]
            if d < 0.0:
                d = -d
            acc += d
        out[unsafe_offset=i] = acc


@export("al_sqdist_rows")
def al_sqdist_rows(
    x_addr: Int, y_addr: Int, ny: Int, f: Int, out_addr: Int, row0: Int, row1: Int
) abi("C"):
    """Rows [row0, row1) of the clipped squared pairwise Euclidean distance.

    Computed from the direct differences rather than alibi's
    ``|x|^2 + |y|^2 - 2 x.y`` expansion: same result, better conditioned. The
    clip bounds are upstream's, and the ``a_min`` bound is load bearing because
    it is what keeps a point's distance to itself off zero.
    """
    var x = fp(x_addr)
    var y = fp(y_addr)
    var out = fp(out_addr)
    for i in range(row0, row1):
        var xb = i * f
        for j in range(ny):
            var yb = j * f
            var acc = Float64(0.0)
            for k in range(f):
                var d = x[unsafe_offset=xb + k] - y[unsafe_offset=yb + k]
                acc = fma(d, d, acc)
            if acc < DIST_MIN:
                acc = DIST_MIN
            elif acc > DIST_MAX:
                acc = DIST_MAX
            out[unsafe_offset=i * ny + j] = acc


@export("al_rbf_kernel")
def al_rbf_kernel(
    dist_addr: Int, gamma_addr: Int, ns: Int, ny: Int, total: Int, out_addr: Int
) abi("C"):
    """Mean-over-sigmas RBF: out[i*ny + j] = mean_s exp(-gamma_s * dist)."""
    var dist = fp(dist_addr)
    var gamma = fp(gamma_addr)
    var out = fp(out_addr)
    for t in range(total):
        var acc = Float64(0.0)
        for s in range(ns):
            acc += exp(-gamma[unsafe_offset=s] * dist[unsafe_offset=t])
        out[unsafe_offset=t] = acc / Float64(ns)


@export("al_order_stat")
def al_order_stat(src_addr: Int, index: Int, out_addr: Int) abi("C"):
    """One order statistic out of a buffer the shim already sorted."""
    var src = fp(src_addr)
    var out = fp(out_addr)
    out[unsafe_offset=0] = src[unsafe_offset=index]


@export("al_mvdm_pairwise")
def al_mvdm_pairwise(
    p_cond_addr: Int, n_cat: Int, n_y: Int, alpha: Float64, out_addr: Int
) abi("C"):
    """Modified value difference measure between categories.

    Upstream fills the strict lower triangle with
    ``sum_y |p(i, y) - p(j, y)| ** alpha`` and then adds the transpose, so the
    result is symmetric with the diagonal at zero.
    """
    var p = fp(p_cond_addr)
    var out = fp(out_addr)
    for i in range(n_cat):
        out[unsafe_offset=i * n_cat + i] = 0.0
        for j in range(n_cat):
            if j == i:
                continue
            var acc = Float64(0.0)
            for t in range(n_y):
                var a = p[unsafe_offset=i * n_y + t]
                var b = p[unsafe_offset=j * n_y + t]
                var d = a - b
                if d < 0.0:
                    d = -d
                acc += pow(d, alpha)
            out[unsafe_offset=i * n_cat + j] = acc


@export("al_abdm_pairwise")
def al_abdm_pairwise(
    p_addr: Int, n_cat: Int, n_terms: Int, eps: Float64, out_addr: Int
) abi("C"):
    """Association-based distance metric between categories.

    `p` is the flattened stack of conditional probability blocks laid out as
    ``[n_terms, n_cat]`` row-major, which is the order upstream's nested loops
    walk. Only the strict lower triangle is accumulated and then mirrored, as
    upstream does, so the result is bit-exactly symmetric: the two terms
    ``a*log(a/b) + b*log(b/a)`` are not individually symmetric under an
    operand swap, and summing them in both orders would leave a 1-ulp skew.
    """
    var p = fp(p_addr)
    var out = fp(out_addr)
    for i in range(n_cat):
        out[unsafe_offset=i * n_cat + i] = 0.0
        for j in range(i):
            var acc = Float64(0.0)
            for t in range(n_terms):
                var a = p[unsafe_offset=t * n_cat + i]
                var b = p[unsafe_offset=t * n_cat + j]
                acc += a * _log((a + eps) / (b + eps)) + b * _log((b + eps) / (a + eps))
            out[unsafe_offset=i * n_cat + j] = acc
            out[unsafe_offset=j * n_cat + i] = acc


@export("al_pd_variance")
def al_pd_variance(
    pd_addr: Int, t: Int, n: Int, ddof: Float64, out_addr: Int
) abi("C"):
    """Row-wise standard deviation of a `T x N` partial dependence tensor.

    This is ``np.std(axis=-1, ddof=1)`` upstream. `ddof` is a runtime argument
    because the reduction is shared with the biased variant; the ported caller
    uses 1.
    """
    var pd = fp(pd_addr)
    var out = fp(out_addr)
    var denom = Float64(n) - ddof
    for r in range(t):
        var base = r * n
        var acc = Float64(0.0)
        for k in range(n):
            acc += pd[unsafe_offset=base + k]
        var mean = acc / Float64(n)
        var acc2 = Float64(0.0)
        for k in range(n):
            var d = pd[unsafe_offset=base + k] - mean
            acc2 += d * d
        out[unsafe_offset=r] = sqrt(acc2 / denom)


@export("al_pd_range")
def al_pd_range(pd_addr: Int, t: Int, n: Int, out_addr: Int) abi("C"):
    """Row-wise (max - min) / 4, the categorical partial dependence variance."""
    var pd = fp(pd_addr)
    var out = fp(out_addr)
    for r in range(t):
        var base = r * n
        var lo = pd[unsafe_offset=base]
        var hi = pd[unsafe_offset=base]
        for k in range(n):
            var v = pd[unsafe_offset=base + k]
            if v < lo:
                lo = v
            elif v > hi:
                hi = v
        out[unsafe_offset=r] = (hi - lo) / 4.0


@export("al_protoselect_masks")
def al_protoselect_masks(
    km_addr: Int, bp_addr: Int, xl_addr: Int, n_lab: Int, nz: Int, nx: Int,
    eps: Float64, lambda_penalty: Float64, b_addr: Int, dxi_addr: Int,
    dnu_addr: Int, scores_addr: Int
) abi("C"):
    """Build the prototype-selection coverage masks and the initial scores.

    `km` is the `NZ x NX` kernel matrix, `bp` the per-label covered mask and
    `xl` the per-label indicator of which `X` row carries that label. The
    `NZ x L x NX` masks follow the paper's page-8 vectorisation verbatim; the
    running sums collapse straight into `scores_addr`, laid out `NZ x L`.
    """
    var km = fp(km_addr)
    var bp = ip(bp_addr)
    var xl = ip(xl_addr)
    var b = ip(b_addr)
    var dxi = ip(dxi_addr)
    var dnu = ip(dnu_addr)
    var scores = fp(scores_addr)
    for i in range(nz):
        for l in range(n_lab):
            var base = (i * n_lab + l) * nx
            var s_xi = Int32(0)
            var s_nu = Int32(0)
            for k in range(nx):
                var bit = Int32(0)
                if km[unsafe_offset=i * nx + k] <= eps:
                    bit = Int32(1)
                b[unsafe_offset=i * nx + k] = bit
                var covered = bp[unsafe_offset=l * nx + k]
                var lab = xl[unsafe_offset=l * nx + k]
                var xi = Int32(0)
                if (bit - covered + lab) >= 2:
                    xi = Int32(1)
                var nu = Int32(0)
                if (bit + (1 - lab)) >= 2:
                    nu = Int32(1)
                dxi[unsafe_offset=base + k] = xi
                dnu[unsafe_offset=base + k] = nu
                s_xi += xi
                s_nu += nu
            scores[unsafe_offset=i * n_lab + l] = (
                Float64(s_xi) - Float64(s_nu) - lambda_penalty
            )


@export("al_protoselect_argmax")
def al_protoselect_argmax(
    scores_addr: Int, j_addr: Int, n_avail: Int, n_lab: Int, out_addr: Int
) abi("C"):
    """Best (row, class) over the still-available prototypes.

    `out[0]` is the row of `j`, `out[1]` the class, and `out[2]` is 1 when the
    search found a non-negative score, which is upstream's stopping criterion
    (``np.all(scores < 0)``).
    """
    var scores = fp(scores_addr)
    var j = ip(j_addr)
    var out = ip(out_addr)
    var best = Float64(-1.0)
    var best_row = Int32(0)
    var best_col = Int32(0)
    for t in range(n_avail):
        var row = Int(j[unsafe_offset=t])
        for c in range(n_lab):
            var s = scores[unsafe_offset=row * n_lab + c]
            if s > best:
                best = s
                best_row = Int32(t)
                best_col = Int32(c)
    out[unsafe_offset=0] = best_row
    out[unsafe_offset=1] = best_col
    if best < 0.0:
        out[unsafe_offset=2] = Int32(0)
    else:
        out[unsafe_offset=2] = Int32(1)


@export("al_protoselect_update")
def al_protoselect_update(
    dxi_addr: Int, b_addr: Int, nz: Int, n_lab: Int, nx: Int, label: Int,
    proto: Int, scores_addr: Int
) abi("C"):
    """Retire the instances of class `label` newly covered by prototype
    `proto`, folding the decrement into `scores_addr`.

    Mirrors ``covered = delta_xi_all[:, l, B[i]].sum()`` followed by
    ``delta_xi_summed[:, l] -= covered; scores_all[:, l] -= covered``,
    including the zeroing of the `delta_xi` entries, which is what makes the
    next iteration's argmax incremental.
    """
    var dxi = ip(dxi_addr)
    var b = ip(b_addr)
    var scores = fp(scores_addr)
    for t in range(nz):
        var base = (t * n_lab + label) * nx
        var covered = Int32(0)
        for k in range(nx):
            if b[unsafe_offset=proto * nx + k] != 0:
                covered += dxi[unsafe_offset=base + k]
        for k in range(nx):
            if b[unsafe_offset=proto * nx + k] != 0:
                dxi[unsafe_offset=base + k] = Int32(0)
        scores[unsafe_offset=t * n_lab + label] -= Float64(covered)


@export("al_perm_gather")
def al_perm_gather(x_addr: Int, perm_addr: Int, n: Int, f: Int, out_addr: Int) abi("C"):
    """Gather rows in permuted order: out[k, :] = x[perm[k], :]."""
    var x = fp(x_addr)
    var perm = ip(perm_addr)
    var out = fp(out_addr)
    for k in range(n):
        var src = Int(perm[unsafe_offset=k]) * f
        var dst = k * f
        for j in range(f):
            out[unsafe_offset=dst + j] = x[unsafe_offset=src + j]


@export("al_feature_swap")
def al_feature_swap(x_addr: Int, mid: Int, f_total: Int, f0: Int, f1: Int) abi("C"):
    """Swap feature columns [f0, f1) between the two halves of the row range.

    `f_total` is the row stride (the full feature count), `f1` the exclusive
    end of the column range. The row extent is implied by `mid` upstream too,
    because the shuffle is only defined on the even-length prefix ``[0:end)``.
    """
    var x = fp(x_addr)
    var width = f1 - f0
    for k in range(mid):
        var lo = k * f_total + f0
        var hi = (k + mid) * f_total + f0
        for c in range(width):
            var t = x[unsafe_offset=lo + c]
            x[unsafe_offset=lo + c] = x[unsafe_offset=hi + c]
            x[unsafe_offset=hi + c] = t


@export("al_class_metrics")
def al_class_metrics(
    y_addr: Int, proba_addr: Int, w_addr: Int, n: Int, n_class: Int, out_addr: Int
) abi("C"):
    """Log loss, Brier, accuracy and 0/1 loss over a probabilistic classifier.

    `y` holds the integer class of each row, `proba` the `N x n_class` score
    matrix, `w` optional non-negative weights (address 0 means uniform). These
    are the sklearn reductions `alibi` calls from its permutation importance.
    """
    var y = ip(y_addr)
    var proba = fp(proba_addr)
    var out = fp(out_addr)
    var has_w = w_addr != 0
    var w = fp(w_addr)
    var wsum = Float64(0.0)
    var ll = Float64(0.0)
    var brier = Float64(0.0)
    var correct = Float64(0.0)
    var zero_one = Float64(0.0)
    for i in range(n):
        var wi = Float64(1.0)
        if has_w:
            wi = w[unsafe_offset=i]
        wsum += wi
        var row = i * n_class
        var c = Int(y[unsafe_offset=i])
        var pc = proba[unsafe_offset=row + c]
        if pc < 1e-15:
            pc = 1e-15
        ll -= wi * _log(pc)
        var best = 0
        var best_v = proba[unsafe_offset=row]
        for k in range(1, n_class):
            var v = proba[unsafe_offset=row + k]
            if v > best_v:
                best_v = v
                best = k
        if best == c:
            correct += wi
        else:
            zero_one += wi
        var b = Float64(0.0)
        for k in range(n_class):
            var t = proba[unsafe_offset=row + k]
            if k == c:
                t -= 1.0
            b += t * t
        brier += wi * b
    out[unsafe_offset=0] = ll / wsum
    out[unsafe_offset=1] = brier / wsum
    out[unsafe_offset=2] = correct / wsum
    out[unsafe_offset=3] = zero_one / wsum


@export("al_reg_metrics")
def al_reg_metrics(
    y_addr: Int, yhat_addr: Int, w_addr: Int, n: Int, out_addr: Int
) abi("C"):
    """MAE, MSE, RMSE and R^2 for a regressor, with optional sample weights."""
    var y = fp(y_addr)
    var yhat = fp(yhat_addr)
    var out = fp(out_addr)
    var has_w = w_addr != 0
    var w = fp(w_addr)
    var wsum = Float64(0.0)
    var wy = Float64(0.0)
    var mae = Float64(0.0)
    var mse = Float64(0.0)
    for i in range(n):
        var wi = Float64(1.0)
        if has_w:
            wi = w[unsafe_offset=i]
        wsum += wi
        wy += wi * y[unsafe_offset=i]
        var d = yhat[unsafe_offset=i] - y[unsafe_offset=i]
        var ad = d
        if ad < 0.0:
            ad = -ad
        mae += wi * ad
        mse += wi * d * d
    if wsum == 0.0:
        for k in range(4):
            out[unsafe_offset=k] = 0.0
        return
    var ybar = wy / wsum
    var ss_tot = Float64(0.0)
    for i in range(n):
        var wi = Float64(1.0)
        if has_w:
            wi = w[unsafe_offset=i]
        var d = y[unsafe_offset=i] - ybar
        ss_tot += wi * d * d
    var ss_res = mse / wsum
    out[unsafe_offset=0] = mae / wsum
    out[unsafe_offset=1] = ss_res
    out[unsafe_offset=2] = sqrt(ss_res)
    if ss_tot == 0.0:
        out[unsafe_offset=3] = 0.0
    else:
        out[unsafe_offset=3] = 1.0 - ss_res / (ss_tot / wsum)


@always_inline
def _importance_one(mp: Float64, orig: Float64, ratio: Int, lower: Int) -> Float64:
    if ratio != 0:
        if lower != 0:
            return mp / orig
        return orig / mp
    if lower != 0:
        return mp - orig
    return orig - mp


@export("al_importance")
def al_importance(
    orig: Float64, permuted_addr: Int, n: Int, ratio: Int, lower_is_better: Int,
    out_addr: Int
) abi("C"):
    """Permutation importance samples, following `_compute_importance`.

    Writes the `n` per-repeat values to `out[0..n-1]` and the mean and
    population standard deviation to `out[n]` and `out[n+1]`, matching the
    `np.mean` / `np.std` (ddof=0) aggregation upstream.
    """
    var perm = fp(permuted_addr)
    var out = fp(out_addr)
    var acc = Float64(0.0)
    var acc2 = Float64(0.0)
    for i in range(n):
        var v = _importance_one(perm[unsafe_offset=i], orig, ratio, lower_is_better)
        out[unsafe_offset=i] = v
        acc += v
        acc2 += v * v
    var mean = acc / Float64(n)
    var var2 = acc2 / Float64(n) - mean * mean
    if var2 < 0.0:
        var2 = 0.0
    out[unsafe_offset=n] = mean
    out[unsafe_offset=n + 1] = sqrt(var2)


@export("al_superpose")
def al_superpose(
    alphas_addr: Int, vecs_addr: Int, nb: Int, na: Int, cols: Int, out_addr: Int
) abi("C"):
    """Superposition out[b, :] = sum_a alphas[a] * vecs[b, a, :]."""
    var alphas = fp(alphas_addr)
    var vecs = fp(vecs_addr)
    var out = fp(out_addr)
    for b in range(nb):
        var dst = b * cols
        var src = b * na * cols
        for c in range(cols):
            var acc = Float64(0.0)
            for a in range(na):
                acc = fma(
                    alphas[unsafe_offset=a], vecs[unsafe_offset=src + a * cols + c], acc
                )
            out[unsafe_offset=dst + c] = acc


@export("al_row_l2")
def al_row_l2(a_addr: Int, nb: Int, cols: Int, out_addr: Int) abi("C"):
    """Row-wise L2 norm of an `nb x cols` block."""
    var a = fp(a_addr)
    var out = fp(out_addr)
    for b in range(nb):
        var acc = Float64(0.0)
        for c in range(cols):
            var v = a[unsafe_offset=b * cols + c]
            acc = fma(v, v, acc)
        out[unsafe_offset=b] = sqrt(acc)


@export("al_diff_row_l2")
def al_diff_row_l2(
    a_addr: Int, b_addr: Int, nb: Int, cols: Int, out_addr: Int
) abi("C"):
    """Row-wise L2 norm of `a - b`, the linearity score of one instance."""
    var a = fp(a_addr)
    var b = fp(b_addr)
    var out = fp(out_addr)
    for r in range(nb):
        var acc = Float64(0.0)
        for c in range(cols):
            var d = a[unsafe_offset=r * cols + c] - b[unsafe_offset=r * cols + c]
            acc = fma(d, d, acc)
        out[unsafe_offset=r] = sqrt(acc)


@export("al_conditional_probs")
def al_conditional_probs(
    col_addr: Int, y_addr: Int, n: Int, n_cat: Int, n_y: Int, out_addr: Int
) abi("C"):
    """Conditional label distribution p(cat, y) for the MVDM metric.

    Upstream's `mvdm` fills `p_cond_col[i, i_y]` with the share of class `i_y`
    among the rows whose categorical column equals `i`. The `+ 1e-12` in its
    denominator is upstream's guard against an empty category, so it is kept
    here rather than being treated as a rounding artefact.
    """
    var col = ip(col_addr)
    var y = ip(y_addr)
    var out = fp(out_addr)
    for c in range(n_cat):
        for t in range(n_y):
            out[unsafe_offset=c * n_y + t] = 0.0
    for r in range(n):
        var c = Int(col[unsafe_offset=r])
        var t = Int(y[unsafe_offset=r])
        if c >= 0 and c < n_cat and t >= 0 and t < n_y:
            out[unsafe_offset=c * n_y + t] += 1.0
    for c in range(n_cat):
        var total = Float64(0.0)
        for t in range(n_y):
            total += out[unsafe_offset=c * n_y + t]
        var denom = total + 1e-12
        for t in range(n_y):
            out[unsafe_offset=c * n_y + t] /= denom


@export("al_conditional_probs_eps")
def al_conditional_probs_eps(
    group_addr: Int, col_addr: Int, n: Int, n_cat: Int, n_cat_t: Int,
    eps: Float64, out_addr: Int
) abi("C"):
    """Conditional distribution block p(t, j) for the ABDM metric.

    `group[r]` is the category `j` of the scored column that row `r` belongs
    to, and `col[r]` the value of the *other* categorical column for that row.
    The output is `n_cat_t x n_cat`, entry `[t, j]` holding the share of rows
    in category `j` whose other-column value is `t`, which is exactly the
    block layout upstream's nested loops build. The two category counts differ
    in general, so both are runtime arguments.
    """
    var group = ip(group_addr)
    var col = ip(col_addr)
    var out = fp(out_addr)
    for t in range(n_cat_t):
        for j in range(n_cat):
            out[unsafe_offset=t * n_cat + j] = 0.0
    for r in range(n):
        var j = Int(group[unsafe_offset=r])
        var t = Int(col[unsafe_offset=r])
        if j >= 0 and j < n_cat and t >= 0 and t < n_cat_t:
            out[unsafe_offset=t * n_cat + j] += 1.0
    for j in range(n_cat):
        var cnt = Float64(0.0)
        for r in range(n):
            if Int(group[unsafe_offset=r]) == j:
                cnt += 1.0
        var denom = cnt + eps
        for t in range(n_cat_t):
            out[unsafe_offset=t * n_cat + j] /= denom


@export("al_exact_gather")
def al_exact_gather(
    x_addr: Int, n: Int, f_total: Int, f0: Int, f1: Int, skip: Int, out_addr: Int
) abi("C"):
    """The leave-one-out construction of the exact permutation importance.

    Upstream tiles row `skip` into `N-1` rows and then overwrites the chosen
    feature columns with `np.delete(X[:, features], obj=skip, axis=0)`. The
    output row `k` therefore holds source row `k` for `k < skip`, the tiled
    row `skip` for `k == skip`, and source row `k + 1` for `k > skip`.
    """
    var x = fp(x_addr)
    var out = fp(out_addr)
    var width = f1 - f0
    for k in range(n - 1):
        var src = k
        if k > skip:
            src = k + 1
        elif k == skip:
            src = skip
        var so = src * f_total + f0
        var do = k * width
        for c in range(width):
            out[unsafe_offset=do + c] = x[unsafe_offset=so + c]
