# mojo-alibi

`mojo-alibi` is the compute-oriented subset of
[alibi](https://github.com/SeldonIO/alibi) with the numeric inner loops
implemented in Mojo and callable from Python. It keeps `alibi`'s function
shapes for the covered operations and moves the array work into one compiled
shared library.

The Python package is named `mojo_alibi`, so it installs alongside the real
`alibi` and the tests can be compared against it directly when it is
installed.

```python
import numpy as np
import mojo_alibi as ma

x = np.random.default_rng(0).standard_normal((500, 16))
d = ma.squared_pairwise_distance(x, x)      # clipped N x N, chunked + threaded
k = ma.gaussian_rbf(x, x)                    # mean-over-sigmas RBF kernel matrix
ma.pd_variance(np.random.default_rng(1).standard_normal((3, 100)))
ma.class_metrics(y, proba)                   # log loss / Brier / accuracy / 0-1
```

## What alibi actually is, and what that means here

`alibi` is a model-interpretability library. Most of it is Python control flow:
constructing an `Explanation`, rendering a matplotlib figure, cloning a model,
picking a metric name from a string. None of that is a compute kernel and none
of it is ported.

What `alibi` does have is a set of genuinely numeric inner routines that sit
underneath its explainers, and those are what this port covers. Each one is a
loop over an array or a reduction over instances, and each maps onto an
exported symbol in `src/kernels.mojo`.

## Covered subset

| upstream surface | what is computed in Mojo | ported API |
| --- | --- | --- |
| `alibi.utils.distributions.kl_bernoulli` | elementwise Bernoulli KL with upstream's `1e-7` / `1-1e-16` clip | `kl_bernoulli` |
| `alibi.utils.distance.cityblock_batch` | batch L1 reduction | `cityblock_batch` |
| `alibi.utils.distance.squared_pairwise_distance` | `Nx x Ny` clipped squared distance, direct differences, row-chunked and threaded | `squared_pairwise_distance` |
| `alibi.utils.distance.mvdm` | conditional label distributions, then the `n_cat x n_cat` value-difference matrix | `mvdm` |
| `alibi.utils.distance.abdm` | conditional distributions per category pair, then the symmetric-KL matrix | `abdm` |
| `alibi.utils.kernel.GaussianRBF` | median-distance sigma inference (LSD radix sort) and the mean-over-sigmas RBF kernel | `infer_sigma`, `rbf_kernel`, `gaussian_rbf`, `sort_f64` |
| `alibi.explainers.pd_variance` | `std(ddof=1)` and `(max - min) / 4` along the grid axis | `pd_variance` |
| `alibi.prototypes.protoselect.ProtoSelect.summarise` | the page-8 coverage masks, the score matrix, the argmax and the incremental score update | `protoselect_summarise` |
| `alibi.explainers.permutation_importance` | the permuted gather, the half swap, the leave-one-out construction, the sklearn reductions and the importance aggregation | `perm_gather`, `feature_swap`, `exact_gather`, `class_metrics`, `reg_metrics`, `permutation_importance_samples` |
| `alibi.confidence.model_linearity` | the linear superposition and the row-wise L2 residual | `superposition`, `row_l2`, `linearity_score` |

### Not implemented

- The explainer classes themselves (`PartialDependence`, `Counterfactual`,
  `Anchor*`, `CEM`, `CFProto`, `IntegratedGradients`, `ALE`,
  `PermutationImportance`, `ProtoSelect`, `LinearityMeasure`): object
  lifecycle, metadata, plotting and the model callables.
- `ProtoSelect`'s k-nearest-neighbour sampling and `LinearityMeasure`'s
  `knn` / `grid` samplers, which call into scikit-learn and `np.random`.
- `alibi.utils.distance.multidim_scaling`, which is a thin wrapper over
  `sklearn.manifold.MDS`.
- `alibi.explainers.permutation_importance`'s `precision`, `recall`, `f1` and
  `roc_auc` scores, which are sklearn functions with no array work of their own.
- `alibi.utils.discretizer` and `alibi.utils.gradients`, which are data
  manipulation and model plumbing.

Use the real `alibi` for all of that; `mojo_alibi` is the numeric layer under
it, not a replacement.

### Numerical contract

C-contiguous `float64` (and `int32` for the categorical masks) throughout.
Non-contiguous and non-`float64` inputs are copied by the shim rather than
narrowed. `squared_pairwise_distance` exposes only upstream's default clip
bounds `(1e-7, 1e30)`; anything else raises rather than silently ignoring them.

Mojo emits FMA, so results match NumPy to a tolerance, never bit for bit, and
the parity tests assert accordingly. The kernel routes `log` to libm through
`external_call` because `std.math.log` on `float64` is only good to 3.4e-9
relative, which is visible directly in the KL and ABDM results; `exp` and
`sqrt` from `std.math` are used as is.

Two behaviours are worth stating because they are deliberate and a caller
could trip over them:

- `squared_pairwise_distance` computes `sum (x - y)^2` from direct differences
  rather than upstream's `|x|^2 + |y|^2 - 2 x.y` expansion. Same result, better
  conditioned; the difference from the expansion is at the 1e-9 relative level
  on well-separated data and the tests assert exactly that.
- `protoselect_summarise` walks the available prototypes in ascending index
  order, where upstream walks a `set`. The argmax is unaffected except under an
  exact tie, which this deterministic order resolves.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-alibi.so`. Set `PYTHONPATH=python` when
using the package outside a Pixi task:

```bash
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

`MOJO_ALIBI_WORKERS` sets the thread-pool width used by the chunked pairwise
kernel (default 8).

## Performance

Best-of-three wall clock against the fastest reasonable NumPy formulation of
the same quantity, in one process. Every case verifies numerical agreement
before it times anything, so a kernel regression shows up as a correctness
failure rather than a good number. Measured on a shared machine, so the
absolute numbers move; the ratios are the stable part.

| case | reference | mojo-alibi | result |
| --- | ---: | ---: | ---: |
| `kl_bernoulli` n=4194304 | 489.74 ms | 190.49 ms | 2.57x faster |
| `cityblock_batch` n=1048576 f=16 | 975.60 ms | 84.21 ms | 11.59x faster |
| `squared_pairwise_distance` 2048x2048x32 | 218.81 ms | 111.91 ms | 1.96x faster |
| `rbf_kernel` 1024x1024 s=8 | 371.27 ms | 138.82 ms | 2.67x faster |
| `rbf` + inferred sigma n=768 | 169.26 ms | 275.89 ms | 0.61x, slower |
| `sort_f64` n=2097152 | 60.09 ms | 622.75 ms | 0.10x, slower |
| `class_metrics` n=1048576 c=5 | 255.55 ms | 69.70 ms | 3.67x faster |
| `mvdm` n=200000 cat=12 | 56.43 ms | 24.02 ms | 2.35x faster |
| `protoselect` mask 400x2000x4 | 66.23 ms | 27.69 ms | 2.39x faster |
| `linearity_score` 4096x16 | 0.32 ms | 0.16 ms | 1.96x faster |

Two results are losses and they are both the sort.

`sort_f64` is a four-stage LSD radix sort: eight stable counting passes over the
order-preserving 64-bit key, with a 1 KiB histogram and two scratch buffers the
shim owns. It is linear in `n` and exact, and it is the right shape for what
`GaussianRBF` actually does with a sorted distance matrix, which is read one
element at a time for the median. It is still ten times slower than
`np.sort`, which is a hand-vectorised AVX-512 quicksort and simply wins a pure
sorting contest; a sixteen-bit-per-pass variant was also tried and is 1.6x
slower still, because a 256 KiB histogram turns every scatter into an L2 miss.
The honest conclusion is that a linear-time sort is not a speed win here, and
the end-to-end row above shows what that costs: `GaussianRBF(sigma=None)` runs
at 0.61x of the NumPy pipeline.

`protoselect`'s mask build and `cityblock_batch` both beat NumPy comfortably;
the pairwise distance kernel threads over row blocks, which is why it is the
second-best case rather than the best.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-alibi.so`.

The `python/mojo_alibi` layer owns every array and the upstream control flow.
It normalises inputs to contiguous `float64`, builds the small index and mask
arrays that upstream's Python builds, and then makes one call into the kernel
per reduction. Buffers cross the C ABI as 64-bit addresses and are reconstructed
in Mojo as `Pointer[Float64, AnyOrigin[mut=True]]`, which is what keeps the
exported symbols non-parametric.

Parallelism is deliberately narrow. 1.2.0 cannot carry a pointer into a
`parallelize` body, so `squared_pairwise_distance` exports a row-range entry
point and the shim fans out over a `ThreadPoolExecutor`; ctypes releases the GIL
for the foreign call, so the fan-out is real. Every other kernel is memory bound
or small enough that threading would only add call overhead, so they run
serially. `squared_pairwise_distance` also falls back to a serial loop below
`THREAD_THRESHOLD` fused multiply-adds for the same reason.

## Tests

`tests/` is a parity suite of 124 tests, all against something other than the
shim itself:

- `test_distributions.py` -- KL against a transcription of the upstream
  expression, plus the clip boundaries, the asymmetry and exact zero on
  identical inputs.
- `test_distance.py` -- `cityblock_batch` against `scipy.spatial.distance`
  (which is what upstream's own test uses), the pairwise distance against the
  expansion form, and MVDM / ABDM against transcriptions of the upstream loop
  nests, including the exact symmetry the lower-triangle-then-transpose
  construction produces.
- `test_kernel.py` -- the sort against `np.sort` bit for bit, the sigma
  inference against upstream's index arithmetic, and the RBF matrix against
  upstream's `exp(-concat(...))` expression.
- `test_pdvariance.py` -- the two variance reductions against numpy.
- `test_protoselect.py` -- the score matrix against a transcription of the
  page-8 vectorisation, the greedy pick list against a replay of upstream's
  loop, and the invariants that define it (a pick only happens on a
  non-negative score, every pick adds new coverage, selection stops when
  nothing scores positive).
- `test_permutation.py` -- the metric reductions against the very sklearn
  functions `alibi` calls, and the data movement against the numpy expressions
  upstream writes.
- `test_linearity.py` -- the superposition against the upstream einsum, and the
  defining property that a linear model scores exactly zero while a quadratic
  one does not.

`alibi` is not installed in the parity test venv, so where a reference exists in
a third party that `alibi` itself delegates to (scikit-learn, scipy) the test
uses that; otherwise the reference is a direct transcription of the upstream
expression, and the docstring of each test file says so.

## License

MIT
