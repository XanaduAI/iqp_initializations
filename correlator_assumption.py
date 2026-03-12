"""
Verify the low-body correlated target assumption for genomic data.

Computes and plots the mean and maximum of |t_A - prod_{j in A} t_j|
as a function of subset size |A|, where t_A = E[(-1)^{sum_{j in A} x_j}].

Uses JAX with jit and lax.scan for vectorised computation over *all* subsets.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from itertools import combinations
from math import comb
from functools import partial
from scipy.optimize import curve_fit

# ── Configuration ────────────────────────────────────────────────────────────

N_QUBITS = 20
MAX_SUBSET_SIZE = 20
BATCH_SIZE = 10_000
FIT_MAX_SIZE = 10       # fit (C/n)^{k/2} using subset sizes 2..FIT_MAX_SIZE
SHOW_FIT = False         # if False, omit the fitted (C/n)^{|A|/2} curve
DPI = 300

# ── Plot style ───────────────────────────────────────────────────────────────

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.facecolor": "#F5F5F5",
    "axes.prop_cycle": mpl.cycler(color=[
        "#4477AA", "#EE6677", "#228833", "#CCBB44",
        "#66CCEE", "#AA3377", "#BBBBBB",
    ]),
})

# ── Load data ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "datasets/genomic/805_SNP_1000G_real_train.csv"
OUTPUT_DIR = SCRIPT_DIR / "mmd_variance_plots"

data = np.loadtxt(DATA_PATH, delimiter=",")[:, :N_QUBITS].astype(np.float32)
n_samples, n = data.shape
print(f"Loaded {n_samples} samples, using {n} features")

data_jnp = jnp.array(data)

# ── Single-qubit correlators ────────────────────────────────────────────────

z = 1 - 2 * data_jnp  # {0,1} -> {+1,-1}
t_single = z.mean(axis=0)

# ── JIT-compiled batch computation ──────────────────────────────────────────
#
# Instead of gathering z[:, A] and taking products (O(samples * subsets * k)
# memory), exploit the identity
#     prod_{j in A} (-1)^{x_j}  =  (-1)^{sum_{j in A} x_j}
# and compute the sum via a matmul with a binary indicator matrix.

@partial(jax.jit, static_argnames=("n_batches",))
def compute_t_A(data, indicator_padded, n_batches):
    """Return t_A for every subset encoded in *indicator_padded*.

    Parameters
    ----------
    data : (n_samples, n) float32, values in {0, 1}
    indicator_padded : (n_batches * BATCH_SIZE, n) float32, binary rows
    n_batches : static int
    """
    batched = indicator_padded.reshape(n_batches, BATCH_SIZE, n)

    def body(_, indicator_batch):
        counts = data @ indicator_batch.T           # (n_samples, BATCH_SIZE)
        signs = 1.0 - 2.0 * jnp.mod(counts, 2.0)   # (-1)^count
        return _, signs.mean(axis=0)                 # (BATCH_SIZE,)

    _, t_A = lax.scan(body, None, batched)
    return t_A.ravel()

# ── Main loop over subset sizes ─────────────────────────────────────────────

subset_sizes = np.arange(2, MAX_SUBSET_SIZE + 1)
mean_diffs = np.empty(len(subset_sizes))
max_diffs = np.empty(len(subset_sizes))

for idx, k in enumerate(subset_sizes):
    t0 = time.perf_counter()
    n_subsets = comb(n, k)

    subsets_idx = np.array(list(combinations(range(n), k)), dtype=np.int32)

    # Build binary indicator matrix via advanced indexing
    indicator = np.zeros((n_subsets, n), dtype=np.float32)
    rows = np.repeat(np.arange(n_subsets), k)
    indicator[rows, subsets_idx.ravel()] = 1.0

    # Pad to a multiple of BATCH_SIZE so every scan iteration has the same shape
    n_pad = (-n_subsets) % BATCH_SIZE
    if n_pad:
        indicator = np.concatenate(
            [indicator, np.zeros((n_pad, n), dtype=np.float32)]
        )
    n_batches = len(indicator) // BATCH_SIZE

    t_A = compute_t_A(data_jnp, jnp.array(indicator), n_batches)
    t_A = t_A[:n_subsets]

    prod_t = jnp.prod(t_single[jnp.array(subsets_idx)], axis=1)
    diffs = jnp.abs(t_A - prod_t)

    mean_diffs[idx] = float(diffs.mean())
    max_diffs[idx] = float(diffs.max())
    elapsed = time.perf_counter() - t0
    print(f"|A|={k:2d}: {n_subsets:>8d} subsets, "
          f"mean={mean_diffs[idx]:.2e}, max={max_diffs[idx]:.2e}  "
          f"({elapsed:.1f}s)")

# ── Fit (C/n)^{k/2} to the max curve ────────────────────────────────────────

if SHOW_FIT:
    def bound_model(k, C):
        return (C / n) ** (k / 2)

    fit_mask = subset_sizes <= FIT_MAX_SIZE
    popt, _ = curve_fit(bound_model, subset_sizes[fit_mask], max_diffs[fit_mask],
                        p0=[1.0], bounds=(0, np.inf))
    C_fit = popt[0]
    print(f"\nFitted C = {C_fit:.4f}  (bound: (C/n)^(|A|/2))")

    k_fine = np.linspace(subset_sizes[0], subset_sizes[-1], 200)
    bound_curve = bound_model(k_fine, C_fit)

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))

ax.semilogy(subset_sizes, mean_diffs, "o-", markersize=5,
            label=r"Mean $\left|t_A - \prod_{j\in A} t_j\right|$")
ax.semilogy(subset_sizes, max_diffs, "s-", markersize=5,
            label=r"Max $\left|t_A - \prod_{j\in A} t_j\right|$")
if SHOW_FIT:
    ax.semilogy(k_fine, bound_curve, "--", color="gray", alpha=0.7,
                label=rf"$(C/n)^{{|A|/2}}$, $C={C_fit:.1f}$")

ax.set_xlabel(r"Subset size $|A|$")
ax.set_ylabel(r"$\left|t_A - \prod_{j\in A} t_j\right|$")
ax.set_title(rf"Low-body correlation assumption (genomic, $n={n}$)")
ax.legend()
ax.set_xticks(subset_sizes)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = OUTPUT_DIR / f"correlator_assumption_genomic_n{n}.png"
fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.close(fig)
