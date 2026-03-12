"""
Compute the variance of the MMD loss (over random parameter initializations) as a
function of qubit number for IQP circuits with all weight-1 and weight-2 gates.

For each qubit count n and each of K trials we:
  1. Draw fresh random parameters θ_i.
  2. Run mmd_loss_iqp M times with different keys to get M noisy estimates.
  3. Record the mean (best estimate of the true loss) and the sample variance
     of the M estimates (estimation noise for that trial).

The observed variance of the K means includes both the true parameter variance
and the residual estimation noise:

    Var(L̄_i) = Var_θ[MMD(θ)] + E_θ[σ²_ε(θ)] / M

so we correct:

    Var_θ[MMD(θ)] ≈ Var(L̄_i) − (1/M) mean(s²_i)

Error bars are obtained by propagation of error on both terms.

When ``state_vector_sim: true``, the squared MMD is computed exactly via
PennyLane state-vector simulation (``circuit.probs()``), so each trial yields a
machine-precision value and no noise correction is needed.

Results are accumulated into a pandas DataFrame and saved to disk.  Set
``fresh_start: true`` in the config to discard any previously saved results.
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from pathlib import Path

jax.config.update("jax_enable_x64", True)

from common import (
    ensure_list, load_config, detect_devices,
    sigma_from_mean_weight, initialize_from_data, initialize_params,
    load_dataset,
)

gpu_device, cpu_device = detect_devices()

from iqpopt import IqpSimulator
from iqpopt.utils import local_gates
from iqpopt.gen_qml import mmd_loss_iqp

# ── Load config ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
config = load_config(SCRIPT_DIR)

qubit_range = config["qubit_range"]
n_trials = config["n_trials"]
n_estimates = config["n_estimates_per_trial"]
n_ops = config["n_ops"]
n_samples = config["n_samples"]
mean_op_weight_configs = ensure_list(config["mean_op_weight_mmd"])
random_seed = config["random_seed"]
output_dir = SCRIPT_DIR / config.get("output_dir", "mmd_variance_plots")
datasets = ensure_list(config.get("dataset", "bernoulli"))
init_strategies = ensure_list(config.get("init_strategy", "random"))
_init_range = config.get("init_range", [1.0, 1.0, 1])
init_scales = list(np.logspace(
    np.log10(_init_range[0]), np.log10(_init_range[1]), int(_init_range[2])
))
init_distribution = config.get("init_distribution", "normal")
param_normalizations = ensure_list(config.get("param_normalization", "sqrt"))
results_file = output_dir / config.get("results_file", "mmd_variance_results.pkl")
fresh_start = config.get("fresh_start", False)
state_vector_sim = config.get("state_vector_sim", False)
pl_device_cfg = config.get("pl_device", None)
use_gpu_pl = config.get("use_gpu_pl", False)
grad_order = config.get("grad_order", 0)
use_jit = config.get("use_jit", True)


def corrected_variance_with_error(trial_means, trial_noise_vars, M):
    """Estimate Var_θ[MMD(θ)] corrected for estimation noise, with error bar.

    Args:
        trial_means:      array of shape (K,) — mean MMD estimate per trial
        trial_noise_vars: array of shape (K,) — sample variance of M estimates per trial
        M:                number of estimates per trial

    Returns:
        var_corrected: noise-corrected variance estimate
        var_se:        standard error of the corrected variance
    """
    K = len(trial_means)
    L_bar = trial_means

    # --- Term A: unbiased sample variance of the trial means ---
    A = np.var(L_bar, ddof=1)

    # SE(A) via delta method on A = E[L̄²] - E[L̄]², including covariance term
    mean_Lbar = np.mean(L_bar)
    se_mean_Lbar = np.std(L_bar, ddof=1) / np.sqrt(K)
    se_mean_Lbar2 = np.std(L_bar ** 2, ddof=1) / np.sqrt(K)
    cov_term = np.cov(L_bar ** 2, L_bar, ddof=1)[0, 1] / K
    se_A = np.sqrt(
        se_mean_Lbar2 ** 2
        + (2 * mean_Lbar * se_mean_Lbar) ** 2
        - 4 * mean_Lbar * cov_term
    )

    # --- Term B: average estimation noise contribution = mean(s²_i) / M ---
    B = np.mean(trial_noise_vars) / M
    se_B = np.std(trial_noise_vars, ddof=1) / np.sqrt(K) / M

    # --- Corrected variance and its SE ---
    var_corrected = A - B
    var_se = np.sqrt(se_A ** 2 + se_B ** 2)

    return var_corrected, var_se


def variance_with_error_exact(trial_values):
    """Compute variance and SE from exact (noiseless) trial values.

    Same delta-method SE as the stochastic path but without the noise
    correction term.
    """
    K = len(trial_values)
    var_est = np.var(trial_values, ddof=1)

    mean_x = np.mean(trial_values)
    se_mean_x = np.std(trial_values, ddof=1) / np.sqrt(K)
    se_mean_x2 = np.std(trial_values ** 2, ddof=1) / np.sqrt(K)
    cov_term = np.cov(trial_values ** 2, trial_values, ddof=1)[0, 1] / K
    var_se = np.sqrt(
        se_mean_x2 ** 2
        + (2 * mean_x * se_mean_x) ** 2
        - 4 * mean_x * cov_term
    )

    return var_est, var_se


# ── Exact MMD via state-vector simulation ────────────────────────────────────

def empirical_distribution(data, n_qubits):
    """Convert binary data rows to a probability vector of length 2^n."""
    data_np = np.asarray(data, dtype=np.float64)
    powers = 2 ** np.arange(n_qubits - 1, -1, -1)
    indices = (data_np @ powers).astype(int)
    q = np.bincount(indices, minlength=2**n_qubits).astype(np.float64)
    q /= q.sum()
    return q


def exact_mmd_squared(circuit_probs, data, n_qubits, sigma):
    """Compute exact MMD² between circuit distribution and empirical data distribution.

    Exploits the product structure of the Gaussian kernel on {0,1}^n:

        k(x,y) = γ^{d_H(x,y)},  γ = exp(-1/(2σ²))

    which factorises as K = ⊗_i [[1, γ], [γ, 1]].  This lets us compute
    (p-q)ᵀ K (p-q) in O(n·2ⁿ) time and O(2ⁿ) space.
    """
    gamma = np.exp(-1.0 / (2.0 * sigma**2))

    p = np.asarray(circuit_probs, dtype=np.float64)
    q = empirical_distribution(data, n_qubits)
    d = p - q

    M = np.array([[1.0, gamma], [gamma, 1.0]])
    result = d.reshape([2] * n_qubits).copy()
    for i in range(n_qubits):
        result = np.tensordot(M, result, axes=([1], [i]))
        result = np.moveaxis(result, 0, i)

    return float(np.dot(d, result.ravel()))


def exact_mmd_squared_jax(circuit_probs, q, n_qubits, gamma):
    """JAX-traceable version of exact_mmd_squared for use with vmap.

    Args:
        circuit_probs: probability vector from the circuit, shape (2^n,).
        q:             empirical distribution from data, shape (2^n,).
        n_qubits:      number of qubits.
        gamma:         kernel decay factor exp(-1/(2σ²)).
    """
    d = circuit_probs - q
    M = jnp.array([[1.0, gamma], [gamma, 1.0]])
    result = d.reshape([2] * n_qubits)
    for i in range(n_qubits):
        result = jnp.tensordot(M, result, axes=([1], [i]))
        result = jnp.moveaxis(result, 0, i)
    return jnp.dot(d, result.ravel())


def make_probs_fn(iqp_circuit):
    """Create a JAX-traceable probs function for vmap over parameter batches.

    Only works with default.qubit (pure-Python PennyLane device).
    """
    import pennylane as qml
    dev = qml.device("default.qubit", wires=iqp_circuit.n_qubits)

    @qml.qnode(dev, interface="jax")
    def probs_circuit(params):
        iqp_circuit.iqp_circuit(params)
        return qml.probs(wires=range(iqp_circuit.n_qubits))

    return probs_circuit


# ── DataFrame management ─────────────────────────────────────────────────────

RESULTS_COLUMNS = [
    "dataset", "init_strategy", "init_scale", "n_qubits",
    "mean_op_weight", "state_vector_sim", "grad_order",
    "param_normalization", "init_distribution",
    "value", "error",
]


def load_results(path):
    """Load existing results DataFrame, or return an empty one."""
    if path.exists():
        df = pd.read_pickle(path)
        if "state_vector_sim" not in df.columns:
            df["state_vector_sim"] = False
        if "grad_order" not in df.columns:
            df["grad_order"] = 0
        if "variance" in df.columns and "value" not in df.columns:
            df = df.rename(columns={"variance": "value"})
        if "param_normalization" not in df.columns:
            df["param_normalization"] = "unknown"
        if "init_distribution" not in df.columns:
            df["init_distribution"] = "unknown"
        return df
    return pd.DataFrame(columns=RESULTS_COLUMNS)


def save_results(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def result_exists(df, dataset, strategy, scale, n_qubits, mean_w, sv_sim,
                   g_order, p_norm, i_dist):
    """Check whether a result row already exists."""
    if df.empty:
        return False
    mask = (
        (df["dataset"] == dataset)
        & (df["init_strategy"] == strategy)
        & (df["init_scale"] == scale)
        & (df["n_qubits"] == n_qubits)
        & (df["mean_op_weight"] == mean_w)
        & (df["state_vector_sim"] == sv_sim)
        & (df["grad_order"] == g_order)
        & (df["param_normalization"] == p_norm)
        & (df["init_distribution"] == i_dist)
    )
    return mask.any()


# ── Main computation ─────────────────────────────────────────────────────────

compute_device = gpu_device if gpu_device is not None else cpu_device
output_dir.mkdir(parents=True, exist_ok=True)

effective_grad_order = 0 if state_vector_sim else grad_order

df = load_results(results_file)

if fresh_start is True and results_file.exists():
    results_file.unlink()
    df = pd.DataFrame(columns=RESULTS_COLUMNS)
    print(f"fresh_start=True → deleted all results from {results_file}")
elif not isinstance(fresh_start, bool) and fresh_start in (0, 1):
    n_before = len(df)
    df = df[df["grad_order"] != fresh_start]
    n_removed = n_before - len(df)
    if n_removed > 0:
        save_results(df, results_file)
        print(f"fresh_start={fresh_start} → removed {n_removed} rows with "
              f"grad_order={fresh_start} from {results_file}")
    else:
        print(f"fresh_start={fresh_start} → no rows with grad_order={fresh_start} to remove")
sim_label = "state-vector" if state_vector_sim else "stochastic"

if effective_grad_order == 1:
    def _mmd_for_grad(params, circuit, data, *, sigma, n_ops, n_samples, key):
        return mmd_loss_iqp(params, circuit, data, sigma=sigma,
                            n_ops=n_ops, n_samples=n_samples, key=key,
                            jit=use_jit)
    grad_fn = jax.grad(_mmd_for_grad)
else:
    grad_fn = None

for ds_name in datasets:
    for scale in init_scales:
        for weight_cfg in mean_op_weight_configs:
            if weight_cfg == "constant_bandwidth":
                ref_weight = 2.0
                n_min = min(qubit_range)
                fixed_sigma = sigma_from_mean_weight(ref_weight, n_min)
                weight_label = "constant_bandwidth"
            else:
                fixed_sigma = None
                weight_label = float(weight_cfg)

            print(f"\n{'='*60}")
            print(f"Dataset: {ds_name}  |  init_scale: {scale}  |  sim: {sim_label}"
                  f"  |  grad_order: {effective_grad_order}"
                  f"  |  mean_op_weight: {weight_label}")
            print(f"{'='*60}")

            for param_normalization in param_normalizations:
                for strategy in init_strategies:
                    rng = np.random.default_rng(random_seed)

                    for n in qubit_range:
                        if result_exists(df, ds_name, strategy, scale, n,
                                         weight_label, state_vector_sim,
                                         effective_grad_order,
                                         param_normalization, init_distribution):
                            print(f"  [{strategy}|{param_normalization}] n_qubits = {n}"
                                  f"  — already computed, skipping")
                            continue

                        if fixed_sigma is not None:
                            sigma = fixed_sigma
                        else:
                            sigma = sigma_from_mean_weight(weight_cfg, n)
                        print(f"  [{strategy}|{param_normalization}] n_qubits = {n}"
                              f"  (sigma = {sigma:.4f})")

                        gates = local_gates(n, max_weight=2)
                        if pl_device_cfg:
                            pl_device = pl_device_cfg
                        elif state_vector_sim and use_gpu_pl:
                            pl_device = "lightning.gpu"
                        else:
                            pl_device = "lightning.qubit"
                        circuit = IqpSimulator(n, gates, device=pl_device)

                        ground_truth = load_dataset(ds_name, n, config, rng, SCRIPT_DIR)

                        if state_vector_sim:
                            all_params = []
                            for t in range(n_trials):
                                if strategy in ("from_data", "from_data_nocov"):
                                    params = initialize_from_data(
                                        gates, ground_truth, rng, scale=scale,
                                        normalization=param_normalization,
                                        distribution=init_distribution,
                                        use_cov=(strategy == "from_data"),
                                    )
                                else:
                                    params = initialize_params(
                                        gates, n, rng, scale=scale,
                                        normalization=param_normalization,
                                        distribution=init_distribution,
                                        center_zero=(strategy == "random_zero"),
                                    )
                                all_params.append(params)

                            if pl_device == "default.qubit":
                                params_batch = jnp.stack(all_params)
                                probs_fn = make_probs_fn(circuit)
                                all_probs = jax.jit(jax.vmap(probs_fn))(params_batch)

                                q = jnp.array(empirical_distribution(ground_truth, n))
                                gamma = jnp.exp(-1.0 / (2.0 * sigma**2))
                                trial_values = np.array(jax.vmap(
                                    lambda p: exact_mmd_squared_jax(p, q, n, gamma)
                                )(all_probs))
                            else:
                                trial_values = []
                                for params in all_params:
                                    probs = circuit.probs(params)
                                    mmd_val = exact_mmd_squared(
                                        probs, ground_truth, n, sigma)
                                    trial_values.append(mmd_val)
                                trial_values = np.array(trial_values)

                            result_val, result_err = variance_with_error_exact(trial_values)

                        elif effective_grad_order == 0:
                            ground_truth_dev = jax.device_put(ground_truth, compute_device)
                            master_key = jax.random.PRNGKey(rng.integers(0, 2**31))
                            trial_means = []
                            trial_noise_vars = []

                            for t in range(n_trials):
                                if strategy in ("from_data", "from_data_nocov"):
                                    params = initialize_from_data(
                                        gates, ground_truth, rng, scale=scale,
                                        normalization=param_normalization,
                                        distribution=init_distribution,
                                        use_cov=(strategy == "from_data"),
                                    )
                                else:
                                    params = initialize_params(
                                        gates, n, rng, scale=scale,
                                        normalization=param_normalization,
                                        distribution=init_distribution,
                                        center_zero=(strategy == "random_zero"),
                                    )

                                params_dev = jax.device_put(params, compute_device)

                                estimates = []
                                for _ in range(n_estimates):
                                    master_key, subkey = jax.random.split(master_key)
                                    subkey_dev = jax.device_put(subkey, compute_device)
                                    loss_val = mmd_loss_iqp(
                                        params_dev,
                                        circuit,
                                        ground_truth_dev,
                                        sigma=sigma,
                                        n_ops=n_ops,
                                        n_samples=n_samples,
                                        key=subkey_dev,
                                        jit=use_jit,
                                    )
                                    estimates.append(float(jax.device_get(loss_val)))

                                estimates = np.array(estimates)
                                trial_means.append(np.mean(estimates))
                                trial_noise_vars.append(np.var(estimates, ddof=1))

                            trial_means = np.array(trial_means)
                            trial_noise_vars = np.array(trial_noise_vars)

                            result_val, result_err = corrected_variance_with_error(
                                trial_means, trial_noise_vars, n_estimates
                            )

                        else:
                            ground_truth_dev = jax.device_put(ground_truth, compute_device)
                            master_key = jax.random.PRNGKey(rng.integers(0, 2**31))
                            trial_means = []

                            for t in range(n_trials):
                                if strategy in ("from_data", "from_data_nocov"):
                                    params = initialize_from_data(
                                        gates, ground_truth, rng, scale=scale,
                                        normalization=param_normalization,
                                        distribution=init_distribution,
                                        use_cov=(strategy == "from_data"),
                                    )
                                else:
                                    params = initialize_params(
                                        gates, n, rng, scale=scale,
                                        normalization=param_normalization,
                                        distribution=init_distribution,
                                        center_zero=(strategy == "random_zero"),
                                    )

                                params_dev = jax.device_put(params, compute_device)

                                estimates = []
                                for _ in range(n_estimates):
                                    master_key, subkey = jax.random.split(master_key)
                                    subkey_dev = jax.device_put(subkey, compute_device)
                                    g = grad_fn(
                                        params_dev,
                                        circuit,
                                        ground_truth_dev,
                                        sigma=sigma,
                                        n_ops=n_ops,
                                        n_samples=n_samples,
                                        key=subkey_dev,
                                    )
                                    max_abs_grad = float(jnp.max(jnp.abs(g)))
                                    estimates.append(max_abs_grad)

                                trial_means.append(np.mean(estimates))

                            trial_means = np.array(trial_means)
                            result_val = np.mean(trial_means)
                            result_err = np.std(trial_means, ddof=1) / np.sqrt(n_trials)

                        print(f"    value = {result_val:.6e} +/- {result_err:.6e}")

                        new_row = pd.DataFrame([{
                            "dataset": ds_name,
                            "init_strategy": strategy,
                            "init_scale": scale,
                            "n_qubits": n,
                            "mean_op_weight": weight_label,
                            "state_vector_sim": state_vector_sim,
                            "grad_order": effective_grad_order,
                            "param_normalization": param_normalization,
                            "init_distribution": init_distribution,
                            "value": result_val,
                            "error": result_err,
                        }])
                        df = pd.concat([df, new_row], ignore_index=True)
                        save_results(df, results_file)

print(f"\nResults saved to {results_file}")
print(df.to_string(index=False))
