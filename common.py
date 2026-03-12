"""Shared utilities for mmd_variance scripts."""

import yaml
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from qml_benchmarks.data.bernoulli import generate as generate_bernoulli


def ensure_list(val):
    """Wrap a scalar config value in a list; pass through if already a list."""
    return val if isinstance(val, list) else [val]


def load_config(script_dir):
    """Load the YAML config from the given directory."""
    config_path = script_dir / "mmd_variance_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def detect_devices():
    """Detect GPU/CPU JAX devices. Returns (gpu_device_or_None, cpu_device)."""
    try:
        gpu = jax.devices("gpu")[0]
        print(f"GPU detected: {gpu}")
    except RuntimeError:
        gpu = None
        print("No GPU found, falling back to CPU.")
    cpu = jax.devices("cpu")[0]
    return gpu, cpu


# ── Kernel bandwidth ─────────────────────────────────────────────────────────

def sigma_from_mean_weight(mean_weight, n_qubits):
    """Convert a desired mean Pauli weight to the Gaussian kernel bandwidth sigma.

    In mmd_loss_iqp, each bit of each operator is sampled i.i.d. Bernoulli with
    probability p = (1 - exp(-1/(2*sigma^2))) / 2, so the mean weight is n * p.
    Inverting: sigma = 1 / sqrt(-2 * ln(1 - 2*w/n)).
    """
    ratio = 2 * mean_weight / n_qubits
    if ratio >= 1:
        raise ValueError(
            f"mean_op_weight_mmd={mean_weight} is too large for n_qubits={n_qubits} "
            f"(must be < n/2 = {n_qubits / 2})"
        )
    return 1.0 / np.sqrt(-2.0 * np.log(1.0 - ratio))


# ── Parameter initialization ─────────────────────────────────────────────────

def _gate_norm(n_params, mode):
    """Return the normalization factor for parameter initialization."""
    if mode == "unit":
        return 1.0
    elif mode == "sqrt":
        return np.sqrt(n_params)
    elif mode == "linear":
        return float(n_params)
    else:
        raise ValueError(f"Unknown param_normalization: {mode!r}")


def _sample_noise(rng, distribution, spread):
    """Draw a single noise sample from the chosen distribution."""
    if distribution == "uniform":
        return rng.uniform(-spread, spread)
    return rng.normal(0, spread)


def initialize_from_data(gates_list, data, rng, scale=1.0, normalization="sqrt",
                         distribution="normal", use_cov=True, param_noise=0.):
    """Initialize gate parameters from data statistics.

    Weight-1 gates: set from the mean of the corresponding data dimension.
    Weight-2 gates: set from the covariance (use_cov=True) or zero (use_cov=False),
    scaled by a randomly sampled factor.

    When use_cov=False, per-parameter noise is also added to the weight-1 gates
    so they are not fixed across trials.
    """
    n_params = len(gates_list)
    norm = _gate_norm(n_params, normalization)
    spread = scale / norm

    means = np.mean(data, axis=0)
    params = []

    if use_cov:
        sampled_scale = _sample_noise(rng, distribution, spread)
        cov_mat = np.cov((2*data-1).T)
        np.fill_diagonal(cov_mat, 0.)
        max_cov = np.max(np.abs(cov_mat))

    for gate in gates_list:
        if len(gate) == 1:
            gen = gate[0]
            if len(gen) == 1:
                base = np.arcsin(np.sqrt(means[gen[0]]))
                if use_cov:
                    params.append(base)
                else:
                    params.append(base + _sample_noise(rng, distribution, spread) * np.pi)
            elif len(gen) == 2:
                if use_cov:
                    params.append(cov_mat[gen[0], gen[1]] * sampled_scale * np.pi / max_cov)
                else:
                    params.append(_sample_noise(rng, distribution, spread) * np.pi)
            else:
                params.append(rng.normal(0, param_noise))
        else:
            params.append(rng.normal(0, param_noise))
    return jnp.array(params)


def initialize_params(gates, n_qubits, rng, scale=1.0, normalization="sqrt",
                      distribution="normal", center_zero=False):
    """Initialize gate parameters.

    Weight-1 (single-qubit) gates get a pi/4 offset by default, or are
    centered on zero when center_zero=True.  Weight-2 gates are always
    centered on zero.

    distribution: "normal" or "uniform".
    """
    n_params = len(gates)
    spread = 1.0 / _gate_norm(n_params, normalization)
    params = []
    for gate in gates:
        weight = len(gate[0])
        if distribution == "uniform":
            noise = rng.uniform(-spread, spread) * np.pi
        else:
            noise = rng.normal(0, spread) * np.pi
        if weight == 1:
            offset = 0.0 if center_zero else np.pi / 4
            params.append(offset + scale * noise)
        else:
            params.append(scale * noise)
    return jnp.array(params)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(dataset_name, n_qubits, config, rng, script_dir):
    """Load or generate a dataset, returning an array of shape (n_samples, n_qubits)."""
    if dataset_name == "bernoulli":
        cfg = config["bernoulli"]
        data = generate_bernoulli(
            cfg["n_data_samples"], n_bits=n_qubits,
            prob_bitflips=cfg["prob_bitflips"], rng=rng,
        )
    elif dataset_name == "genomic":
        cfg = config["genomic"]
        data_path = script_dir / cfg["path"]
        full_data = np.loadtxt(data_path, delimiter=",")
        if n_qubits > full_data.shape[1]:
            raise ValueError(
                f"qubit_range entry {n_qubits} exceeds genomic feature count "
                f"({full_data.shape[1]})"
            )
        data = full_data[:, :n_qubits]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")
    return jnp.array(data, dtype="float64")
