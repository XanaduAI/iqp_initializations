"""
Train IQP circuits using the MMD² loss and save training loss curves.

For each combination of (dataset, init_scale, mean_op_weight, init_strategy,
n_qubits, param_normalization) and for n_trials independent random
initialisations, trains an IQP circuit using the iqpopt Trainer and records
the per-step loss values.

Results are saved to a pickle file as a list of dicts, each containing
the run metadata and a numpy array of per-step losses.
"""

import numpy as np
import pickle
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

from iqpopt import IqpSimulator, Trainer
from iqpopt.utils import local_gates
from iqpopt.gen_qml import mmd_loss_iqp

# ── Load config ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
config = load_config(SCRIPT_DIR)

qubit_range = config["qubit_range"]
n_trials = config["n_trials"]
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
pl_device_cfg = config.get("pl_device", None)
use_jit = config.get("use_jit", True)

n_steps = config.get("n_steps", 500)
learning_rate = config.get("learning_rate", 0.01)

results_path = output_dir / config.get("training_results_file", "training_results.pkl")
fresh_start = config.get("fresh_start", False)

# ── Results management ───────────────────────────────────────────────────────

def load_results(path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)


def result_exists(results, dataset, strategy, scale, n_qubits, mean_w,
                  p_norm, i_dist):
    for r in results:
        if (r["dataset"] == dataset
                and r["init_strategy"] == strategy
                and r["init_scale"] == scale
                and r["n_qubits"] == n_qubits
                and r["mean_op_weight"] == mean_w
                and r["param_normalization"] == p_norm
                and r["init_distribution"] == i_dist):
            return True
    return False


# ── Main computation ─────────────────────────────────────────────────────────

compute_device = gpu_device if gpu_device is not None else cpu_device
output_dir.mkdir(parents=True, exist_ok=True)

if fresh_start is True and results_path.exists():
    results_path.unlink()
    print(f"fresh_start=True → deleted {results_path}")

results = load_results(results_path)

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
            print(f"Dataset: {ds_name}  |  init_scale: {scale}"
                  f"  |  mean_op_weight: {weight_label}"
                  f"  |  lr: {learning_rate}  |  steps: {n_steps}")
            print(f"{'='*60}")

            for param_normalization in param_normalizations:
                for strategy in init_strategies:
                    for n in qubit_range:
                        if result_exists(results, ds_name, strategy, scale, n,
                                         weight_label, param_normalization,
                                         init_distribution):
                            print(f"  [{strategy}|{param_normalization}] n={n}"
                                  f" — already computed, skipping")
                            continue

                        if fixed_sigma is not None:
                            sigma = fixed_sigma
                        else:
                            sigma = sigma_from_mean_weight(weight_cfg, n)
                        print(f"  [{strategy}|{param_normalization}] n={n}"
                              f" (sigma={sigma:.4f})")

                        gates = local_gates(n, max_weight=2)
                        pl_device = pl_device_cfg or "lightning.qubit"
                        circuit = IqpSimulator(n, gates, device=pl_device)

                        strat_seed = random_seed + hash(strategy) % (2**31)
                        rng = np.random.default_rng(strat_seed)
                        ground_truth = load_dataset(ds_name, n, config, rng, SCRIPT_DIR)
                        ground_truth_dev = jax.device_put(ground_truth, compute_device)

                        all_losses = []
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

                            trainer = Trainer("Adam", mmd_loss_iqp, learning_rate)
                            loss_kwargs = {
                                "params": params_dev,
                                "iqp_circuit": circuit,
                                "ground_truth": ground_truth_dev,
                                "sigma": sigma,
                                "n_ops": n_ops,
                                "n_samples": n_samples,
                                "jit": use_jit,
                            }
                            trainer.train(
                                n_steps, loss_kwargs,
                                random_state=rng.integers(0, 2**31),
                            )

                            losses = np.array(trainer.losses)
                            all_losses.append(losses)
                            print(f"    trial {t}: final_loss={losses[-1]:.6e} "
                                  f"(time={trainer.run_time:.1f}s)")

                        entry = {
                            "dataset": ds_name,
                            "init_strategy": strategy,
                            "init_scale": scale,
                            "n_qubits": n,
                            "mean_op_weight": weight_label,
                            "param_normalization": param_normalization,
                            "init_distribution": init_distribution,
                            "losses": np.stack(all_losses),
                        }
                        results.append(entry)
                        save_results(results, results_path)

print(f"\nTraining results saved to {results_path}")
print(f"Total entries: {len(results)}")
