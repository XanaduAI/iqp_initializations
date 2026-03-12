# IQP Initializations

Studying how initialization strategies affect the trainability of IQP (Instantaneous Quantum Polynomial) circuits, measured via MMD (Maximum Mean Discrepancy) loss variance and training dynamics.

## Structure

| Path | Description |
|------|-------------|
| `mmd_variance_calc.py` | Compute MMD² variance across init strategies, scales, and qubit counts |
| `mmd_variance_render.py` | Plot results from precomputed variance data |
| `mmd_variance_plots/training.py` | Train IQP circuits and record loss curves |
| `correlator_assumption.py` | Verify the correlator assumption on genomic data |
| `datasets/genomic/download_data.py` | Download genomic SNP data |
| `mmd_variance_config.yaml` | Central configuration file |
| `common.py` | Shared utilities |

## Setup

Install dependencies:

```
pip install numpy jax jaxlib pandas matplotlib scipy pyyaml scikit-learn pennylane iqpopt qml_benchmarks
```

## Usage

```bash
# 1. Download genomic data
python datasets/genomic/download_data.py

# 2. Compute MMD² variance
python mmd_variance_calc.py

# 3. (Optional) Run training curves
python mmd_variance_plots/training.py

# 4. Render plots
python mmd_variance_render.py
```

All scripts should be run from the project root.
