"""
Plot the variance of the MMD loss from pre-computed results.

Reads the pandas DataFrame produced by mmd_variance_calc.py and generates:

1. Variance vs init_scale (exact simulations only) — one plot per
   (dataset, init_strategy, mean_op_weight) group, with different qubit counts
   as separate series.
2. Mean max|grad| vs qubit number — one plot per (dataset, init_scale,
   mean_op_weight) group, with different init_strategies as separate series.
3. Training loss curves — one panel plot per (dataset, mean_op_weight, n_qubits)
   group, with panels for each param_normalization value.
"""

import yaml
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

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

# ── Load config ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent

config_path = SCRIPT_DIR / "mmd_variance_config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

output_dir = SCRIPT_DIR / config.get("output_dir", "mmd_variance_plots")
results_file = output_dir / config.get("results_file", "mmd_variance_results.pkl")
dpi = config["dpi"]
log_scale_x = config.get("log_scale_x", False)
log_scale_y = config.get("log_scale_y", False)
init_scale_xlim = config.get("init_scale_xlim", [0, 1])
show_error_bars = config.get("show_error_bars", True)
train_log_y = config.get("train_log_y", True)
train_show_mean = config.get("train_show_mean", True)

# ── Load results ─────────────────────────────────────────────────────────────

if not results_file.exists():
    raise FileNotFoundError(
        f"Results file not found: {results_file}\n"
        "Run mmd_variance_calc.py first to compute the data."
    )

df = pd.read_pickle(results_file)
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
print(f"Loaded {len(df)} rows from {results_file}")

# ── Strategy display names ────────────────────────────────────────────────────

strategy_labels = {
    "random": "unbiased",
    "random_zero": "identity",
    "from_data_nocov": "data dependent",
    "from_data": "covariance",
}

# ── Generate plots ───────────────────────────────────────────────────────────

output_dir.mkdir(parents=True, exist_ok=True)

# ── 1. Variance vs init_scale (exact simulations only, grad_order=0) ────────

df_var = df[df["grad_order"] == 0]
df_exact = df_var[df_var["state_vector_sim"] == True]

if not df_exact.empty:
    cmap = plt.get_cmap("tab20b")
    strategy_order = ["random", "random_zero", "from_data_nocov", "from_data"]
    strategy_labels = {
        "random": "unbiased",
        "random_zero": "identity",
        "from_data_nocov": "data dependent",
        "from_data": "covariance",
    }
    norm_linestyles = ["-", "--", ":", "-."]

    for (ds_name, mow), mow_group in df_exact.groupby(["dataset", "mean_op_weight"]):
        strategies_present = [s for s in strategy_order
                              if s in mow_group["init_strategy"].unique()]
        n_cols = len(strategies_present)
        if n_cols == 0:
            continue

        qubit_vals = sorted(mow_group["n_qubits"].unique())
        n_series = len(qubit_vals)
        if n_series == 1:
            color_map = {qubit_vals[0]: cmap(0.5)}
        else:
            color_map = {n: cmap(i / (n_series - 1))
                         for i, n in enumerate(qubit_vals)}

        norm_vals = sorted(mow_group["param_normalization"].unique())
        multi_norm = len(norm_vals) > 1
        norm_ls_map = {nv: norm_linestyles[i % len(norm_linestyles)]
                       for i, nv in enumerate(norm_vals)}

        fig, axes = plt.subplots(
            2, n_cols, figsize=(5.5 * n_cols, 10),
            squeeze=False,
        )

        for col, strategy in enumerate(strategies_present):
            label = strategy_labels.get(strategy, strategy)
            strat_group = mow_group[mow_group["init_strategy"] == strategy]
            ax1 = axes[0, col]
            ax2 = axes[1, col]

            for pnorm in norm_vals:
                group = strat_group[strat_group["param_normalization"] == pnorm]
                ls = norm_ls_map[pnorm]

                if strategy != "from_data":
                    plot_group = group[
                        group["init_scale"].between(init_scale_xlim[0], init_scale_xlim[1])
                    ]
                else:
                    plot_group = group

                for n_qubits in qubit_vals:
                    q_group = plot_group[plot_group["n_qubits"] == n_qubits].sort_values("init_scale")
                    ax1.errorbar(
                        q_group["init_scale"],
                        q_group["value"],
                        yerr=q_group["error"] if show_error_bars else None,
                        fmt="o", capsize=4, linewidth=1.5,
                        linestyle=ls,
                        color=color_map[n_qubits],
                    )

                idx_max = group.groupby("n_qubits")["value"].idxmax()
                max_var_per_n = group.loc[idx_max].set_index("n_qubits").reindex(qubit_vals)

                ax2.plot(
                    qubit_vals, max_var_per_n["value"].values,
                    marker="o", linestyle=ls, linewidth=1.5, color="#4477AA",
                )
                for n_qubits in qubit_vals:
                    ax2.plot(
                        n_qubits, max_var_per_n.loc[n_qubits, "value"],
                        "o", markersize=6, color=color_map[n_qubits],
                    )

                ax2r = ax2.twinx()
                ax2r.plot(
                    qubit_vals, max_var_per_n["init_scale"].values,
                    marker="s", linestyle=ls, linewidth=1.2,
                    markersize=5, color="#EE6677",
                )
                ax2r.tick_params(axis="y", labelcolor="black")
                if log_scale_y:
                    ax2r.set_yscale("log")
                if col == n_cols - 1:
                    ax2r.set_ylabel("Best initialization scale", fontsize=12, color="#EE6677")

            ax1.set_title(label, fontsize=13, fontstyle="italic")
            if log_scale_x:
                ax1.set_xscale("log")
            if strategy != "from_data":
                ax1.set_xlim(init_scale_xlim)
            if log_scale_y:
                ax1.set_yscale("log")
            else:
                ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax1.grid(True, alpha=0.3)
            if col == 0:
                ax1.set_ylabel("Var$_{\\theta}$[MMD$^2$]", fontsize=12)
                ax1.set_xlabel("Initialization scale", fontsize=12)

            if log_scale_x:
                ax2.set_xscale("log")
            ax2.set_xticks(qubit_vals)
            ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            if log_scale_y:
                ax2.set_yscale("log")
            else:
                ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax2.grid(True, alpha=0.3)
            if col == 0:
                ax2.set_ylabel("Best variance", fontsize=12)
                ax2.set_xlabel("Number of qubits ($n$)", fontsize=12)

        mow_label = f"$w={mow}$" if isinstance(mow, (int, float)) else str(mow)
        fig.suptitle(
            f"MMD$^2$ Variance — {ds_name} ({mow_label})",
            fontsize=15, y=1.02,
        )

        legend_y = 1.0 - 0.15 / fig.get_size_inches()[1]
        if multi_norm:
            norm_legend_y = legend_y
            legend_y -= 0.03

        pad = 0.02
        box_size = 0.016
        total_width = sum(
            box_size + 0.008 + len(f"n={n}") * 0.011 + pad
            for n in qubit_vals
        ) - pad
        x = 0.5 - total_width / 2
        for n_qubits in qubit_vals:
            fig.patches.append(mpl.patches.FancyBboxPatch(
                (x, legend_y - box_size / 2), box_size, box_size,
                boxstyle="square,pad=0",
                facecolor=color_map[n_qubits], edgecolor="none",
                transform=fig.transFigure, figure=fig,
            ))
            fig.text(
                x + box_size + 0.008, legend_y,
                f"$n={n_qubits}$", fontsize=10,
                va="center", ha="left",
            )
            x += box_size + 0.008 + len(f"n={n_qubits}") * 0.011 + pad

        if multi_norm:
            line_len = 0.03
            ls_pad = 0.015
            total_ls_width = sum(
                line_len + 0.008 + len(nv) * 0.009 + ls_pad
                for nv in norm_vals
            ) - ls_pad
            lx = 0.5 - total_ls_width / 2
            for nv in norm_vals:
                ls = norm_ls_map[nv]
                fig.lines.append(mpl.lines.Line2D(
                    [lx, lx + line_len], [norm_legend_y, norm_legend_y],
                    linestyle=ls, color="black", linewidth=1.5,
                    transform=fig.transFigure, figure=fig,
                ))
                fig.text(
                    lx + line_len + 0.008, norm_legend_y,
                    nv, fontsize=10,
                    va="center", ha="left",
                )
                lx += line_len + 0.008 + len(nv) * 0.009 + ls_pad

        fig.tight_layout(rect=[0, 0, 1, legend_y - 0.005])
        mow_slug = mow if isinstance(mow, str) else f"w{mow}"
        fname = output_dir / f"mmd_var_vs_scale_{ds_name}_{mow_slug}_exact.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {fname}")
        plt.close(fig)

# ── 2. Mean max|∇MMD| vs qubit number (grad_order=1) ────────────────────────

df_grad = df[df["grad_order"] == 1]

if not df_grad.empty:
    grad_strategy_order = ["random", "random_zero", "from_data_nocov", "from_data"]
    grad_colors = [
        "#5A7D9A", "#C2655A", "#6A9F6A", "#B8A252",
        "#7BAFC4", "#9C6B8A", "#999999",
    ]

    for (ds_name, scale, mow), mow_group in df_grad.groupby(
        ["dataset", "init_scale", "mean_op_weight"]
    ):
        norm_vals = sorted(mow_group["param_normalization"].unique())
        n_cols = len(norm_vals)
        if n_cols == 0:
            continue

        strat_vals = [s for s in grad_strategy_order
                      if s in mow_group["init_strategy"].unique()]
        strat_color = {s: grad_colors[i % len(grad_colors)]
                       for i, s in enumerate(strat_vals)}

        fig, axes = plt.subplots(
            1, n_cols, figsize=(5.5 * n_cols, 5),
            squeeze=False,
        )

        for col, pnorm in enumerate(norm_vals):
            ax = axes[0, col]
            group = mow_group[mow_group["param_normalization"] == pnorm]

            for strategy in strat_vals:
                strat_group = group[group["init_strategy"] == strategy]
                if strat_group.empty:
                    continue
                strat_group = strat_group.sort_values("n_qubits")
                display_name = strategy_labels.get(strategy, strategy)
                ax.errorbar(
                    strat_group["n_qubits"],
                    strat_group["value"],
                    yerr=strat_group["error"] if show_error_bars else None,
                    fmt="o-", capsize=4, linewidth=1.5,
                    color=strat_color[strategy],
                    label=display_name if col == 0 else "_nolegend_",
                )

            ax.set_title(pnorm, fontsize=13, fontstyle="italic")
            qubit_values = sorted(group["n_qubits"].unique())
            if log_scale_x:
                ax.set_xscale("log")
            max_ticks = 8
            if len(qubit_values) > max_ticks:
                step = len(qubit_values) // max_ticks + 1
                shown_ticks = qubit_values[::step]
                if qubit_values[-1] not in shown_ticks:
                    shown_ticks.append(qubit_values[-1])
            else:
                shown_ticks = qubit_values
            ax.set_xticks(shown_ticks)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            if log_scale_y:
                ax.set_yscale("log")
            else:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Number of qubits", fontsize=12)
            if col == 0:
                ax.set_ylabel(
                    r"$\mathrm{mean}_{\theta}\;\max_i |\partial_i \mathrm{MMD}^2|$",
                    fontsize=12,
                )
                ax.legend(fontsize=11)

        mow_label = f"$w={mow}$" if isinstance(mow, (int, float)) else str(mow)
        fig.suptitle(
            f"Max Gradient Magnitude vs Qubits — {ds_name}, scale={scale} ({mow_label})",
            fontsize=14, y=1.02,
        )
        fig.tight_layout()

        mow_slug = mow if isinstance(mow, str) else f"w{mow}"
        fname = output_dir / f"mmd_max_grad_{ds_name}_scale{scale}_{mow_slug}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {fname}")
        plt.close(fig)

# ── 3. Training loss curves ──────────────────────────────────────────────────

training_results_file = output_dir / config.get(
    "training_results_file", "training_results.pkl"
)

if training_results_file.exists():
    with open(training_results_file, "rb") as f:
        training_results = pickle.load(f)
    print(f"\nLoaded {len(training_results)} training entries from {training_results_file}")

    train_strategy_order = ["random", "random_zero", "from_data_nocov", "from_data"]
    train_colors = [
        "#5A7D9A", "#C2655A", "#6A9F6A", "#B8A252",
        "#7BAFC4", "#9C6B8A", "#999999",
    ]

    plot_groups = {}
    for entry in training_results:
        plot_key = (entry["dataset"], entry["mean_op_weight"], entry["n_qubits"])
        plot_groups.setdefault(plot_key, []).append(entry)

    for (ds_name, mow, n_qubits), entries in sorted(plot_groups.items(),
                                                      key=lambda kv: tuple(str(v) for v in kv[0])):
        pnorm_entries = {}
        for entry in entries:
            pnorm_entries.setdefault(entry.get("param_normalization", "unknown"), []).append(entry)

        norm_vals = sorted(pnorm_entries.keys())
        n_cols = len(norm_vals)
        if n_cols == 0:
            continue

        all_strategies = set()
        for elist in pnorm_entries.values():
            for e in elist:
                all_strategies.add(e["init_strategy"])
        strategies_present = [s for s in train_strategy_order if s in all_strategies]
        if not strategies_present:
            continue

        strat_color = {s: train_colors[i % len(train_colors)]
                       for i, s in enumerate(strategies_present)}

        fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5), squeeze=False)

        for col, pnorm in enumerate(norm_vals):
            ax = axes[0, col]

            col_strat_entries = {}
            for entry in pnorm_entries[pnorm]:
                col_strat_entries.setdefault(entry["init_strategy"], []).append(entry)

            for strategy in strategies_present:
                if strategy not in col_strat_entries:
                    continue
                label = strategy_labels.get(strategy, strategy)
                color = strat_color[strategy]

                all_losses = []
                for entry in col_strat_entries[strategy]:
                    losses_arr = entry["losses"]
                    if losses_arr.ndim == 1:
                        losses_arr = losses_arr[np.newaxis, :]
                    for trial_losses in losses_arr:
                        ax.plot(
                            trial_losses, color=color, alpha=0.5, linewidth=0.1,
                        )
                        all_losses.append(trial_losses)

                if all_losses and train_show_mean:
                    max_len = max(len(l) for l in all_losses)
                    padded = np.full((len(all_losses), max_len), np.nan)
                    for i, l in enumerate(all_losses):
                        padded[i, :len(l)] = l
                    mean_loss = np.nanmean(padded, axis=0)
                    ax.plot(mean_loss, color=color, alpha=1.0, linewidth=1.2,
                            label=label)

            ax.set_title(pnorm, fontsize=13, fontstyle="italic")
            ax.set_xlabel("Training step", fontsize=12)
            if col == 0:
                ax.set_ylabel("MMD$^2$ loss", fontsize=12)
            if train_log_y:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=11)

        mow_label = f"$w={mow}$" if isinstance(mow, (int, float)) else str(mow)
        fig.suptitle(
            f"Training Loss — {ds_name}, $n={n_qubits}$ ({mow_label})",
            fontsize=14, y=1.02,
        )
        fig.tight_layout()

        mow_slug = mow if isinstance(mow, str) else f"w{mow}"
        fname = output_dir / f"train_loss_{ds_name}_n{n_qubits}_{mow_slug}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {fname}")
        plt.close(fig)
else:
    print(f"\nNo training results file found at {training_results_file}, skipping training plots.")
