import torch
import torch.nn as nn
import perceval as pcvl
import merlin as ML
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

n_samples = 5000
theta_inits = ["uniform", "arcsin"]
encodings = ["dual-rail", "fock", "unbunched"]

def parse_args():
    parser = argparse.ArgumentParser(description="Run photonic gradient variance experiment.")
    parser.add_argument("--qubits", type=str, default=None,
        help="Comma-separated list of qubit counts to evaluate (e.g. '2,4,6'). Overrides --qubit-start/--qubit-end.")
    parser.add_argument("--qubit-start", type=int, default=2,
        help="Start of qubit range (inclusive). Used when --qubits is not specified. Default: 2.")
    parser.add_argument("--qubit-end", type=int, default=12,
        help="End of qubit range (inclusive). Used when --qubits is not specified. Default: 12.")
    parser.add_argument("--samples", type=int, default=n_samples,
        help=f"Number of random parameter samples per (theta_init, encoding, qubit) configuration. Default: {n_samples}.")
    parser.add_argument("--output-root", type=str, default="results",
        help="Root directory under which a timestamped run folder will be created. Default: 'results'.")
    parser.add_argument("--float64", action="store_true",
        help="Use float64 precision instead of the default float32.")
    return parser.parse_args()

def resolve_qubit_list(args):
    if args.qubits:
        qubits = [int(x.strip()) for x in args.qubits.split(",") if x.strip()]
    else:
        qubits = list(range(args.qubit_start, args.qubit_end + 1))
    if not qubits:
        raise ValueError
    if any(q <= 0 for q in qubits):
        raise ValueError
    return qubits

def create_run_dir(output_root):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{timestamp}_variance"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def save_config_json(run_dir, args, qubit_list, n_samples):
    config = {
        "qubits": qubit_list,
        "theta_inits": theta_inits,
        "encodings": encodings,
        "n_samples": n_samples,
        "device": str(device),
        "float64": args.float64,
        "depth": 1.0,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

def init_results_csv(run_dir):
    with open(run_dir / "results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qubits","theta_init","encoding","gradient_variance_mean","valid_samples","status"])

def append_result_csv(run_dir, n_qubits, theta_init, encoding, var, valid_samples, status):
    with open(run_dir / "results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([n_qubits,theta_init,encoding,"" if np.isnan(var) else float(var),valid_samples,status])

def save_checkpoint_npz(run_dir, results_var, qubit_list):
    np.savez(run_dir / "gradient_stats.npz",variance=results_var,qubits=np.array(qubit_list))

def create_pvqc_ansatz(n_modes, depth):
    return pcvl.GenericInterferometer(
        n_modes,
        lambda i: pcvl.BS(pcvl.P(f"phi_{2*i}")) // pcvl.PS(pcvl.P(f"phi_{2*i+1}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
        depth=depth,
    )

class PVQC_Model(nn.Module):
    def __init__(self, n_qubits, encoding, theta_init, dtype):
        super().__init__()
        total_modes = 2 * n_qubits
        depth = total_modes
        n_params = math.ceil(depth / 2) * total_modes + (depth // 2) * (total_modes - 2)

        self.n_params = n_params
        self.depth = depth
        self.total_modes = total_modes
        self.theta_init = theta_init
        self.dtype = dtype

        initial_state = pcvl.BasicState([1, 0] * n_qubits)
        circuit = create_pvqc_ansatz(total_modes, depth)

        if encoding == "unbunched":
            measurement = ML.MeasurementStrategy.probs(ML.ComputationSpace.UNBUNCHED)
        elif encoding == "dual-rail":
            measurement = ML.MeasurementStrategy.probs(ML.ComputationSpace.DUAL_RAIL)
        else:
            measurement = ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK)

        self.quantum_layer = ML.QuantumLayer(
            circuit=circuit,
            input_parameters=["phi"],
            input_state=initial_state,
            device=device,
            measurement_strategy=measurement,
            dtype=dtype,
        )

    def forward(self):
        self.zero_grad(set_to_none=True)
        with torch.no_grad():
            if self.theta_init == "arcsin":
                thetas = torch.arcsin(torch.sqrt(torch.rand(self.n_params, device=device, dtype=self.dtype)))
            else:
                thetas = torch.rand(self.n_params, device=device, dtype=self.dtype) * (np.pi / 2)

        thetas.requires_grad_(True)
        psi = self.quantum_layer(thetas).squeeze(0).to(self.dtype)
        psi = psi / psi.sum()
        psi = torch.clamp(psi, min=1e-20)
        with torch.no_grad():
            target = torch.rand_like(psi)
            target = target / target.sum()
        loss = 1 - torch.sum(torch.sqrt(psi * target))**2
        loss.backward()
        return thetas.grad

args = parse_args()
dtype = torch.float64 if args.float64 else torch.float32
torch.set_default_dtype(dtype)

qubit_list = resolve_qubit_list(args)
n_samples = args.samples
print_every = max(1, n_samples // 10)

run_dir = create_run_dir(args.output_root)
save_config_json(run_dir, args, qubit_list, n_samples)
init_results_csv(run_dir)

results_var = np.full((len(theta_inits), len(encodings), len(qubit_list)), np.nan)

for ti, theta_init in enumerate(theta_inits):
    for ei, encoding in enumerate(encodings):
        for qi, n_qubits in enumerate(qubit_list):
            pvqc_model = PVQC_Model(n_qubits, encoding, theta_init, dtype).to(device)
            all_vars = []
            print(f"\n[Config] theta_init={theta_init!r}  encoding={encoding!r}  n_qubits={n_qubits}")
            for s in range(n_samples):
                gradients = pvqc_model()
                if not torch.isnan(gradients).any():
                    all_vars.append(gradients.detach().cpu().numpy())
                if (s + 1) % print_every == 0 or (s + 1) == n_samples:
                    running_var = np.mean(np.var(np.array(all_vars), axis=0)) if len(all_vars) >= 2 else float("nan")
                    print(f"  sample {s+1:>{len(str(n_samples))}}/{n_samples}  "
                          f"valid={len(all_vars)}  running_var={running_var:.4e}")
            if len(all_vars) == 0:
                append_result_csv(run_dir,n_qubits,theta_init,encoding,np.nan,0,"no_valid_gradients")
                continue
            all_vars = np.array(all_vars)
            var_all_mean = np.mean(np.var(all_vars, axis=0))
            results_var[ti, ei, qi] = var_all_mean
            append_result_csv(run_dir,n_qubits,theta_init,encoding,var_all_mean,len(all_vars),"ok")

save_checkpoint_npz(run_dir, results_var, qubit_list)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

x = np.array(qubit_list, dtype=float)
xfit = np.linspace(x.min(), x.max(), 300)

for ti, theta_init in enumerate(theta_inits):
    ax = axes[ti]
    for ei, encoding in enumerate(encodings):
        y = results_var[ti, ei, :]
        valid = np.isfinite(y) & (y > 0)
        line, = ax.plot(x, y, marker="o", linewidth=2, label=encoding)
        color = line.get_color()
        if valid.sum() >= 2:
            if encoding == "dual-rail":
                coeffs = np.polyfit(x[valid], np.log(y[valid]), 1)
                yfit = np.exp(np.polyval(coeffs, xfit))
            else:
                deg = min(2, valid.sum() - 1)
                coeffs = np.polyfit(x[valid], y[valid], deg)
                yfit = np.polyval(coeffs, xfit)
                yfit = np.clip(yfit, 1e-30, None)
            ax.plot(xfit, yfit, linestyle="--", linewidth=2, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("Qubits")
    ax.set_title(theta_init)
    ax.grid(True)

axes[0].set_ylabel("Mean Gradient Variance")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.05))

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(run_dir / "variance_plots.pdf")
plt.close()

print(f"\nExperiment completed. Results saved to: {run_dir}")