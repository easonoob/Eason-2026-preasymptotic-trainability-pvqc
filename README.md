# Pre-Asymptotic Trainability in Photonic Variational Circuits under Postselection
## Eason Xie, Cassandre Notton, Jean Senellart (2026)

Reproduces **Figure 2** (mean gradient variance vs. qubit count across post-selection schemes, comparing non-Haar and uniform initializations).

---

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install numpy matplotlib scipy torch perceval-quandela==0.13.1 merlinquantum==0.3.1
```

---

### Run

Examples:
```bash
python scripts/experiment.py --qubit-start 5 --qubit-end 12
python scripts/experiment.py --qubits 5,6,7,8,9,10,11,12
python scripts/experiment.py --qubit-start 5 --qubit-end 12 --samples 2000 --float64
```

All `theta_init` x `encoding` combinations run automatically. Progress prints every 10% of each config.

---

### Output

Each run creates `results/<YYYYMMDD_HHMMSS>_variance/` containing:

| File | Contents |
|---|---|
| `config.json` | Run configuration |
| `results.csv` | `qubits, theta_init, encoding, gradient_variance_mean, valid_samples, status` |
| `gradient_stats.npz` | Raw variance arrays |
| `variance_plots.pdf` | Figure 2 |

Use `--output-root <dir>` to change the output location.

---

### Plot Initialization Comparison

`scripts/plot_variance_across_init.py` compares the non-Haar initialization against the uniform initialization.

The plotting input root must contain one folder per initialization:

```text
<results-root>/
  non-haar/
    fock.csv
    unbunched.csv
    dual_rail.csv
  uniform/
    fock.csv
    unbunched.csv
    dual_rail.csv
```

Run:

```bash
python scripts/plot_variance_across_init.py --results-root <results-root>
```

Useful options:

```bash
python scripts/plot_variance_across_init.py --results-root <results-root> --depth 1
python scripts/plot_variance_across_init.py --results-root <results-root> --norm
python scripts/plot_variance_across_init.py --results-root <results-root> --loglog
```

For variance plots, the script fits two scaling models on the averaged data:

| Model | Fit performed | Interpretation |
|---|---|---|
| Exponential | Linear fit of `log10(variance)` against `qubits` | Exponential decay or growth with qubit count |
| Polynomial | Linear fit of `log10(variance)` against `log10(qubits)` | Power-law scaling with qubit count |

The displayed fit is selected by lowest AIC. Dual rail uses the exponential fit by default; pass `--fitdr` to include the polynomial comparison for dual rail too.

In practice, the script:

1. Reads each `encoding.csv` file for one initialization.
2. Keeps the rows matching `--depth`, or the nearest available depth if the exact value is missing.
3. Groups repeated samples by `qubits` and averages `gradient_variance_mean`.
4. Fits the averaged points with [`numpy.polyfit`](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html):
   - exponential fit: `np.polyfit(qubits, log10(variance), 1)`
   - polynomial fit: `np.polyfit(log10(qubits), log10(variance), 1)`
5. Converts the fitted log-scale curve back with `10 ** fitted_log10_variance`.
6. Computes AIC for each candidate fit and plots the lowest-AIC model.

By default, the script writes `gradient_variance_all_depth_<depth>v2.png` under `<results-root>`. With `--norm`, it writes `gradient_norm_rms_all_depth_<depth>v2.png`.