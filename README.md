# Pre-Asymptotic Trainability in Photonic Variational Circuits under Postselection
## Eason Xie, Cassandre Notton, Jean Senellart (2026)

Reproduces **Figure 2** (mean gradient variance vs. qubit count across post-selection schemes and initializations).

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

All `theta_init` × `encoding` combinations run automatically. Progress prints every 10% of each config.

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