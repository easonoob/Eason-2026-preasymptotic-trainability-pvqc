#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter


PLOT_BLUE = "#435BEC"
PLOT_WHITE = "#FFFFFF"
PLOT_BLACK = "#000000"
PLOT_PURPLE = "#6A4C93"
PLOT_CORAL = "#FF6F61"

DISTRIBUTION_ORDER = ["non-haar", "uniform"]
DISTRIBUTION_TEXT_LABELS = {
    "non-haar": "asin(sqrt(U)) init",
    "uniform": "Uniform [0, 2pi) init",
}
DISTRIBUTION_PLOT_LABELS = {
    "non-haar": r"$\arcsin(\sqrt{U})$ init",
    "uniform": r"$U(0, 2\pi)$ init",
}

ENCODING_ORDER = ["fock", "unbunched", "dual_rail"]
ENCODING_LABELS = {
    "fock": "Fock",
    "unbunched": "Unbunched",
    "dual_rail": "Dual rail",
}
ENCODING_COLORS = {
    "fock": PLOT_BLUE,
    "unbunched": PLOT_CORAL,
    "dual_rail": PLOT_PURPLE,
}
ENCODING_MARKERS = {
    "fock": "o",
    "unbunched": "o",
    "dual_rail": "o",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot gradient variance or RMS gradient norm and fitted curves for the Fock, "
            "unbunched, and dual-rail CSV files in results/{non-haar,uniform}."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory that contains the non-haar/ and uniform/ folders.",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Plot gradient_norm_rms instead of gradient_variance_mean.",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=1.0,
        help="Target depth to plot (default: 1.0).",
    )
    parser.add_argument(
        "--depth-tol",
        type=float,
        default=1e-9,
        help="Tolerance for matching depth (default: 1e-9).",
    )
    parser.add_argument(
        "--only-ok",
        action="store_true",
        help="Use only rows where status == 'ok' if the column exists.",
    )
    parser.add_argument(
        "--fit-degree",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help="Deprecated compatibility option. Ignored by the current variance-fit workflow.",
    )
    parser.add_argument(
        "--fitdr",
        action="store_true",
        help="Also compare polynomial-vs-exponential models for dual rail. By default dual rail stays exponential.",
    )
    parser.add_argument(
        "--loglog",
        action="store_true",
        help="Use log-log axes for the plot panels. Useful for visualizing polynomial scaling.",
    )
    parser.add_argument(
        "--hide-bubbles",
        action="store_true",
        help="Disable the faint subspace-dimension circle overlay.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default depends on whether variance or --norm is selected).",
    )
    return parser.parse_args()


def metric_config(args: argparse.Namespace) -> dict[str, str]:
    if args.norm:
        return {
            "column": "gradient_norm_rms",
            "symbol": r"\mathrm{norm}_{\mathrm{rms}}",
            "y_label": "RMS gradient norm",
            "title_name": "RMS gradient norm",
            "filename_stem": "gradient_norm_rms_all",
            "with_fit": "false",
        }
    return {
        "column": "gradient_variance_mean",
        "symbol": r"\mathrm{var}",
        "y_label": "Mean gradient variance",
        "title_name": "Gradient variance",
        "filename_stem": "gradient_variance_all",
        "with_fit": "true",
    }


def should_fit(metric: dict[str, str]) -> bool:
    return metric["with_fit"] == "true"


def required_csv_paths(results_root: Path) -> list[tuple[str, str, Path]]:
    paths: list[tuple[str, str, Path]] = []
    for distribution in DISTRIBUTION_ORDER:
        for encoding in ENCODING_ORDER:
            paths.append(
                (
                    distribution,
                    encoding,
                    results_root / distribution / f"{encoding}.csv",
                )
            )
    return paths


def load_rows(csv_path: Path, metric_column: str) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    required = {"qubits", "depth", metric_column}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    return rows


def clean_qubits(raw_value: object, csv_path: Path) -> int:
    text = str(raw_value).strip()
    match = re.search(r"-?\d+", text)
    if match is None:
        raise ValueError(f"Could not parse qubits value {text!r} in {csv_path}")
    return int(match.group(0))


def parse_float(raw_value: object, field_name: str, csv_path: Path) -> float:
    try:
        return float(str(raw_value).strip())
    except ValueError as exc:
        raise ValueError(
            f"Could not parse {field_name} value {raw_value!r} in {csv_path}"
        ) from exc


def select_depth_rows(
    rows: list[dict[str, str]],
    csv_path: Path,
    target_depth: float,
    depth_tol: float,
    only_ok: bool,
) -> tuple[list[dict[str, str]], float]:
    filtered = rows
    if only_ok and rows and "status" in rows[0]:
        filtered = [row for row in rows if str(row.get("status", "")).strip() == "ok"]
        if not filtered:
            raise ValueError(f"No rows with status == 'ok' in {csv_path}")

    depths = np.array(
        [parse_float(row["depth"], "depth", csv_path) for row in filtered], dtype=float
    )
    exact_mask = np.isclose(depths, target_depth, atol=depth_tol, rtol=0.0)
    if np.any(exact_mask):
        return [row for row, keep in zip(filtered, exact_mask) if keep], float(
            target_depth
        )

    unique_depths = np.unique(depths)
    nearest_depth = float(
        unique_depths[np.argmin(np.abs(unique_depths - target_depth))]
    )
    selected_rows = [
        row
        for row, depth in zip(filtered, depths)
        if np.isclose(depth, nearest_depth, atol=depth_tol, rtol=0.0)
        or depth == nearest_depth
    ]
    return selected_rows, nearest_depth


def aggregate_by_qubits(
    rows: list[dict[str, str]],
    csv_path: Path,
    metric_column: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    grouped: dict[int, list[float]] = {}
    dropped_rows = 0
    for row in rows:
        qubits = clean_qubits(row["qubits"], csv_path)
        raw_metric = str(row[metric_column]).strip()
        if raw_metric == "":
            dropped_rows += 1
            continue
        metric_value = parse_float(raw_metric, metric_column, csv_path)
        if not np.isfinite(metric_value):
            dropped_rows += 1
            continue
        grouped.setdefault(qubits, []).append(metric_value)

    if not grouped:
        raise ValueError(f"No finite {metric_column} values found in {csv_path}")

    ordered_qubits = np.array(sorted(grouped), dtype=int)
    mean_metric = np.array([np.mean(grouped[q]) for q in ordered_qubits], dtype=float)
    return ordered_qubits, mean_metric, dropped_rows


def aggregate_secondary_by_qubits(
    rows: list[dict[str, str]],
    csv_path: Path,
    value_column: str,
) -> np.ndarray:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        qubits = clean_qubits(row["qubits"], csv_path)
        raw_value = str(row.get(value_column, "")).strip()
        if raw_value == "":
            continue
        value = parse_float(raw_value, value_column, csv_path)
        if not np.isfinite(value):
            continue
        grouped.setdefault(qubits, []).append(value)

    ordered_qubits = np.array(sorted(grouped), dtype=int)
    return np.array([np.mean(grouped[q]) for q in ordered_qubits], dtype=float)


def fit_polynomial_log10(
    qubits: np.ndarray, metric_values: np.ndarray, degree: int, metric_column: str
) -> np.ndarray:
    if np.any(metric_values <= 0):
        raise ValueError(
            f"{metric_column} contains non-positive values; cannot fit log10 polynomial."
        )
    if qubits.size < degree + 1:
        raise ValueError(
            f"Need at least {degree + 1} data points for degree-{degree} polynomial fit; got {qubits.size}."
        )
    return np.polyfit(qubits.astype(float), np.log10(metric_values), degree)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


def aic(y_true: np.ndarray, y_pred: np.ndarray, num_params: int) -> float:
    residuals = y_true - y_pred
    n = y_true.size
    rss = float(np.sum(residuals**2))
    rss = max(rss, np.finfo(float).tiny)
    return float(n * np.log(rss / n) + 2 * num_params)


def fit_exponential_model(
    qubits: np.ndarray, metric_values: np.ndarray, metric_column: str
) -> dict[str, Any]:
    if np.any(metric_values <= 0):
        raise ValueError(
            f"{metric_column} contains non-positive values; cannot fit exponential model."
        )
    x = qubits.astype(float)
    y = np.log10(metric_values)
    slope, intercept = np.polyfit(x, y, 1)
    fit_x = np.linspace(float(x.min()), float(x.max()), 250)
    fit_log10 = intercept + slope * fit_x
    fitted_at_points = intercept + slope * x
    return {
        "model": "exponential",
        "fit_x": fit_x,
        "fit_curve": 10**fit_log10,
        "r2": r2_score(y, fitted_at_points),
        "aic": aic(y, fitted_at_points, num_params=2),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def fit_powerlaw_model(
    qubits: np.ndarray, metric_values: np.ndarray, metric_column: str
) -> dict[str, Any]:
    if np.any(metric_values <= 0):
        raise ValueError(
            f"{metric_column} contains non-positive values; cannot fit power-law model."
        )
    if np.any(qubits <= 0):
        raise ValueError("qubits must be positive for log-log fitting.")
    x = np.log10(qubits.astype(float))
    y = np.log10(metric_values)
    slope, intercept = np.polyfit(x, y, 1)
    fit_x = np.linspace(float(qubits.min()), float(qubits.max()), 250)
    fit_log10 = intercept + slope * np.log10(fit_x)
    fitted_at_points = intercept + slope * x
    return {
        "model": "polynomial",
        "fit_x": fit_x,
        "fit_curve": 10**fit_log10,
        "r2": r2_score(y, fitted_at_points),
        "aic": aic(y, fitted_at_points, num_params=2),
        "slope": float(slope),
        "intercept": float(intercept),
        "exponent": float(-slope),
    }


def select_preferred_fit(candidate_fits: list[dict[str, Any]]) -> dict[str, Any]:
    return min(candidate_fits, key=lambda fit: float(fit["aic"]))


def format_polynomial(coeffs: np.ndarray) -> str:
    degree = len(coeffs) - 1
    parts: list[str] = []
    for idx, coeff in enumerate(coeffs):
        power = degree - idx
        if power == 0:
            term = f"{coeff:.6g}"
        elif power == 1:
            term = f"{coeff:.6g}*q"
        else:
            term = f"{coeff:.6g}*q^{power}"
        parts.append(term)
    return " + ".join(parts)


def _format_math_number(value: float) -> str:
    text = f"{value:.3g}"
    if "e" not in text and "E" not in text:
        return text
    mantissa, exponent = text.lower().split("e")
    exponent_i = int(exponent)
    if np.isclose(float(mantissa), 1.0):
        return rf"10^{{{exponent_i}}}"
    return rf"{mantissa} \times 10^{{{exponent_i}}}"


def format_polynomial_math(coeffs: np.ndarray, variable: str = "q") -> str:
    degree = len(coeffs) - 1
    parts: list[tuple[str, str]] = []
    for idx, coeff in enumerate(coeffs):
        if np.isclose(coeff, 0.0):
            continue
        power = degree - idx
        magnitude = abs(float(coeff))
        coeff_text = _format_math_number(magnitude)
        if power == 0:
            term = coeff_text
        elif power == 1:
            term = rf"{coeff_text}\,{variable}"
        else:
            term = rf"{coeff_text}\,{variable}^{{{power}}}"
        parts.append(("-" if coeff < 0 else "+", term))

    if not parts:
        return "0"

    first_sign, first_term = parts[0]
    expression = first_term if first_sign == "+" else rf"-{first_term}"
    for sign, term in parts[1:]:
        expression += rf" {sign} {term}"
    return expression


def compute_subspace_dimension(encoding: str, qubits: int) -> int:
    modes = 2 * qubits
    photons = qubits
    if encoding == "fock":
        return int(math.comb(modes + photons - 1, photons))
    if encoding == "unbunched":
        return int(math.comb(modes, photons))
    if encoding == "dual_rail":
        return int(2**photons)
    raise ValueError(f"Unsupported encoding: {encoding}")


def compute_parameter_count(qubits: int, depth: float) -> int:
    total_modes = 2 * qubits
    effective_depth = int(depth * total_modes)
    return effective_depth * total_modes


def scale_bubble_sizes(
    all_dimensions: list[int], size_min: float = 170.0, size_max: float = 1200.0
) -> tuple[float, float]:
    logs = np.log10(np.asarray(all_dimensions, dtype=float))
    return float(logs.min()), float(logs.max())


def bubble_sizes_for_curve(
    dimensions: np.ndarray, log_dim_min: float, log_dim_max: float
) -> np.ndarray:
    logs = np.log10(dimensions.astype(float))
    if np.isclose(log_dim_min, log_dim_max):
        return np.full(logs.shape, 500.0, dtype=float)
    return np.interp(logs, (log_dim_min, log_dim_max), (170.0, 1200.0))


def load_curve(
    distribution: str,
    encoding: str,
    csv_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metric = metric_config(args)
    rows = load_rows(csv_path, metric["column"])
    selected_rows, used_depth = select_depth_rows(
        rows=rows,
        csv_path=csv_path,
        target_depth=args.depth,
        depth_tol=args.depth_tol,
        only_ok=args.only_ok,
    )
    qubits, metric_values, dropped_rows = aggregate_by_qubits(
        selected_rows, csv_path, metric["column"]
    )
    dimensions = np.array(
        [compute_subspace_dimension(encoding, int(q)) for q in qubits], dtype=float
    )
    parameter_counts = np.array(
        [compute_parameter_count(int(q), used_depth) for q in qubits], dtype=float
    )
    variance_values = None
    norm_comparison = None
    if args.norm:
        variance_values = aggregate_secondary_by_qubits(
            selected_rows, csv_path, "gradient_variance_mean"
        )
        if variance_values.shape != metric_values.shape or not np.array_equal(
            qubits, qubits.astype(int)
        ):
            variance_values = variance_values
        if variance_values.shape == metric_values.shape:
            norm_comparison = np.sqrt(parameter_counts * variance_values)
    if should_fit(metric):
        model_fits: dict[str, dict[str, Any]] = {
            "exponential": fit_exponential_model(
                qubits, metric_values, metric["column"]
            )
        }
        if encoding != "dual_rail" or args.fitdr:
            model_fits["polynomial"] = fit_powerlaw_model(
                qubits, metric_values, metric["column"]
            )
        preferred_fit = select_preferred_fit(list(model_fits.values()))
        fit_x = np.asarray(preferred_fit["fit_x"], dtype=float)
        fit_curve = np.asarray(preferred_fit["fit_curve"], dtype=float)
        fit_r2 = float(preferred_fit["r2"])
    else:
        model_fits = {}
        preferred_fit = None
        fit_x = None
        fit_curve = None
        fit_r2 = None
    return {
        "distribution": distribution,
        "encoding": encoding,
        "csv_path": csv_path,
        "used_depth": used_depth,
        "qubits": qubits,
        "metric_column": metric["column"],
        "metric_values": metric_values,
        "fit_degree": None,
        "coeffs": None,
        "fit_x": fit_x,
        "fit_curve": fit_curve,
        "fit_r2": fit_r2,
        "dimensions": dimensions,
        "parameter_counts": parameter_counts,
        "variance_values": variance_values,
        "norm_comparison": norm_comparison,
        "dropped_rows": dropped_rows,
        "model_fits": model_fits,
        "preferred_fit": preferred_fit,
    }


def summary_line(curve: dict[str, Any]) -> str:
    preferred_fit = curve["preferred_fit"]
    if preferred_fit is None:
        return f"{ENCODING_LABELS[str(curve['encoding'])]}: no fit"
    if str(preferred_fit["model"]) == "polynomial":
        return f"{ENCODING_LABELS[str(curve['encoding'])]}: polynomial preferred"
    return (
        f"{ENCODING_LABELS[str(curve['encoding'])]}: "
        f"exponential fit, slope={float(preferred_fit['slope']):.4g}"
    )


def summary_text(curve: dict[str, Any]) -> str:
    preferred_fit = curve["preferred_fit"]
    if preferred_fit is None:
        return "No fit"
    if str(preferred_fit["model"]) == "polynomial":
        return "polynomial preferred\nover tested sizes"
    return f"exponential fit retained\nslope={float(preferred_fit['slope']):.3g}, $R^2$={float(preferred_fit['r2']):.3f}"


def render_plot(
    curves: list[dict[str, Any]],
    output_path: Path,
    args: argparse.Namespace,
    metric: dict[str, str],
) -> None:
    with_fit = should_fit(metric)
    curves_by_distribution = {
        distribution: [
            curve for curve in curves if curve["distribution"] == distribution
        ]
        for distribution in DISTRIBUTION_ORDER
    }
    all_dimensions = [
        int(dim)
        for curve in curves
        for dim in np.asarray(curve["dimensions"], dtype=float)
    ]
    log_dim_min, log_dim_max = scale_bubble_sizes(all_dimensions)
    max_qubits = max(
        int(np.max(np.asarray(curve["qubits"], dtype=int))) for curve in curves
    )
    min_metric = min(
        float(np.min(np.asarray(curve["metric_values"], dtype=float)))
        for curve in curves
    )
    max_metric = max(
        float(np.max(np.asarray(curve["metric_values"], dtype=float)))
        for curve in curves
    )

    num_distributions = len(DISTRIBUTION_ORDER)
    fig = plt.figure(figsize=(12, 9.2 if with_fit else 6.2))
    fig.patch.set_facecolor(PLOT_WHITE)
    if with_fit:
        grid = fig.add_gridspec(2, num_distributions, height_ratios=[4.0, 1.7])
    else:
        grid = fig.add_gridspec(1, num_distributions)

    plot_axes: list[plt.Axes] = []
    shared_ax: plt.Axes | None = None

    for col, distribution in enumerate(DISTRIBUTION_ORDER):
        ax = fig.add_subplot(grid[0, col] if with_fit else grid[col], sharey=shared_ax)
        if shared_ax is None:
            shared_ax = ax
        plot_axes.append(ax)

        ax.set_facecolor(PLOT_WHITE)
        ax.set_title(
            DISTRIBUTION_PLOT_LABELS[distribution],
            color=PLOT_BLACK,
            fontsize=13,
            pad=10,
        )
        ax.set_yscale("log")
        if args.loglog:
            ax.set_xscale("log")
            ax.set_xlim(0.95, max_qubits * 1.05)
            ax.set_xticks(
                [1, 2, 3, 5, 8, 10, 13, 21]
                if max_qubits >= 21
                else [1, 2, 3, 5, 8, 10, 13]
            )
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
        else:
            ax.set_xlim(0.5, max_qubits + 0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.set_ylim(min_metric * 0.7, max_metric * 1.5)
        ax.grid(True, which="major", color=PLOT_BLACK, alpha=0.15, linewidth=0.8)
        ax.grid(True, which="minor", color=PLOT_BLACK, alpha=0.06, linewidth=0.6)
        ax.tick_params(axis="both", colors=PLOT_BLACK)
        for spine in ax.spines.values():
            spine.set_color(PLOT_BLACK)

        if with_fit:
            summary_ax = fig.add_subplot(grid[1, col])
            summary_ax.set_facecolor(PLOT_WHITE)
            summary_ax.set_xlim(0.0, 1.0)
            summary_ax.set_ylim(0.0, 1.0)
            summary_ax.set_xticks([])
            summary_ax.set_yticks([])
            for spine in summary_ax.spines.values():
                spine.set_color(PLOT_BLACK)

        distribution_curves = curves_by_distribution[distribution]
        for curve in distribution_curves:
            encoding = str(curve["encoding"])
            color = ENCODING_COLORS[encoding]
            marker = ENCODING_MARKERS[encoding]
            qubits = np.asarray(curve["qubits"], dtype=float)
            metric_values = np.asarray(curve["metric_values"], dtype=float)
            dimensions = np.asarray(curve["dimensions"], dtype=float)

            if not args.hide_bubbles:
                ax.scatter(
                    qubits,
                    metric_values,
                    s=bubble_sizes_for_curve(dimensions, log_dim_min, log_dim_max),
                    c=color,
                    alpha=0.11,
                    linewidths=0,
                    zorder=1,
                )

            ax.plot(
                qubits,
                metric_values,
                color=color,
                marker=marker,
                markersize=7,
                markerfacecolor=PLOT_WHITE,
                markeredgecolor=color,
                markeredgewidth=1.4,
                linewidth=2.2,
                zorder=3,
            )
            if args.norm and curve["norm_comparison"] is not None:
                ax.plot(
                    qubits,
                    np.asarray(curve["norm_comparison"], dtype=float),
                    color=color,
                    linestyle=(0, (5, 3)),
                    linewidth=1.9,
                    alpha=0.92,
                    zorder=2,
                )
            if with_fit:
                ax.plot(
                    np.asarray(curve["fit_x"], dtype=float),
                    np.asarray(curve["fit_curve"], dtype=float),
                    color=color,
                    linestyle=(0, (5, 3)),
                    linewidth=1.9,
                    alpha=0.92,
                    zorder=2,
                )

        if with_fit:
            summary_ax.text(
                0.03,
                0.90,
                "Fit summary",
                color=PLOT_BLACK,
                fontsize=10,
                fontweight="bold",
                ha="left",
                va="top",
            )

            y = 0.74
            for curve in distribution_curves:
                summary_ax.text(
                    0.03,
                    y,
                    ENCODING_LABELS[str(curve["encoding"])],
                    color=ENCODING_COLORS[str(curve["encoding"])],
                    fontsize=8.8,
                    fontweight="bold",
                    ha="left",
                    va="top",
                )
                summary_ax.text(
                    0.03,
                    y - 0.10,
                    summary_text(curve),
                    color=ENCODING_COLORS[str(curve["encoding"])],
                    fontsize=7.2,
                    ha="left",
                    va="top",
                )
                y -= 0.24

    plot_axes[0].set_ylabel(metric["y_label"], color=PLOT_BLACK)
    for ax in plot_axes:
        ax.set_xlabel("Qubits", color=PLOT_BLACK)
    for ax in plot_axes[1:]:
        ax.tick_params(labelleft=False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=ENCODING_COLORS[encoding],
            marker=ENCODING_MARKERS[encoding],
            markerfacecolor=PLOT_WHITE,
            markeredgecolor=ENCODING_COLORS[encoding],
            markeredgewidth=1.4,
            linewidth=2.2,
            label=ENCODING_LABELS[encoding],
        )
        for encoding in ENCODING_ORDER
    ]
    if with_fit:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=PLOT_BLACK,
                linestyle=(0, (5, 3)),
                linewidth=1.9,
                label="Preferred fit curve",
            )
        )
    if args.norm:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=PLOT_BLACK,
                linestyle=(0, (5, 3)),
                linewidth=1.9,
                label=r"$\sqrt{P\,\widehat{V}}$ comparison",
            )
        )
    if not args.hide_bubbles:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=PLOT_BLACK,
                markeredgecolor="none",
                alpha=0.11,
                markersize=14,
                label="Circle size ~ subspace dimension",
            )
        )

    show_title = args.title is not None
    if show_title:
        fig.suptitle(args.title, color=PLOT_BLACK, fontsize=16, y=0.99)

    show_legend = not args.norm
    if show_legend:
        legend = fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=len(legend_handles),
            bbox_to_anchor=(0.5, 0.93),
            frameon=True,
            facecolor=PLOT_WHITE,
            edgecolor=PLOT_BLACK,
            title="Encoding / guide",
        )
        if legend.get_title() is not None:
            legend.get_title().set_color(PLOT_BLACK)
        for text in legend.get_texts():
            text.set_color(PLOT_BLACK)

    fig.subplots_adjust(
        left=0.05,
        right=0.995,
        bottom=0.10 if with_fit else 0.11,
        top=(
            0.78
            if show_title and with_fit
            else 0.82
            if with_fit
            else 0.80
            if show_title and show_legend
            else 0.85
            if show_legend
            else 0.92
        ),
        hspace=0.14,
        wspace=0.12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    metric = metric_config(args)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else results_root / f"{metric['filename_stem']}_depth_{args.depth:g}v2.png"
    )
    if args.output is None and args.loglog:
        output_path = (
            results_root
            / f"{metric['filename_stem']}_depth_{args.depth:g}_loglogv2.png"
        )

    curves = [
        load_curve(distribution, encoding, csv_path, args)
        for distribution, encoding, csv_path in required_csv_paths(results_root)
    ]
    render_plot(curves, output_path, args, metric)

    print(f"Saved plot: {output_path}")
    for distribution in DISTRIBUTION_ORDER:
        print(f"\n[{DISTRIBUTION_TEXT_LABELS[distribution]}]")
        for curve in [
            curve for curve in curves if curve["distribution"] == distribution
        ]:
            qubits = np.asarray(curve["qubits"], dtype=int)
            requested = args.depth
            used = float(curve["used_depth"])
            depth_note = ""
            if not np.isclose(used, requested, atol=args.depth_tol, rtol=0.0):
                depth_note = f", requested depth={requested:g}, used depth={used:g}"
            if should_fit(metric):
                preferred_fit = curve["preferred_fit"]
                model_fits = curve["model_fits"]
                if "polynomial" in model_fits:
                    poly_fit = model_fits["polynomial"]
                    exp_fit = model_fits["exponential"]
                    print(
                        f"  {ENCODING_LABELS[str(curve['encoding'])]}: "
                        f"preferred={preferred_fit['model']}, "
                        f"poly_exponent={float(poly_fit['exponent']):.6f}, "
                        f"poly_R2={float(poly_fit['r2']):.6f}, "
                        f"poly_AIC={float(poly_fit['aic']):.3f}, "
                        f"exp_slope={float(exp_fit['slope']):.6f}, "
                        f"exp_R2={float(exp_fit['r2']):.6f}, "
                        f"exp_AIC={float(exp_fit['aic']):.3f}, "
                        f"qubits=[{int(qubits.min())}, {int(qubits.max())}]"
                        f"{depth_note}"
                    )
                else:
                    print(
                        f"  {ENCODING_LABELS[str(curve['encoding'])]}: "
                        f"preferred=exponential, "
                        f"exp_slope={float(preferred_fit['slope']):.6f}, "
                        f"exp_R2={float(preferred_fit['r2']):.6f}, "
                        f"exp_AIC={float(preferred_fit['aic']):.3f}, "
                        f"qubits=[{int(qubits.min())}, {int(qubits.max())}]"
                        f"{depth_note}"
                    )
            else:
                metric_values = np.asarray(curve["metric_values"], dtype=float)
                print(
                    f"  {ENCODING_LABELS[str(curve['encoding'])]}: "
                    f"qubits=[{int(qubits.min())}, {int(qubits.max())}], "
                    f"{metric['column']} range=[{float(metric_values.min()):.6g}, {float(metric_values.max()):.6g}]"
                    f"{depth_note}"
                )
                if args.norm and curve["norm_comparison"] is not None:
                    comparison = np.asarray(curve["norm_comparison"], dtype=float)
                    params = np.asarray(curve["parameter_counts"], dtype=int)
                    print(
                        "    "
                        + rf"comparison sqrt(P*var) range=[{float(comparison.min()):.6g}, {float(comparison.max()):.6g}], "
                        + f"P range=[{int(params.min())}, {int(params.max())}]"
                    )
            if int(curve["dropped_rows"]) > 0:
                print(
                    f"    skipped {int(curve['dropped_rows'])} non-finite {metric['column']} row(s)"
                )


if __name__ == "__main__":
    main()
