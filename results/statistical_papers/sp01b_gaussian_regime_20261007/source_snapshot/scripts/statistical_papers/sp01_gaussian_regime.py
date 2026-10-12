"""Bounded assessment of the Gaussian limit, requiring only NumPy.

Run from the project root with Python 3 and NumPy. Existing outputs are never
overwritten. The integral is reimplemented from src/v5/normal_integral.py with
math.erfc, and checked against independently generated Gaussian vectors and
the full knockoff+ threshold. This is not a new toroidal simulation study.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/statistical_papers/sp01b_gaussian_regime.json"
SQRT_TWO = math.sqrt(2.0)
SQRT_TWO_PI = math.sqrt(2.0 * math.pi)
ERFC = np.frompyfunc(math.erfc, 1, 1)


def cdf(value: np.ndarray | float) -> np.ndarray:
    """Stable standard-normal CDF without a SciPy dependency."""
    return 0.5 * np.asarray(ERFC(-np.asarray(value) / SQRT_TWO), dtype=float)


def leading_run_fdr(variance: float, nodes: int, k: int = 10) -> float:
    """Two-dimensional integral for ten strong signals and ten nulls."""
    if not math.isfinite(variance) or variance < 0 or nodes < 2 or k != 10:
        raise ValueError("Require finite nonnegative variance, nodes >= 2, K=10")
    z, weight = leggauss(nodes)
    latent = 10.0 * z
    latent_weight = 10.0 * weight * np.exp(-latent**2 / 2.0) / SQRT_TWO_PI
    cutoff = 12.0 * math.sqrt(1.0 + variance)
    magnitude = (z + 1.0) * cutoff / 2.0
    magnitude_weight = weight * cutoff / 2.0
    mu = math.sqrt(variance) * latent[:, None]
    r = magnitude[None, :]
    density = np.exp(-(r + mu)**2 / 2.0) / SQRT_TWO_PI
    upper = cdf(mu - r)
    middle = np.maximum(0.0, cdf(r - mu) - cdf(-r - mu))
    conditional = 0.5 * cdf(math.sqrt(variance) * latent)**k
    for count in range(1, k):
        integrand = density * upper**count * middle**(k - 1 - count)
        conditional += (
            k * math.comb(k - 1, count) * count / (k + count)
            * np.sum(integrand * magnitude_weight[None, :], axis=1)
        )
    value = float(np.sum(latent_weight * conditional))
    if not math.isfinite(value) or not 0.0 <= value <= 0.5:
        raise ArithmeticError("Nonfinite or out-of-range Gaussian FDR integral")
    return value


def half_total_variation(eigenvalue: float) -> float:
    """Half TV between N(0,1) and N(0,eigenvalue), eigenvalue >= 1."""
    if eigenvalue < 1 or not math.isfinite(eigenvalue):
        raise ValueError("Require a finite eigenvalue >= 1")
    if eigenvalue == 1:
        return 0.0
    boundary = math.sqrt(eigenvalue * math.log(eigenvalue) / (eigenvalue - 1.0))
    return float(cdf(boundary) - cdf(boundary / math.sqrt(eigenvalue)))


def bisect(function, target: float, left: float, right: float) -> tuple[float, float]:
    """Bracket one numerical crossing; no global monotonicity assertion."""
    if not function(left) <= target <= function(right):
        raise ValueError("Registered bracket does not contain a crossing")
    for _ in range(36):
        middle = (left + right) / 2.0
        if function(middle) < target:
            left = middle
        else:
            right = middle
    return left, right


def direct_fdp(null_scores: np.ndarray, q: float, k: int) -> float:
    """Independent scalar implementation of the actual knockoff+ threshold."""
    scores = np.r_[np.repeat(1e6, k), null_scores]
    if np.max(np.abs(null_scores)) >= 1e6:
        raise ValueError("Strong-signal ordering is not realized in this check")
    for threshold in np.unique(np.abs(scores[scores != 0])):
        selected = scores >= threshold
        ratio = (1 + np.count_nonzero(scores <= -threshold)) / max(
            1, np.count_nonzero(selected)
        )
        if ratio <= q:
            return float(np.count_nonzero(selected[k:]) / np.count_nonzero(selected))
    return 0.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text())
    out = ROOT / "results/statistical_papers" / config["id"]
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    inputs = [CONFIG, Path(__file__).resolve(), ROOT / config["retained_calibration"],
              ROOT / "src/v5/normal_integral.py", ROOT / "manuscript/obstacles_v3.tex"]
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    for path in inputs[:3]:
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (out / "execution_start.json").write_text(json.dumps({
        "status": "started", "config": config, "input_sha256": manifest,
        "python": sys.executable, "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }, indent=2) + "\n")
    k, count = config["nulls"], config["draws"]
    variances = config["variances"]
    independent = sum(r / (k + r) * 2**(-r - 1) for r in range(k)) + 0.5 * 2**(-k)
    rows = []
    for variance in variances:
        values = [leading_run_fdr(variance, nodes) for nodes in config["quadrature_nodes"]]
        if not all(math.isfinite(value) for value in values):
            raise ArithmeticError("Nonfinite quadrature result")
        if abs(values[0] - values[1]) > config["checks"]["quadrature_difference_tolerance"]:
            raise AssertionError("Quadrature convergence criterion failed")
        rows.append({"common_variance": variance, "fdr_256": values[0],
                     "fdr_512": values[1], "tv_upper_bound": min(
                         0.5, independent + half_total_variation(1 + k * variance))})
    if abs(rows[0]["fdr_512"] - independent) > config["checks"]["zero_variance_absolute_tolerance"]:
        raise AssertionError("Fair-sign exact-value check failed")

    rng = np.random.default_rng(config["seed"])
    runs = np.empty((count, len(variances)), dtype=np.uint8)
    checks = 0
    for first in range(0, count, config["chunk"]):
        size = min(config["chunk"], count - first)
        original_noise = rng.normal(size=(size, k))
        shared_noise = rng.normal(size=(size, 1))
        for column, variance in enumerate(variances):
            scores = original_noise + math.sqrt(variance) * shared_noise
            ordered = np.take_along_axis(scores, np.argsort(-np.abs(scores), axis=1), axis=1)
            r = np.cumprod(ordered > 0, axis=1).sum(axis=1)
            runs[first:first + size, column] = r
            if first == 0:
                for row in range(config["checks"]["threshold_checks_per_variance"]):
                    if abs(direct_fdp(scores[row], config["q"], k) - r[row] / (k + r[row])) > 1e-14:
                        raise AssertionError("Full knockoff+ threshold comparison failed")
                    checks += 1
    np.savez_compressed(out / "individual_gaussian_runs.npz", positive_run=runs,
                        common_variances=np.asarray(variances))
    fdp = runs / (k + runs.astype(float))
    for column, row in enumerate(rows):
        paired = fdp[:, column] - fdp[:, 0]
        row.update({"gaussian_mc_fdr": float(fdp[:, column].mean()),
                    "mcse": float(fdp[:, column].std(ddof=1) / math.sqrt(count)),
                    "paired_difference_from_zero": float(paired.mean()),
                    "paired_mcse": float(paired.std(ddof=1) / math.sqrt(count))})
        if abs(row["gaussian_mc_fdr"] - row["fdr_512"]) > (
            config["checks"]["monte_carlo_standard_errors"] * row["mcse"] + 1e-7
        ):
            raise AssertionError("Independent Gaussian Monte Carlo check failed")
    crossing = bisect(lambda t: leading_run_fdr(t, 128), config["q"],
                      *config["crossing_bracket"])
    center = sum(crossing) / 2.0
    crossing_validation = {str(nodes): leading_run_fdr(center, nodes) for nodes in [256, 512]}
    if any(not math.isfinite(value) or abs(value - config["q"]) > 1e-7
           for value in crossing_validation.values()):
        raise AssertionError("Numerical crossing validation failed")
    tv_crossing = bisect(half_total_variation, config["q"] - independent, 1.0, 3.0)
    eigenvalue = sum(tv_crossing) / 2.0
    calibration = json.loads(inputs[2].read_text())
    summary = {
        "status": "completed", "config": config, "input_sha256": manifest,
        "fair_sign_fdr_exact_formula": independent, "full_threshold_checks": checks,
        "rows": rows, "numerical_crossing_variance_bracket": crossing,
        "crossing_quadrature_validation": crossing_validation,
        "crossing_scope": "One numerical crossing in the registered bracket; no proof of global monotonicity or interval certification.",
        "tv_bound_eigenvalue_crossing": tv_crossing,
        "tv_sufficient_rho_using_c_bound": 1.0 / (eigenvalue - 1.0),
        "retained_c": calibration["ratio_constant"],
        "retained_c_mcse": calibration["ratio_constant_mcse"],
        "calibrated_crossing_n_over_m": center / calibration["ratio_constant"],
        "calibration_scope": config["retained_calibration_scope"],
        "seconds": time.perf_counter() - start,
        "scope": config["scope"],
    }
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["output_sha256"] = {p.name: sha(p) for p in [
        out / "individual_gaussian_runs.npz", out / "summary.csv", out / "summary.json"]}
    (out / "execution_completed.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({name: summary[name] for name in [
        "status", "rows", "calibrated_crossing_n_over_m",
        "tv_sufficient_rho_using_c_bound", "seconds"]}, indent=2))


if __name__ == "__main__":
    main()
