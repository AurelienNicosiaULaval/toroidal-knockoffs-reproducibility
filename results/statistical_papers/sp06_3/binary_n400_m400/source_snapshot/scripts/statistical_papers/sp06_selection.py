#!/usr/bin/env python3
"""Frozen SP06 selection study with individually resumable exact-iid repetitions."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import scipy
from scipy.special import expit
from threadpoolctl import threadpool_limits

from provenance import atomic_json, digest, stream
from sp06.star import canonical_statistics, exact_star, pilot_fit, true_star
from sp06.wald import by_select, calibrated_blocks
from toroidal import EdgeSampler, knockstat, select, vec

SOURCES = ["src/sp06/__init__.py", "src/sp06/wald.py", "src/sp06/star.py", "src/fit_model.py",
           "src/toroidal.py", "src/provenance.py", "src/v3/fit_verified.py", "src/v3/population_pair.py",
           "tests/test_sp06_star.py", "tests/test_sp06_wald.py", "scripts/statistical_papers/sp06_selection.py",
           "configs/statistical_papers/sp06_1_selection.json", "docs/statistical_papers/SP06_1_PROTOCOLE_FR_20261010.md"]


def serial(item):
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, np.generic):
        return item.item()
    if isinstance(item, dict):
        return {key: serial(value) for key, value in item.items()}
    if isinstance(item, (tuple, list)):
        return [serial(value) for value in item]
    return item


def prepare(config: dict) -> Path:
    manifest = {"config": config, "code_sha256": {p: digest(ROOT / p) for p in SOURCES},
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "platform": platform.platform(), "blas_threads": 1}}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    output = ROOT / "results/statistical_papers" / config["version"] / config["id"]
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text())["signature"] != signature:
            raise RuntimeError("Code/configuration/environment changed: create a new experiment version")
    else:
        manifest.update(signature=signature, created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(path, manifest)
        for relative in SOURCES:
            destination = output / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())
    return output


def repetition(config: dict, index: int) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    p, n, m = config["p"], config["n"], config["m"]
    truth = true_star(config)
    active = [] if config["response"] == "null" else config["active"]
    null = np.setdiff1d(np.arange(p), active)
    k = 2 * p + 4 * len(truth.edges)
    arrays = {"pvalues": np.full((3, p), np.nan), "knockoff_scores": np.full((2, p), np.nan),
              "contrasts": np.full((2, p, 2), np.nan), "oracle_variance_components": np.full((2, p, 2, 2), np.nan),
              "pilot_derivative": np.full((p, 2, k), np.nan), "total_variance": np.full((p, 2, 2), np.nan),
              "pilot_parameters": np.full(k, np.nan), "pilot_covariance_upper": np.full(k * (k + 1) // 2, np.nan),
              "conditional_field_mse": np.full(p, np.nan), "selected_mask": np.zeros((5, p), dtype=bool)}
    failures = np.ones(5, dtype=bool)
    selections = [np.empty(0, dtype=int) for _ in range(5)]
    errors = []
    times = {}
    record = {"replicate": index, "config_id": config["id"], "seed": config["seed"],
              "active": active, "raw_subjects_retained": False,
              "raw_data_reconstruction": "Frozen code and per-repetition role streams; no MCMC"}
    try:
        phase = time.perf_counter()
        x, original_sampler = exact_star(stream(config, index, "original"), truth, n)
        times["original_sampling"] = time.perf_counter() - phase
        phase = time.perf_counter()
        pilot, pilot_sampler = exact_star(stream(config, index, "pilot"), truth, m)
        times["pilot_sampling"] = time.perf_counter() - phase
        record.update(original_sampler=original_sampler, pilot_sampler=pilot_sampler)
        rng_y = stream(config, index, "response")
        if config["response"] == "null":
            y = rng_y.normal(scale=config["noise_sd"], size=n)
        else:
            nonlinear = np.cos(2 * (x[:, active] - config["field_phase"])).sum(axis=1) / np.sqrt(len(active))
            if config["response"] == "binary":
                probability = expit(config["binary_scale"] * nonlinear + config["binary_intercept"])
                y = rng_y.binomial(1, probability)
                record["binary_response_fraction"] = float(y.mean())
            else:
                y = config["continuous_scale"] * nonlinear + rng_y.normal(scale=config["noise_sd"], size=n)
        t = canonical_statistics(x, truth.edges)
        original = vec(x).reshape(n, 2 * p)

        phase = time.perf_counter()
        oracle = EdgeSampler(truth).sample(stream(config, index, "oracle_witness"), x)
        times["oracle_witness"] = time.perf_counter() - phase
        phase = time.perf_counter()
        try:
            result = calibrated_blocks(original, vec(oracle).reshape(n, 2 * p), y, t, np.zeros((k, k)), m,
                                       level=config["level"], ridge_penalty=config["ridge_penalty"])
            arrays["pvalues"][2] = result.pvalues
            arrays["contrasts"][1] = result.contrasts
            arrays["oracle_variance_components"][1] = result.oracle_variance_component
            selections[2] = result.selected
            failures[2] = False
        except Exception as error:
            errors.append({"stage": "wald_oracle", "error": repr(error)})
        try:
            w = knockstat(x, oracle, y, penalty=config["ridge_penalty"])
            arrays["knockoff_scores"][1] = w
            selections[4] = select(w, config["level"])
            failures[4] = False
        except Exception as error:
            errors.append({"stage": "knockoff_oracle", "error": repr(error)})
        times["oracle_statistics"] = time.perf_counter() - phase

        try:
            fitted, covariance, diagnostic = pilot_fit(pilot, truth.edges, config["pilot_penalty_numerator"])
            if diagnostic["gradient_l2"] > config["pilot_gradient_tolerance"]:
                raise ArithmeticError("Pilot stationarity did not meet frozen tolerance")
            record["pilot_diagnostics"] = diagnostic
            times["pilot_estimation"] = diagnostic["seconds"]
            times["pilot_sandwich"] = diagnostic["covariance_seconds"]
            arrays["pilot_parameters"] = np.r_[fitted.h.ravel(), fitted.M.ravel()]
            arrays["pilot_covariance_upper"] = covariance[np.triu_indices(k)]
            arrays["conditional_field_mse"] = ((truth.fields(x) - fitted.fields(x)) ** 2).sum(axis=2).mean(axis=0)
            phase = time.perf_counter()
            witness = EdgeSampler(fitted).sample(stream(config, index, "estimated_witness"), x)
            times["estimated_witness"] = time.perf_counter() - phase
            phase = time.perf_counter()
            try:
                result = calibrated_blocks(original, vec(witness).reshape(n, 2 * p), y, t, covariance, m,
                                           level=config["level"], ridge_penalty=config["ridge_penalty"])
                arrays["pvalues"][0] = result.pvalues
                arrays["pvalues"][1] = result.naive_pvalues
                arrays["contrasts"][0] = result.contrasts
                arrays["oracle_variance_components"][0] = result.oracle_variance_component
                arrays["pilot_derivative"] = result.pilot_derivative
                arrays["total_variance"] = result.total_variance
                selections[0], selections[1] = result.selected, result.naive_selected
                failures[:2] = False
            except Exception as error:
                errors.append({"stage": "wald_fitted", "error": repr(error)})
            try:
                w = knockstat(x, witness, y, penalty=config["ridge_penalty"])
                arrays["knockoff_scores"][0] = w
                selections[3] = select(w, config["level"])
                failures[3] = False
            except Exception as error:
                errors.append({"stage": "knockoff_fitted", "error": repr(error)})
            times["estimated_statistics"] = time.perf_counter() - phase
        except Exception as error:
            errors.append({"stage": "pilot_fit_or_witness", "error": repr(error)})
    except Exception as error:
        errors.append({"stage": "simulation", "error": repr(error), "traceback": traceback.format_exc()})

    methods = []
    for j, method in enumerate(config["method_order"]):
        chosen = selections[j]
        arrays["selected_mask"][j, chosen] = True
        false_count = np.isin(chosen, null).sum()
        true_count = np.isin(chosen, active).sum()
        item = {"method": method, "selected": chosen.tolist(), "fdp": float(false_count / max(1, len(chosen))),
                "power": float(true_count / max(1, len(active))), "empty": not len(chosen),
                "failed": bool(failures[j]), "failure_policy": "abstain, counted unconditionally"}
        if j < 3:
            item["marginal_null_rejection"] = 0. if failures[j] else float((arrays["pvalues"][j, null] <= config["marginal_level"]).mean())
        methods.append(item)
    record.update(methods=methods, errors=errors, times_seconds=times,
                  elapsed_seconds=time.perf_counter() - started,
                  max_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return serial(record), arrays


def atomic_arrays(path: Path, values: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def read_record(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    record = json.loads(path.read_text())
    data = path.with_suffix(".npz")
    if digest(data) != record["arrays_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {data}")
    with np.load(data, allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
    return record, arrays


def moments(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def aggregate(config: dict, output: Path) -> dict:
    records = []
    for index in range(config["repetitions"]):
        record, arrays = read_record(output / f"rep_{index:04d}.json")
        if record["replicate"] != index:
            raise RuntimeError("Replicate index mismatch")
        for j, method in enumerate(record["methods"]):
            if not method["failed"]:
                if j < 3:
                    chosen = by_select(arrays["pvalues"][j], config["level"])
                else:
                    chosen = select(arrays["knockoff_scores"][j - 3], config["level"])
                if chosen.tolist() != method["selected"]:
                    raise RuntimeError("Recomputed selection disagrees with individual record")
        records.append(record)
    rows, paired = [], []
    matrix = {}
    for index, name in enumerate(config["method_order"]):
        row = {"configuration": config["id"], "n": config["n"], "m": config["m"], "method": name,
               "repetitions": len(records), "central": config["central"]}
        failed = np.array([r["methods"][index]["failed"] for r in records])
        for metric in ["fdp", "power", "empty"] + (["marginal_null_rejection"] if index < 3 else []):
            values = np.array([r["methods"][index][metric] for r in records], float)
            average, se = moments(values)
            row[metric], row[metric + "_mcse"] = average, se
            row[metric + "_failure_lower"] = float((values * (~failed)).sum() / len(values))
            row[metric + "_failure_upper"] = float(((values * (~failed)).sum() + failed.sum()) / len(values))
            matrix[name, metric] = values
        row["failures"] = int(failed.sum())
        row["failure_rate"] = float(failed.mean())
        if index == 0:
            row["fdr_diagnostic"] = row["fdp"] <= config["level"] + config["gate_mcse_multiplier"] * row["fdp_mcse"]
            row["marginal_diagnostic"] = row["marginal_null_rejection"] <= config["marginal_level"] + config["gate_mcse_multiplier"] * row["marginal_null_rejection_mcse"]
            row["calibration_diagnostic_complete"] = row["failures"] == 0
        rows.append(row)
    for left, right in itertools.combinations(config["method_order"], 2):
        for metric in ["fdp", "power", "empty"]:
            diff, se = moments(matrix[left, metric] - matrix[right, metric])
            paired.append({"configuration": config["id"], "left": left, "right": right,
                           "metric": metric, "difference_left_minus_right": diff, "mcse": se})
    timing = {}
    keys = set().union(*(r["times_seconds"] for r in records))
    for key in sorted(keys):
        values = np.array([r["times_seconds"].get(key, 0.) for r in records])
        timing[key] = {"total_seconds": float(values.sum()), "mean_seconds_per_attempt": float(values.mean()),
                       "median_seconds_per_attempt": float(np.median(values))}
    result = serial({"config": config, "summary": rows, "paired": paired, "timing": timing,
                     "failed_repetitions": [r["replicate"] for r in records if r["errors"]],
                     "max_rss_bytes_macos": max(r["max_rss_bytes_macos"] for r in records),
                     "individual_recomputed_selections": len(records) * 5,
                     "scope": config["scope"]})
    path = output / "summary.json"
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise RuntimeError("Reaggregation differs; existing summary preserved")
    else:
        atomic_json(path, result)
    return result


def equivalent(record: dict) -> dict:
    trimmed = dict(record)
    for key in ["elapsed_seconds", "max_rss_bytes_macos", "times_seconds", "arrays_sha256"]:
        trimmed.pop(key, None)
    if "pilot_diagnostics" in trimmed:
        trimmed["pilot_diagnostics"] = {k: v for k, v in trimmed["pilot_diagnostics"].items()
                                        if k not in ["seconds", "covariance_seconds"]}
    return trimmed


def run(config_path: Path, aggregate_only: bool, replay: bool) -> None:
    base = json.loads(config_path.read_text())
    started = time.perf_counter()
    invocation = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    log = ROOT / "results/statistical_papers" / base["version"] / f"invocation_{invocation}.json"
    completed = []
    try:
        with threadpool_limits(limits=1):
            for case_index, case in enumerate(base["cases"]):
                config = dict(base, **case)
                config["seed"] = base["seed"] + 100000 * case_index
                output = prepare(config)
                case_started, new = time.perf_counter(), 0
                for index in range(config["repetitions"]):
                    path = output / f"rep_{index:04d}.json"
                    if path.exists():
                        # Aggregation verifies all NPZ hashes before reporting completion.
                        continue
                    if aggregate_only:
                        raise FileNotFoundError(f"Missing individual repetition {path}")
                    data_path = path.with_suffix(".npz")
                    if data_path.exists():
                        raise RuntimeError(f"Orphaned atomic data file; inspect before resuming: {data_path}")
                    record, arrays = repetition(config, index)
                    atomic_arrays(data_path, arrays)
                    record["arrays_sha256"] = digest(data_path)
                    atomic_json(path, record)
                    new += 1
                    if (index + 1) % 100 == 0:
                        print(config["id"], index + 1, "saved", flush=True)
                        atomic_json(log, {"status": "running", "completed": completed, "current_case": config["id"],
                                         "saved_index": index, "elapsed_seconds": time.perf_counter() - started})
                result = aggregate(config, output)
                replays = []
                if replay:
                    for index in [0, 499, 999]:
                        recorded, retained = read_record(output / f"rep_{index:04d}.json")
                        generated, fresh = repetition(config, index)
                        if equivalent(recorded) != equivalent(generated):
                            raise RuntimeError(f"JSON replay mismatch at {config['id']}:{index}")
                        for name in fresh:
                            np.testing.assert_array_equal(fresh[name], retained[name], err_msg=f"{config['id']}:{index}:{name}")
                        replays.append(index)
                item = {"id": config["id"], "new_repetitions": new, "recorded_repetitions": config["repetitions"],
                        "replayed_indices": replays, "elapsed_seconds_including_aggregation_and_replay": time.perf_counter() - case_started,
                        "summary_sha256": digest(output / "summary.json"), "failed_repetitions": result["failed_repetitions"]}
                completed.append(item)
                print(json.dumps(item), flush=True)
        atomic_json(log, {"status": "complete", "completed": completed, "elapsed_seconds": time.perf_counter() - started,
                         "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "scope": base["scope"]})
    except BaseException as error:
        atomic_json(log, {"status": "incomplete", "completed": completed,
                         "elapsed_seconds": time.perf_counter() - started, "error": repr(error), "traceback": traceback.format_exc()})
        raise
    print(log, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/statistical_papers/sp06_1_selection.json")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    run(args.config.resolve(), args.aggregate_only, args.replay)
