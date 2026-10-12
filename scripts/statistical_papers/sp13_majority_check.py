"""Deterministic and exact-count checks of SP13, without simulating subjects.

Score perturbations test the actual finite knockoff+ rule. Exhaustive sign
counts include zero reference coordinates and a shared-sign negative control.
These computations are not a toroidal FDR study or proof of an asymptotic limit.
"""
from __future__ import annotations

import csv
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/statistical_papers/SP13_PROTOCOLE_FR_20261010.md"
PROOF = ROOT / "docs/statistical_papers/SP13_GROWING_PROOF_EN_20261010.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(w: np.ndarray, q: Fraction = Fraction(1, 10)) -> np.ndarray:
    for t in sorted(set(abs(float(x)) for x in w if x != 0)):
        pos = sum(float(x) >= t for x in w)
        neg = sum(float(x) <= -t for x in w)
        if q.denominator * (1 + neg) <= q.numerator * max(1, pos):
            return w >= t
    return np.zeros(len(w), dtype=bool)


def exact_tail(total: int, nonzero: int, cut: int) -> Fraction:
    # Enumerate every sign assignment, rather than only evaluating a formula.
    hit = 0
    for code in range(1 << nonzero):
        positives = code.bit_count()
        negatives = nonzero - positives
        hit += positives >= cut or negatives >= cut
    return Fraction(hit, 1 << nonzero)


def main() -> None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = ROOT / "results/statistical_papers" / ("sp13_majority_" + stamp + "_" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    started = time.perf_counter()
    record = {"status": "running", "new_independent_datasets": 0,
              "scope": "Deterministic score checks and exhaustive sign counts",
              "source_sha256": sha(Path(__file__)), "protocol_sha256": sha(PROTOCOL),
              "proof_sha256": sha(PROOF)}
    shutil.copyfile(Path(__file__), out / "executed_source.py")
    shutil.copyfile(PROTOCOL, out / "protocol.md")
    shutil.copyfile(PROOF, out / "proof.md")
    sign_rows, filter_rows = [], []
    try:
        for k in (4, 8, 12, 16):
            cut = math.ceil(3 * k / 4)
            envelope = min(Fraction(1), 2 * Fraction(sum(math.comb(k, j) for j in range(cut, k + 1)), 2**k))
            for nz in (k, k - 1, k // 2, 0):
                observed = exact_tail(k, nz, cut)
                if observed > envelope:
                    raise AssertionError("Sign-flip reference exceeds its binomial envelope")
                sign_rows.append({"K": k, "nonzero": nz, "cut": cut,
                    "exact_tail_numerator": observed.numerator, "exact_tail_denominator": observed.denominator,
                    "binomial_envelope": float(envelope), "exact_probability": float(observed),
                    "hoeffding_envelope": min(1., 2 * math.exp(-2 * k * .25**2))})
            if k >= 8:
                # All signs agree under this two-state law: exact probability
                # one violates the independent-flip reference bound.
                if not 1 > float(envelope):
                    raise AssertionError("Shared-sign negative control was not detected")
        for k in (20, 100, 1000):
            for common_sign in (-1, 1):
                for bad in sorted(set((0, 1, k // 20, k // 10))):
                    active = np.ones(k)
                    null = np.full(k, common_sign * .001)
                    null[:bad] = -common_sign * (.01 if common_sign < 0 else .001)
                    w = np.column_stack([active, null]).ravel()
                    chosen = select(w)
                    selected = int(chosen.sum())
                    false = int(chosen[1::2].sum())
                    fdp = Fraction(false, max(1, selected))
                    if not chosen[::2].all():
                        raise AssertionError("A strong active score was excluded")
                    if common_sign > 0:
                        if Fraction(1 + bad, 2*k - bad) > Fraction(1, 10):
                            raise AssertionError("Protocol did not ensure the minimum threshold is admissible")
                        if false != k - bad or fdp != Fraction(k-bad, 2*k-bad):
                            raise AssertionError("Positive-majority threshold calculation")
                    elif false > bad or fdp > Fraction(bad, k+bad):
                        raise AssertionError("Negative-majority FDP bound")
                    errors = null - common_sign * .001
                    mean_square = float(errors @ errors / k)
                    mismatches = float(np.mean(null * common_sign <= 0))
                    if mismatches > mean_square / .001**2 + 1e-12:
                        raise AssertionError("Mean-square sign-mismatch bound")
                    np.savez_compressed(out / f"score_K{k}_s{common_sign}_bad{bad}.npz",
                                        scores=w, selected=chosen)
                    filter_rows.append({"K": k, "common_sign": common_sign, "bad": bad,
                        "selected": selected, "false": false, "FDP": float(fdp),
                        "mean_square_score_error": mean_square,
                        "fraction_wrong_sign": mismatches,
                        "all_null_signs_agree": bad == 0})
        floor = np.column_stack([np.ones(5), np.full(5, -.001)]).ravel()
        if select(floor).any():
            raise AssertionError("Small-K discovery floor control failed")
        for name, rows in (("reference_sign_counts.csv", sign_rows), ("deterministic_filters.csv", filter_rows)):
            with (out / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        record.update(status="complete", exact_reference_cases=len(sign_rows),
                      deterministic_filter_cases=len(filter_rows),
                      negative_controls_passed=4,
                      files_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    except BaseException as error:
        record.update(status="incomplete", error=repr(error))
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        (out / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
