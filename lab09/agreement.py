
from __future__ import annotations

from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    try:
        f = float(v)
    except OverflowError as exc:
        raise AgreementError(
            f"{where} is outside the finite floating-point range."
        ) from exc
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    samples = []
    for x in ms:
        sample = _finite(x, "latency sample")
        if sample < 0:
            raise AgreementError("latency samples must be non-negative.")
        samples.append(sample)
    if not samples:
        raise AgreementError("no latency samples.")
    n = len(samples)
    mean = _finite(sum(samples) / n, "latency mean")
    worst_case = max(samples)
    p50, p95, p99 = (
        _percentile(samples, percentile)
        for percentile in LATENCY_PERCENTILES
    )
    jitter = _finite(p95 - p50, "latency jitter")
    return {
        "n": n,
        "mean": mean,
        "worst": worst_case,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "jitter": jitter,
    }


def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    base_ms = _finite(base_ms, "base latency")
    other_ms = _finite(other_ms, "other latency")
    if base_ms <= 0:
        raise AgreementError("base latency must be positive.")
    if other_ms <= 0:
        raise AgreementError("other latency must be positive.")
    factor = _finite(base_ms / other_ms, "speedup factor")
    faster = factor > 1.0
    saved_ms = _finite(base_ms - other_ms, "saved latency")
    return {
        "factor": factor,
        "faster": faster,
        "saved_ms": saved_ms,
    }


def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    if len(a) != len(b):
        raise AgreementError("a and b must have the same length.")
    n = len(a)
    if n == 0:
        raise AgreementError("a and b must not be empty.")
    values_a = [_finite(value, f"a[{i}]") for i, value in enumerate(a)]
    values_b = [_finite(value, f"b[{i}]") for i, value in enumerate(b)]
    best_i = 0
    best = _finite(abs(values_a[0] - values_b[0]), "absolute deviation")
    for i in range(1, n):
        dev = _finite(abs(values_a[i] - values_b[i]), f"absolute deviation at {i}")
        if dev > best:
            best = dev
            best_i = i
    return {
        "max_abs": best,
        "index": best_i,
        "a": values_a[best_i],
        "b": values_b[best_i],
        "n": n,
    }




def max_rel_dev(a: Sequence[float], b: Sequence[float],
                floor: float = REL_FLOOR) -> dict[str, Any]:
    floor = _finite(floor, "floor")
    if len(a) != len(b):
        raise AgreementError("a and b must have the same length.")
    if floor <= 0:
        raise AgreementError("floor must be positive.")
    n = len(a)
    if n == 0:
        raise AgreementError("a and b must not be empty.")
    values_a = [_finite(value, f"a[{i}]") for i, value in enumerate(a)]
    values_b = [_finite(value, f"b[{i}]") for i, value in enumerate(b)]
    floored = 0
    denom_min = min(abs(value) for value in values_b)
    best = -1.0
    best_i = 0
    for i in range(n):
        d = abs(values_b[i])
        if d < floor:
            floored += 1
        r = _finite(
            abs(values_a[i] - values_b[i]) / max(d, floor),
            f"relative deviation at {i}",
        )
        if r > best:
            best = r
            best_i = i
    return {
        "max_rel": best,
        "index": best_i,
        "floored": floored,
        "denom_min": denom_min,
        "n": n,
    }


def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not rows_a or not rows_b:
        raise AgreementError("rows_a and rows_b must not be empty.")
    if len(rows_a) != len(rows_b):
        raise AgreementError("rows_a and rows_b must have the same length.")
    N = len(rows_a)
    agree = 0
    first_disagreement = None
    for k in range(N):
        row_a = rows_a[k]
        row_b = rows_b[k]
        if not row_a or not row_b:
            raise AgreementError(f"row {k} must not be empty.")
        if len(row_a) != len(row_b):
            raise AgreementError(f"row {k} has different lengths in rows_a and rows_b.")
        values_a = [_finite(value, f"rows_a[{k}][{i}]") for i, value in enumerate(row_a)]
        values_b = [_finite(value, f"rows_b[{k}][{i}]") for i, value in enumerate(row_b)]
        ia = max(range(len(values_a)), key=lambda i: values_a[i])
        ib = max(range(len(values_b)), key=lambda i: values_b[i])
        if ia == ib:
            agree += 1
        elif first_disagreement is None:
            first_disagreement = k
    disagree = N - agree
    rate = agree / N
    return {
        "n": N,
        "agree": agree,
        "disagree": disagree,
        "rate": rate,
        "first_disagreement": first_disagreement,
    }



def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:
    if not rows:
        raise AgreementError("rows must not be empty.")
    tol = _finite(tol, "tol")
    if tol < 0:
        raise AgreementError("tol must be non-negative.")
    N = len(rows)
    margins = []
    for k, row in enumerate(rows):
        if len(row) < 2:
            raise AgreementError(f"row {k} has fewer than 2 classes.")
        scores = sorted(
            (_finite(value, f"rows[{k}][{i}]") for i, value in enumerate(row)),
            reverse=True,
        )
        margins.append(_finite(scores[0] - scores[1], f"margin at row {k}"))
    min_margin = min(margins)
    argmin = margins.index(min_margin)
    median = _percentile(margins, 50)
    mean = _finite(sum(margins) / N, "mean decision margin")
    below_tol = sum(1 for margin in margins if margin < tol)
    safe = (below_tol == 0)
    return {
        "n": N,
        "min": min_margin,
        "argmin": argmin,
        "median": median,
        "mean": mean,
        "below_tol": below_tol,
        "safe": safe,
    }

#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()