from __future__ import annotations

import statistics
import math
import re
from typing import Any

from bench import Bench, measured, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


def _safe_read_text(root: Any, relative_path: str) -> str | None:
    try:
        return read_text(root, relative_path)
    except (OSError, UnicodeError, TypeError):
        return None


def _safe_read_first(
    root: Any, candidates: tuple[str, ...]
) -> tuple[str, str] | None:
    for relative_path in candidates:
        contents = _safe_read_text(root, relative_path)
        if contents:
            return relative_path, contents
    return None


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    bench.workload.synchronize()
    elapsed = []
    for times in range(repeats):
        start_time = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end_time = bench.clock()
        elapsed.append((end_time - start_time) / 1000000.0) # convert to milliseconds
    return elapsed


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    source = "leading prefix above (1 + 0.5) x median of the run's second half"
    if len(samples) < 4:
        return unknown(source, "there are too few samples")
    middle = len(samples) // 2
    second_half = samples[middle:]
    median = statistics.median(second_half)
    if median <= 0:
        return unknown(source, "the settled median is <= 0")
    threshold = median * (1 + WARMUP_TOL)
    discarded = 0
    for sample in samples:
        if sample <= threshold:
            break
        discarded += 1
    return measured(
        discarded,
        source,
        settled_rate_ms=round(median, 4),
        threshold_ms=round(threshold, 4),
        tolerance=WARMUP_TOL,
        retained=len(samples) - discarded,
    )



def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    samples_sorted = sorted(samples)
    mean = statistics.fmean(samples_sorted)
    minimum = samples_sorted[0]
    n = len(samples_sorted)
    maximum = samples_sorted[-1]
    std = statistics.stdev(samples_sorted) if n > 2 else 0.0

    def percentile(percent: int) -> float:
        position = (n - 1) * percent / 100.0
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return samples_sorted[lower]
        fraction = position - lower
        return samples_sorted[lower] + fraction * (
            samples_sorted[upper] - samples_sorted[lower]
        )

    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(minimum, 4),
        "max": round(maximum, 4),
        "p50": round(percentile(50), 4),
        "p95": round(percentile(95), 4),
        "p99": round(percentile(99), 4),
    }

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    source = (
        "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side"
    )
    if len(samples) < MIN_SAMPLES_FOR_MODALITY:
        return unknown(source, "there are too few samples")
    sorted_samples = sorted(samples)
    length = len(sorted_samples)
    lower_trim = int(length * 0.05)
    upper_trim = int(length * 0.95)
    trimmed = sorted_samples[lower_trim:upper_trim]
    gaps = [trimmed[i+1] - trimmed[i] for i in range(len(trimmed)-1)]
    if not gaps:
        return unknown(source, "there are too few samples after trimming")
    median = statistics.median(gaps)
    if median <= 0:
        return unknown(source, "timer resolution is too coarse")
    widest_gap = max(gaps)
    ratio = widest_gap / median
    gap_index = gaps.index(widest_gap)
    split_index = lower_trim + gap_index + 1
    left = sorted_samples[:split_index]
    right = sorted_samples[split_index:]
    multimodal = (
        ratio >= MULTIMODAL_GAP_RATIO
        and len(left) >= length * MIN_MODE_FRACTION
        and len(right) >= length * MIN_MODE_FRACTION
    )

    def mode_stats(mode: list[float]) -> dict[str, Any]:
        return {
            "n": len(mode),
            "share": round(len(mode) / length, 4),
            "median_ms": round(statistics.median(mode), 4),
        }

    return measured(
        multimodal,
        source,
        gap_ratio=round(ratio, 2),
        widest_gap_ms=round(widest_gap, 5),
        typical_gap_ms=round(median, 5),
        modes=[mode_stats(left), mode_stats(right)],
    )



# ===========================================================================
# 7. The clock ceiling the run happened under
# =========================================================================##


def probe_power_state(bench: Bench) -> dict[str, Any]:
    result = bench.runner(["nvpmodel", "-q"])
    source = "nvpmodel -q"
    if not result.ok or result.returncode != 0:
        detail = result.error or f"command exited with status {result.returncode}"
        return unknown(source, detail)

    lines = result.stdout.splitlines()
    mode_name = None
    mode_index = None
    for line_index, line in enumerate(lines):
        if "NV Power Mode:" not in line:
            continue
        mode_name = line.split("NV Power Mode:", 1)[1].strip()
        if line_index + 1 < len(lines):
            match = re.search(r"-?\d+", lines[line_index + 1])
            if match:
                mode_index = int(match.group())
        break
    if not mode_name or mode_index is None:
        return unknown(source, "could not parse power mode name and ID")

    minimum_text = _safe_read_text(bench.telemetry, CPUFREQ_MIN)
    maximum_text = _safe_read_text(bench.telemetry, CPUFREQ_MAX)
    frequency_source = f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}"
    if minimum_text is None or maximum_text is None:
        clocks_locked = None
        frequency_record = unknown(
            frequency_source, "could not read both CPU frequency limits"
        )
    else:
        try:
            minimum = int(minimum_text)
            maximum = int(maximum_text)
        except ValueError:
            clocks_locked = None
            frequency_record = unknown(
                frequency_source, "CPU frequency limits were not integers"
            )
        else:
            clocks_locked = minimum == maximum
            frequency_record = measured(
                f"scaling_min_freq={minimum}, scaling_max_freq={maximum}",
                frequency_source,
            )

    return measured(
        mode_name,
        source,
        mode_index=mode_index,
        jetson_clocks=clocks_locked,
        jetson_clocks_source=frequency_record,
    )

def probe_telemetry(bench: Bench) -> dict[str, Any]:
    zone_root = bench.telemetry / THERMAL_ZONES
    temperatures = []
    for zone_path in sorted(zone_root.glob("thermal_zone*")):
        relative_temp = str(zone_path.relative_to(bench.telemetry) / "temp")
        raw_temp = _safe_read_text(bench.telemetry, relative_temp)
        if raw_temp is None:
            continue
        try:
            raw_temperature = int(raw_temp)
        except ValueError:
            continue
        if raw_temperature <= -1000:
            continue
        temperature = raw_temperature / 1000.0
        relative_type = str(zone_path.relative_to(bench.telemetry) / "type")
        zone_name = _safe_read_text(bench.telemetry, relative_type) or zone_path.name
        temperatures.append((temperature, zone_name))

    if temperatures:
        hottest, zone_name = max(temperatures, key=lambda item: item[0])
        temperature_record = measured(
            round(hottest, 2),
            f"{THERMAL_ZONES}/*/temp",
            zone=zone_name,
            zones_read=len(temperatures),
        )
    else:
        temperature_record = unknown(
            f"{THERMAL_ZONES}/*/temp", "no valid thermal zone readings were available"
        )

    power_source = " | ".join(POWER_RAIL_CANDIDATES)
    power_reading = _safe_read_first(bench.telemetry, POWER_RAIL_CANDIDATES)
    if power_reading is None:
        power_record = unknown(
            power_source, "none of the documented INA3221 rail paths could be read"
        )
    else:
        path, raw_power = power_reading
        try:
            power_record = measured(int(raw_power), path)
        except ValueError:
            power_record = unknown(path, "power reading was not an integer")

    gpu_source = " | ".join(GPU_LOAD_CANDIDATES)
    gpu_reading = _safe_read_first(bench.telemetry, GPU_LOAD_CANDIDATES)
    if gpu_reading is None:
        gpu_record = unknown(gpu_source, "no GPU load file could be read")
    else:
        path, raw_load = gpu_reading
        try:
            gpu_record = measured(
                int(raw_load) / 10.0, path, units="per-mille / 10"
            )
        except ValueError:
            gpu_record = unknown(path, "GPU load reading was not an integer")

    return {
        "temperature_c": temperature_record,
        "power_mw": power_record,
        "gpu_utilization_percent": gpu_record,
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)