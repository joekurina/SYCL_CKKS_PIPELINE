#!/usr/bin/env python3
"""Validate raw FPT 2026 records and generate deterministic paper summaries."""

from __future__ import annotations

import argparse
import csv
import html
import io
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from common import (
    ContractError,
    canonical_json_bytes,
    load_json,
    read_jsonl,
    require_absolute,
    require_unique,
    sha256_bytes,
    sha256_file,
    write_new_bytes,
    write_new_json,
)
from plan_matrix import count_units

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
MINIMUM_CI_OBSERVATIONS = 2
ALGORITHM_VERSION = "fpt2026-summary-v1"
BOOTSTRAP_ALGORITHM = "percentile-bootstrap-v1"
PERCENTILE_ALGORITHM = "linear-interpolation-r7"
TIMING_FIELDS = (
    "application_e2e",
    "cold_first_result",
    "program_time",
    "preparation",
    "pack",
    "h2d_wall",
    "h2d_device",
    "graph_device",
    "d2h_device",
    "d2h_wall",
    "graph_submit_wait_wall",
    "unpack_and_assembly",
    "unattributed_wall",
)
GROUP_FIELDS = (
    "experiment_id",
    "backend",
    "boundary_id",
    "phase",
    "case_id",
    "batch_size",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--analysis-seed",
        type=int,
        help="nonnegative stored analysis seed; default derives from the immutable manifest hash",
    )
    return parser.parse_args()


def percentile(values: Sequence[float], probability: float) -> float:
    """R-7 linear-interpolation percentile used by all summaries and CIs."""
    if not values:
        raise ContractError("percentile requires at least one observation")
    if not 0.0 <= probability <= 1.0:
        raise ContractError(f"percentile probability is outside [0,1]: {probability}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def median(values: Sequence[float]) -> float:
    return percentile(values, 0.5)


def descriptive_statistics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ContractError("descriptive statistics require at least one observation")
    materialized = [float(value) for value in values]
    center = median(materialized)
    p25 = percentile(materialized, 0.25)
    p75 = percentile(materialized, 0.75)
    return {
        "min": min(materialized),
        "p5": percentile(materialized, 0.05),
        "p25": p25,
        "median": center,
        "p75": p75,
        "p95": percentile(materialized, 0.95),
        "max": max(materialized),
        "mad": median([abs(value - center) for value in materialized]),
        "iqr": p75 - p25,
    }


def _derived_seed(analysis_seed: int, label: str) -> int:
    payload = f"{analysis_seed}\0{label}".encode("utf-8")
    return int.from_bytes(bytes.fromhex(sha256_bytes(payload))[:8], "big")


def percentile_bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    analysis_seed: int,
    label: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    materialized = [float(value) for value in values]
    if len(materialized) < MINIMUM_CI_OBSERVATIONS:
        return {
            "lower": None,
            "upper": None,
            "confidence": confidence,
            "resamples": resamples,
            "algorithm": BOOTSTRAP_ALGORITHM,
            "analysis_seed": analysis_seed,
            "group_seed": None,
            "observation_count": len(materialized),
            "unavailable_reason": f"requires_at_least_{MINIMUM_CI_OBSERVATIONS}_observations",
        }
    if resamples != BOOTSTRAP_RESAMPLES:
        raise ContractError(f"official bootstrap requires exactly {BOOTSTRAP_RESAMPLES} resamples")
    if confidence != BOOTSTRAP_CONFIDENCE:
        raise ContractError(f"official bootstrap confidence must be {BOOTSTRAP_CONFIDENCE}")
    group_seed = _derived_seed(analysis_seed, label)
    generator = random.Random(group_seed)
    size = len(materialized)
    estimates = [
        float(statistic([materialized[generator.randrange(size)] for _ in range(size)]))
        for _ in range(resamples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "lower": percentile(estimates, alpha),
        "upper": percentile(estimates, 1.0 - alpha),
        "confidence": confidence,
        "resamples": resamples,
        "algorithm": BOOTSTRAP_ALGORITHM,
        "analysis_seed": analysis_seed,
        "group_seed": group_seed,
        "observation_count": size,
        "unavailable_reason": None,
    }


def outlier_counts(values: Sequence[float]) -> dict[str, int]:
    if not values:
        return {"tukey_fence": 0, "mad": 0}
    materialized = [float(value) for value in values]
    stats = descriptive_statistics(materialized)
    lower = stats["p25"] - 1.5 * stats["iqr"]
    upper = stats["p75"] + 1.5 * stats["iqr"]
    tukey = sum(value < lower or value > upper for value in materialized)
    mad = stats["mad"]
    mad_count = 0
    if mad > 0:
        center = stats["median"]
        mad_count = sum(0.6744897501960817 * abs(value - center) / mad > 3.5 for value in materialized)
    return {"tukey_fence": tukey, "mad": mad_count}


def _valid_sample(row: dict[str, Any]) -> bool:
    summary = row.get("correctness_summary")
    return (
        row.get("status") == "pass"
        and isinstance(summary, dict)
        and summary.get("verified_after_timing") is True
        and row.get("frame_count_submitted") == row.get("frame_count_completed")
    )


def _sample_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in GROUP_FIELDS)


def _group_dict(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(GROUP_FIELDS, key))


def _reliability(rows: Sequence[dict[str, Any]], finished_by_attempt: dict[str, dict[str, Any]]) -> dict[str, int]:
    attempt_ids = {row["attempt_id"] for row in rows}
    terminals = [finished_by_attempt[attempt_id] for attempt_id in attempt_ids if attempt_id in finished_by_attempt]
    return {
        "submitted": len(rows),
        "completed": sum(row.get("frame_count_completed") == row.get("frame_count_submitted") for row in rows),
        "successful": sum(_valid_sample(row) for row in rows),
        "failed": sum(row.get("status") == "fail" for row in rows),
        "timed_out": sum(bool(row.get("timed_out")) for row in terminals),
        "invalid_control": sum(row.get("status") == "invalid_control" for row in rows),
    }


def latency_summary_rows(
    samples: Sequence[dict[str, Any]],
    finished_by_attempt: dict[str, dict[str, Any]],
    analysis_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        grouped[_sample_group_key(row)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(part) for part in value)):
        rows = grouped[key]
        valid = [row for row in rows if _valid_sample(row)]
        reliability = _reliability(rows, finished_by_attempt)
        for timing_field in TIMING_FIELDS:
            values = [float(row["timing_ns"][timing_field]) for row in valid if row["timing_ns"].get(timing_field) is not None]
            if not values:
                continue
            identity = _group_dict(key)
            label = "latency:" + ":".join(str(identity[field]) for field in GROUP_FIELDS) + f":{timing_field}"
            stats = descriptive_statistics(values)
            ci = percentile_bootstrap_ci(
                values, median, analysis_seed=analysis_seed, label=label
            )
            output.append(
                {
                    **identity,
                    "metric": timing_field,
                    "raw_unit": "ns",
                    "display_unit": "ms",
                    "observation_count": len(values),
                    **reliability,
                    **stats,
                    "ci95_lower": ci["lower"],
                    "ci95_upper": ci["upper"],
                    "ci95_unavailable_reason": ci["unavailable_reason"],
                    "tukey_outlier_count": outlier_counts(values)["tukey_fence"],
                    "mad_outlier_count": outlier_counts(values)["mad"],
                }
            )
    return output


def _throughput(sample: dict[str, Any], timing_field: str) -> float | None:
    duration = sample["timing_ns"].get(timing_field)
    if duration is None or duration <= 0:
        return None
    return float(sample["frame_count_completed"]) * 1_000_000_000.0 / float(duration)


def throughput_summary_rows(
    samples: Sequence[dict[str, Any]],
    finished_by_attempt: dict[str, dict[str, Any]],
    analysis_seed: int,
) -> list[dict[str, Any]]:
    e8 = [row for row in samples if row.get("experiment_id") == "E8"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in e8:
        grouped[_sample_group_key(row)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: int(value[-1])):
        rows = grouped[key]
        valid = [row for row in rows if _valid_sample(row)]
        for timing_field, metric in (
            ("application_e2e", "application_e2e_frames_per_second"),
            ("graph_device", "graph_frames_per_second"),
        ):
            values = [value for row in valid if (value := _throughput(row, timing_field)) is not None]
            if not values:
                continue
            identity = _group_dict(key)
            label = f"throughput:{identity['phase']}:{identity['batch_size']}:{metric}"
            stats = descriptive_statistics(values)
            ci = percentile_bootstrap_ci(
                values, median, analysis_seed=analysis_seed, label=label
            )
            output.append(
                {
                    **identity,
                    "metric": metric,
                    "raw_unit": "frames_per_second",
                    "display_unit": "frames_per_second",
                    "observation_count": len(values),
                    **_reliability(rows, finished_by_attempt),
                    **stats,
                    "ci95_lower": ci["lower"],
                    "ci95_upper": ci["upper"],
                    "ci95_unavailable_reason": ci["unavailable_reason"],
                    "tukey_outlier_count": outlier_counts(values)["tukey_fence"],
                    "mad_outlier_count": outlier_counts(values)["mad"],
                }
            )
    return output


def accuracy_summary_rows(correctness: Sequence[dict[str, Any]], analysis_seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in correctness:
        if row.get("verification_kind") in {"decrypt_decode", "paired_reference"}:
            grouped[(row["experiment_id"], row["backend"], row["case_id"], row["verification_kind"])].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        for field in ("max_abs_error", "rms_error", "component_max_abs_error"):
            values = [float(row[field]) for row in rows if row.get("passed") is True]
            if not values:
                continue
            label = "accuracy:" + ":".join(key) + f":{field}"
            stats = descriptive_statistics(values)
            ci = percentile_bootstrap_ci(
                values, median, analysis_seed=analysis_seed, label=label
            )
            output.append(
                {
                    "experiment_id": key[0],
                    "backend": key[1],
                    "case_id": key[2],
                    "verification_kind": key[3],
                    "metric": field,
                    "raw_unit": "absolute_error",
                    "display_unit": "absolute_error",
                    "submitted": len(rows),
                    "completed": len(rows),
                    "successful": len(values),
                    "failed": sum(row.get("passed") is not True for row in rows),
                    "timed_out": 0,
                    "invalid_control": 0,
                    "observation_count": len(values),
                    **stats,
                    "ci95_lower": ci["lower"],
                    "ci95_upper": ci["upper"],
                    "ci95_unavailable_reason": ci["unavailable_reason"],
                    "tukey_outlier_count": outlier_counts(values)["tukey_fence"],
                    "mad_outlier_count": outlier_counts(values)["mad"],
                }
            )
    return output


def saturation_batch(points: Sequence[dict[str, Any]]) -> int | None:
    ordered = sorted(points, key=lambda point: int(point["batch_size"]))
    if len(ordered) < 3:
        return None
    medians = [float(point["median_e2e_frames_per_second"]) for point in ordered]
    threshold = 0.95 * max(medians)
    for index in range(len(ordered) - 2):
        if all(value >= threshold for value in medians[index:index + 3]):
            return int(ordered[index]["batch_size"])
    return None


def e8_evidence_document(
    samples: Sequence[dict[str, Any]], *, required_batches: Sequence[int]
) -> dict[str, Any]:
    required = list(required_batches)
    if required != sorted(set(required)) or any(batch < 1 for batch in required):
        raise ContractError("required E8 batches must be positive, unique, and sorted")
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        if row.get("experiment_id") == "E8" and row.get("backend") == "fpga":
            batch = row.get("batch_size")
            if isinstance(batch, int) and batch in required:
                by_batch[batch].append(row)
    points: list[dict[str, Any]] = []
    for batch in required:
        rows = by_batch.get(batch, [])
        if len(rows) != 20:
            raise ContractError(f"E8 batch {batch} requires exactly 20 measured observations, observed {len(rows)}")
        raw_sample_ids = [row.get("sample_id") for row in rows]
        if any(not isinstance(identifier, str) or not identifier for identifier in raw_sample_ids) or len(set(raw_sample_ids)) != 20:
            raise ContractError(f"E8 batch {batch} sample IDs are missing or duplicated")
        sample_ids = [str(identifier) for identifier in raw_sample_ids]
        if any(row.get("frame_count_submitted") != batch for row in rows):
            raise ContractError(f"E8 batch {batch} contains a mismatched submitted-frame count")
        correctness_qualified = all(_valid_sample(row) for row in rows)
        liveness_qualified = all(row.get("frame_count_completed") == batch for row in rows)
        if not correctness_qualified or not liveness_qualified:
            raise ContractError(f"E8 batch {batch} is not correctness- and liveness-qualified")
        e2e_values = [_throughput(row, "application_e2e") for row in rows]
        if any(value is None for value in e2e_values):
            raise ContractError(f"E8 batch {batch} has a zero/missing application interval")
        graph_values = [_throughput(row, "graph_device") for row in rows]
        points.append(
            {
                "batch_size": batch,
                "observation_count": 20,
                "median_e2e_frames_per_second": median([float(value) for value in e2e_values if value is not None]),
                "median_graph_frames_per_second": (
                    median([float(value) for value in graph_values if value is not None])
                    if all(value is not None for value in graph_values)
                    else None
                ),
                "correctness_qualified": correctness_qualified,
                "liveness_qualified": liveness_qualified,
                "sample_ids": sorted(sample_ids),
            }
        )
    candidate = saturation_batch(points)
    return {
        "schema_version": "1.0",
        "document_type": "e8_throughput_evidence",
        "saturation_rule": (
            "smallest batch at least 95 percent of maximum median whose next two measured "
            "larger batches are also at least 95 percent of that maximum"
        ),
        "throughput_points": points,
        "maximum_median_e2e_frames_per_second": max(
            float(point["median_e2e_frames_per_second"]) for point in points
        ),
        "saturation_batch": candidate,
        "saturation_status": "established" if candidate is not None else "saturation_not_reached",
    }


def speedup_summary(samples: Sequence[dict[str, Any]], analysis_seed: int) -> dict[str, Any]:
    rows = [
        row for row in samples
        if row.get("experiment_id") == "E6"
        and row.get("case_id") == "real_full_4096"
        and row.get("boundary_id") == "application_e2e_v1"
        and _valid_sample(row)
    ]
    fpga = [float(row["timing_ns"]["application_e2e"]) for row in rows if row.get("backend") == "fpga"]
    cpu = [float(row["timing_ns"]["application_e2e"]) for row in rows if row.get("backend") == "seal-embedded"]
    if len(fpga) != 50 or len(cpu) != 50:
        raise ContractError(f"E6 speedup requires 50 FPGA and 50 SEAL-Embedded values, observed {len(fpga)} and {len(cpu)}")
    if any(value <= 0 for value in fpga + cpu):
        raise ContractError("E6 application_e2e values must be positive")
    ratio = median(cpu) / median(fpga)
    group_seed = _derived_seed(analysis_seed, "E6:ratio-of-medians:application_e2e_v1")
    generator = random.Random(group_seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        resampled_cpu = [cpu[generator.randrange(len(cpu))] for _ in cpu]
        resampled_fpga = [fpga[generator.randrange(len(fpga))] for _ in fpga]
        estimates.append(median(resampled_cpu) / median(resampled_fpga))
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return {
        "boundary_id": "application_e2e_v1",
        "case_id": "real_full_4096",
        "numerator_backend": "seal-embedded",
        "denominator_backend": "fpga",
        "cpu_observation_count": len(cpu),
        "fpga_observation_count": len(fpga),
        "ratio_of_medians": ratio,
        "ci95_lower": percentile(estimates, alpha),
        "ci95_upper": percentile(estimates, 1.0 - alpha),
        "confidence": BOOTSTRAP_CONFIDENCE,
        "resamples": BOOTSTRAP_RESAMPLES,
        "algorithm": BOOTSTRAP_ALGORITHM,
        "analysis_seed": analysis_seed,
        "group_seed": group_seed,
    }


def _load_planned_units(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    paths = manifest["paths"]
    phase_specs = (
        ("planned_units_base", "effective_plan_base"),
        ("planned_units_e8_extension", "effective_plan_e8"),
        ("planned_units_e4_sustained", "effective_plan_e4"),
    )
    all_units: list[dict[str, Any]] = []
    authorizing_hash: dict[str, str] = {}
    seen: set[str] = set()
    for units_key, snapshot_key in phase_specs:
        document_path = require_absolute(paths[units_key], units_key)
        snapshot_path = require_absolute(paths[snapshot_key], snapshot_key)
        document = load_json(document_path)
        snapshot = load_json(snapshot_path)
        units = document.get("planned_units")
        if not isinstance(units, list):
            raise ContractError(f"{units_key} lacks planned_units[]")
        if document.get("declared_counts") != count_units(units):
            raise ContractError(f"{units_key} declared counts differ from enumeration")
        digest = sha256_file(snapshot_path)
        for unit in units:
            unit_id = unit["planned_unit_id"]
            if unit_id in seen:
                raise ContractError(f"duplicate planned unit across immutable arrays: {unit_id}")
            seen.add(unit_id)
            all_units.append(unit)
            authorizing_hash[unit_id] = digest
    final_snapshot = load_json(require_absolute(paths["effective_plan_e4"], "final effective plan"))
    if final_snapshot.get("planned_unit_ids") != [unit["planned_unit_id"] for unit in all_units]:
        raise ContractError("final effective plan does not enumerate the immutable unit arrays")
    if final_snapshot.get("declared_counts") != count_units(all_units):
        raise ContractError("final effective-plan counts differ from unit enumeration")
    return all_units, authorizing_hash


def validate_summary_inputs(
    manifest: dict[str, Any],
    samples: list[dict[str, Any]],
    events: list[dict[str, Any]],
    correctness: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    units, authorizing_hash = _load_planned_units(manifest)
    totals = count_units(units)
    require_unique(samples, "sample_id", "sample ID")
    require_unique(events, "event_record_id", "event ID")
    require_unique(correctness, "correctness_record_id", "correctness ID")
    require_unique(controls, "snapshot_id", "control snapshot ID")
    if len(samples) != totals["timing_rows"]:
        raise ContractError(f"summary refuses timing-count mismatch: expected {totals['timing_rows']}, observed {len(samples)}")
    if len(events) != totals["event_rows"]:
        raise ContractError(f"summary refuses event-count mismatch: expected {totals['event_rows']}, observed {len(events)}")
    if len(correctness) != totals["correctness_rows"]:
        raise ContractError(f"summary refuses correctness-count mismatch: expected {totals['correctness_rows']}, observed {len(correctness)}")
    if len(controls) != totals["control_snapshots"]:
        raise ContractError(f"summary refuses control-count mismatch: expected {totals['control_snapshots']}, observed {len(controls)}")
    started = {row["attempt_id"]: row for row in attempts if row.get("record_type") == "attempt_started"}
    finished = {row["attempt_id"]: row for row in attempts if row.get("record_type") == "attempt_finished"}
    if len(started) != totals["attempts"] or set(started) != set(finished) or len(attempts) != totals["attempt_rows"]:
        raise ContractError("summary refuses an incomplete or count-mismatched attempt journal")
    expected_units = Counter(unit["planned_unit_id"] for unit in units)
    observed_units = Counter(row["planned_unit_id"] for row in started.values())
    if observed_units != expected_units:
        raise ContractError("summary refuses attempts that do not close over the planned units")
    for attempt_id, row in started.items():
        expected_hash = authorizing_hash[row["planned_unit_id"]]
        if row["effective_plan_sha256"] != expected_hash:
            raise ContractError(f"attempt {attempt_id} cites an unauthorized effective-plan hash")
        if sha256_bytes(canonical_json_bytes(row["command"])) != row["command_sha256"]:
            raise ContractError(f"attempt {attempt_id} command digest does not resolve")
        if sha256_bytes(canonical_json_bytes(row["environment"])) != row["environment_sha256"]:
            raise ContractError(f"attempt {attempt_id} environment digest does not resolve")
    correctness_by_id = {row["correctness_record_id"]: row for row in correctness}
    events_by_id = {row["event_record_id"]: row for row in events}
    for sample in samples:
        if sample["attempt_id"] not in started:
            raise ContractError(f"sample {sample['sample_id']} references an unknown attempt")
        if not _valid_sample(sample):
            continue
        if any(
            identifier not in correctness_by_id or correctness_by_id[identifier].get("passed") is not True
            for identifier in sample["correctness_record_ids"]
        ):
            raise ContractError(f"sample {sample['sample_id']} has unresolved/failing correctness evidence")
        if any(identifier not in events_by_id for identifier in sample["event_record_ids"]):
            raise ContractError(f"sample {sample['sample_id']} has unresolved event evidence")
    require_unique(statuses, "experiment_id", "experiment status")
    if {row["experiment_id"] for row in statuses} != {f"E{index}" for index in range(1, 9)}:
        raise ContractError("summary requires exactly one terminal status for E1 through E8")
    return finished


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ContractError(f"refusing empty summary CSV: {path.name}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_new_bytes(path, stream.getvalue().encode("utf-8"))


def _format_number(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown(
    metadata: dict[str, Any],
    latency_rows: Sequence[dict[str, Any]],
    throughput_rows: Sequence[dict[str, Any]],
    accuracy_rows: Sequence[dict[str, Any]],
    speedup: dict[str, Any],
    statuses: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# FPT 2026 generated paper tables",
        "",
        f"Analysis algorithm: `{metadata['algorithm_version']}`; percentile algorithm: "
        f"`{metadata['percentile_algorithm']}`; bootstrap: `{metadata['bootstrap_algorithm']}`, "
        f"{metadata['bootstrap_resamples']:,} resamples, seed `{metadata['analysis_seed']}`.",
        "",
        "All latency rows retain every valid official observation. Outliers are flagged but not removed.",
        "",
        "## Latency and matched speedup",
        "",
        "| Experiment | Backend | Phase | Batch | Metric | n | Median (ms) | 95% CI (ms) |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    selected_latency = [
        row for row in latency_rows
        if row["metric"] == "application_e2e" and row["experiment_id"] in {"E2", "E6"}
    ]
    for row in selected_latency:
        lower = None if row["ci95_lower"] is None else row["ci95_lower"] / 1_000_000.0
        upper = None if row["ci95_upper"] is None else row["ci95_upper"] / 1_000_000.0
        interval = "not available" if lower is None else f"{lower:.6g}–{upper:.6g}"
        lines.append(
            f"| {row['experiment_id']} | {row['backend']} | {row['phase']} | {row['batch_size']} | "
            f"{row['metric']} | {row['observation_count']} | {row['median'] / 1_000_000.0:.6g} | {interval} |"
        )
    lines.extend(
        [
            "",
            f"Matched E6 speedup (median SEAL-Embedded / median FPGA, `application_e2e_v1`): "
            f"**{speedup['ratio_of_medians']:.6g}×** "
            f"(bootstrap 95% CI {speedup['ci95_lower']:.6g}–{speedup['ci95_upper']:.6g}).",
            "",
            "## Sustained throughput",
            "",
            "| Batch | Boundary | n | Median frames/s | 95% CI |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in throughput_rows:
        interval = (
            "not available"
            if row["ci95_lower"] is None
            else f"{row['ci95_lower']:.6g}–{row['ci95_upper']:.6g}"
        )
        lines.append(
            f"| {row['batch_size']} | {row['metric']} | {row['observation_count']} | "
            f"{row['median']:.6g} | {interval} |"
        )
    lines.extend(
        [
            "",
            "## Numerical accuracy",
            "",
            "| Backend | Case | Metric | n | Median | Maximum |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in accuracy_rows:
        if row["metric"] == "max_abs_error" and row["experiment_id"] == "E5":
            lines.append(
                f"| {row['backend']} | {row['case_id']} | {row['metric']} | "
                f"{row['observation_count']} | {row['median']:.6g} | {row['max']:.6g} |"
            )
    e7 = next(row for row in statuses if row["experiment_id"] == "E7")
    lines.extend(
        [
            "",
            "## Prior-work context (not a matched speedup baseline)",
            "",
            "| Work/platform | Parameters | Boundary | Frequency | Latency | Throughput | Classification and caveat | Source |",
            "|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for evidence in e7.get("source_evidence", []):
        parameters = evidence.get("parameter_set", {})
        frequency = evidence.get("frequency", {})
        latency = evidence.get("latency", {})
        throughput = evidence.get("throughput", {})
        lines.append(
            f"| Aloha-HE / {evidence.get('platform')} | n={parameters.get('poly_modulus_degree')}, "
            f"{parameters.get('coefficient_modulus')} | {evidence.get('operation_boundary')} | "
            f"{frequency.get('value')} {frequency.get('unit')} | {latency.get('value')} {latency.get('unit')} | "
            f"{throughput.get('unavailable_reason') or throughput.get('value')} | {evidence.get('classification')}; "
            f"{parameters.get('comparison_caveat')} | "
            f"`{evidence['source_path']}`, {evidence['locator']}, SHA-256 `{evidence['source_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "No cross-platform or unmatched-operation ratio is calculated from the E7 rows. E6 alone supplies the matched denominator.",
            "",
        ]
    )
    return "\n".join(lines)


def _line_svg(title: str, points: Sequence[tuple[float, float]], x_label: str, y_label: str) -> bytes:
    width, height = 800, 480
    left, right, top, bottom = 90, 30, 55, 75
    if not points:
        body = f'<text x="{width/2}" y="{height/2}" text-anchor="middle">no data</text>'
    else:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        if x_max == x_min:
            x_max = x_min + 1.0
        if y_max == y_min:
            y_max = y_min + 1.0
        coordinates = []
        circles = []
        for x_value, y_value in points:
            x = left + (x_value - x_min) * (width - left - right) / (x_max - x_min)
            y = top + (y_max - y_value) * (height - top - bottom) / (y_max - y_min)
            coordinates.append(f"{x:.3f},{y:.3f}")
            circles.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="#005a9c"/>')
        body = f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="#005a9c" stroke-width="2"/>' + "".join(circles)
    document = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>
<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif">{html.escape(x_label)}</text>
<text x="22" y="{height/2}" text-anchor="middle" font-family="sans-serif" transform="rotate(-90 22 {height/2})">{html.escape(y_label)}</text>
{body}
</svg>
'''
    return document.encode("utf-8")


def _bar_svg(title: str, labels: Sequence[str], values: Sequence[float], y_label: str) -> bytes:
    width, height = 900, 480
    left, right, top, bottom = 90, 30, 55, 105
    maximum = max(values) if values else 1.0
    if maximum <= 0:
        maximum = 1.0
    plot_width = width - left - right
    bar_width = plot_width / max(len(values), 1) * 0.7
    body: list[str] = []
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + (index + 0.15) * plot_width / max(len(values), 1)
        bar_height = value * (height - top - bottom) / maximum
        y = height - bottom - bar_height
        body.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{bar_width:.3f}" height="{bar_height:.3f}" fill="#005a9c"/>')
        body.append(f'<text x="{x + bar_width/2:.3f}" y="{height-bottom+18}" text-anchor="end" font-family="sans-serif" font-size="10" transform="rotate(-35 {x + bar_width/2:.3f} {height-bottom+18})">{html.escape(label)}</text>')
    document = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>
<text x="22" y="{height/2}" text-anchor="middle" font-family="sans-serif" transform="rotate(-90 22 {height/2})">{html.escape(y_label)}</text>
{"".join(body)}
</svg>
'''
    return document.encode("utf-8")


def main() -> int:
    args = parse_args()
    run_root = require_absolute(args.run_root.resolve(), "run root")
    manifest_path = run_root / "manifest.json"
    manifest = load_json(manifest_path)
    if require_absolute(manifest["paths"]["run_root"]) != run_root:
        raise ContractError("manifest run root differs from the requested result directory")
    analysis_seed = args.analysis_seed
    if analysis_seed is None:
        analysis_seed = int(sha256_file(manifest_path)[:16], 16)
    if analysis_seed < 0:
        raise ContractError("analysis seed must be nonnegative")

    paths = manifest["paths"]
    samples = read_jsonl(require_absolute(paths["samples"]))
    events = read_jsonl(require_absolute(paths["events"]))
    correctness = read_jsonl(require_absolute(paths["correctness"]))
    attempts = read_jsonl(require_absolute(paths["attempts"]))
    controls = read_jsonl(require_absolute(paths["controls"]))
    statuses = read_jsonl(require_absolute(paths["experiment_status"]))
    finished = validate_summary_inputs(
        manifest, samples, events, correctness, attempts, controls, statuses
    )

    latency_rows = latency_summary_rows(samples, finished, analysis_seed)
    throughput_rows = throughput_summary_rows(samples, finished, analysis_seed)
    accuracy_rows = accuracy_summary_rows(correctness, analysis_seed)
    speedup = speedup_summary(samples, analysis_seed)
    final_batches = sorted({int(row["batch_size"]) for row in samples if row.get("experiment_id") == "E8"})
    e8_evidence = e8_evidence_document(samples, required_batches=final_batches)
    metadata = {
        "schema_version": "1.0",
        "algorithm_version": ALGORITHM_VERSION,
        "analysis_seed": analysis_seed,
        "percentile_algorithm": PERCENTILE_ALGORITHM,
        "bootstrap_algorithm": BOOTSTRAP_ALGORITHM,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "minimum_ci_observations": MINIMUM_CI_OBSERVATIONS,
        "outlier_policy": "flag_tukey_and_mad_keep_all_successful_observations",
        "manifest_sha256": sha256_file(manifest_path),
    }

    summary_root = run_root / "summaries"
    plot_root = run_root / "plots"
    summary_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_root / "latency.csv", latency_rows)
    _write_csv(summary_root / "throughput.csv", throughput_rows)
    _write_csv(summary_root / "accuracy.csv", accuracy_rows)
    write_new_json(summary_root / "analysis-metadata.json", metadata)
    write_new_json(summary_root / "speedup.json", speedup)
    write_new_json(summary_root / "e8-throughput-evidence.json", e8_evidence)
    write_new_json(
        summary_root / "e1-e8-status.json",
        {
            "schema_version": "1.0",
            "source_path": str(require_absolute(paths["experiment_status"])),
            "source_sha256": sha256_file(require_absolute(paths["experiment_status"])),
            "statuses": statuses,
        },
    )
    write_new_bytes(
        summary_root / "paper_tables.md",
        _markdown(metadata, latency_rows, throughput_rows, accuracy_rows, speedup, statuses).encode("utf-8"),
    )

    e2_stage_rows = [
        row for row in latency_rows
        if row["experiment_id"] == "E2"
        and row["backend"] == "fpga"
        and row["metric"] in {
            "preparation", "pack", "h2d_wall", "graph_submit_wait_wall", "d2h_wall",
            "unpack_and_assembly", "unattributed_wall",
        }
    ]
    _write_svg_latency = _bar_svg(
        "E2 median wall-time breakdown",
        [str(row["metric"]) for row in e2_stage_rows],
        [float(row["median"]) / 1_000_000.0 for row in e2_stage_rows],
        "milliseconds",
    )
    write_new_bytes(plot_root / "latency_breakdown.svg", _write_svg_latency)
    application_throughput = [
        row for row in throughput_rows if row["metric"] == "application_e2e_frames_per_second"
    ]
    write_new_bytes(
        plot_root / "throughput_sweep.svg",
        _line_svg(
            "E8 sustained throughput",
            [(float(row["batch_size"]), float(row["median"])) for row in application_throughput],
            "batch size",
            "frames per second",
        ),
    )
    accuracy_plot_rows = [
        row for row in accuracy_rows
        if row["experiment_id"] == "E5" and row["metric"] == "max_abs_error"
    ]
    write_new_bytes(
        plot_root / "accuracy.svg",
        _bar_svg(
            "E5 maximum absolute error",
            [f"{row['backend']}:{row['case_id']}" for row in accuracy_plot_rows],
            [float(row["max"]) for row in accuracy_plot_rows],
            "absolute error",
        ),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, FileExistsError, FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"summarize_results: {error}", file=sys.stderr)
        raise SystemExit(2)
