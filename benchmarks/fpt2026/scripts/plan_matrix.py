#!/usr/bin/env python3
"""Exact executable-unit definitions for fpt2026-paper-matrix-v1."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

from common import ContractError

SCHEMA_VERSION = "1.0"
MATRIX_ID = "fpt2026-paper-matrix-v1"
PERFORMANCE_EVENTS = 27
ONE_NTT_EVENTS = 39
FULL_DIAGNOSTIC_EVENTS = 51
REAL_CASES = [
    "real_impulse", "real_short_mixed", "real_partial_64",
    "real_partial_1024", "real_full_4096",
]
COMPLEX_CASES = [
    "complex_short_mixed", "complex_partial_64", "complex_full_4096",
]
ALL_CASES = REAL_CASES + COMPLEX_CASES

COUNT_KEYS = (
    "attempts",
    "attempt_rows",
    "control_snapshots",
    "frames",
    "event_rows",
    "correctness_rows",
    "timing_rows",
)

EXPECTED_BASE_COUNTS = {
    "attempts": 32,
    "attempt_rows": 64,
    "control_snapshots": 64,
    "frames": 5654,
    "event_rows": 150336,
    "correctness_rows": 5722,
    "timing_rows": 350,
}

EXPECTED_E8_EXTENSION_COUNTS = {
    "attempts": 1,
    "attempt_rows": 2,
    "control_snapshots": 2,
    "frames": 16128,
    "event_rows": 435456,
    "correctness_rows": 16128,
    "timing_rows": 40,
}


def _unit(
    unit_id: str,
    experiment: str,
    phase: str,
    backend: str,
    frames: int,
    events: int,
    correctness: int,
    timing: int,
    conditional_branch: bool = False,
    **metadata: Any,
) -> dict[str, Any]:
    unit = {
        "planned_unit_id": unit_id,
        "experiment_id": experiment,
        "backend": backend,
        "child_attempts": 1,
        "frames": frames,
        "event_rows": events,
        "correctness_rows": correctness,
        "timing_rows": timing,
        "metadata": {"phase": phase, **metadata},
    }
    if conditional_branch:
        unit["initial_status"] = "pending"
    return unit


def rotated_passes(values: list[int], passes: int) -> list[list[int]]:
    if not values:
        raise ContractError("rotation requires at least one batch size")
    return [values[offset % len(values):] + values[:offset % len(values)] for offset in range(passes)]


def base_units() -> list[dict[str, Any]]:
    units = [
        _unit("E1-C1-C2", "E1", "correctness_patterns", "fpga", 4, 4 * FULL_DIAGNOSTIC_EVENTS, 72, 0,
              mode="full_diagnostics", patterns=[
                  "ntt_sparse_impulse", "ntt_alternating_ternary",
                  "ntt_negative_boundary", "ntt_shake256",
              ], selectors=list(range(6)), c1_roles=["NTT-A", "NTT-B"]),
        _unit("E1-FPGA-TEST", "E1", "complete_fpga_test", "fpga", 5, 5 * FULL_DIAGNOSTIC_EVENTS, 5, 0,
              mode="full_diagnostics", case_ids=REAL_CASES),
        _unit("E1-C3", "E1", "standalone_semantic", "fpga", 8, 8 * PERFORMANCE_EVENTS, 8, 0,
              mode="performance", case_ids=ALL_CASES),
        _unit("E2-LATENCY", "E2", "warm_latency", "fpga", 54, 54 * PERFORMANCE_EVENTS, 54, 50,
              mode="performance", case_ids=["real_full_4096"], warmups=4,
              measured_repetitions=50, batch_size=1),
        _unit("E3-REPORT", "E3", "report_extraction", "extraction", 0, 0, 0, 0),
    ]
    for process_index in range(20):
        units.append(_unit(
            f"E4-PROCESS-{process_index:02d}", "E4", "process_start", "fpga", 2,
            2 * PERFORMANCE_EVENTS, 2, 2, mode="performance", process_index=process_index,
            case_ids=["real_short_mixed"], frame_indices=[0, 1],
            first_result_frames=1, warm_frames=1,
        ))
    units.extend([
        _unit("E5-FPGA", "E5", "semantic_matrix", "fpga", 40, 40 * PERFORMANCE_EVENTS, 40, 0,
              mode="performance", case_ids=ALL_CASES, trial_seed_indices=list(range(5))),
        _unit("E5-SEAL-EMBEDDED", "E5", "semantic_matrix", "seal-embedded", 25, 0, 25, 0,
              case_ids=REAL_CASES, trial_seed_indices=list(range(5))),
        _unit("E5-STOCK-SEAL", "E5", "semantic_matrix", "stock-seal", 15, 0, 15, 0,
              case_ids=COMPLEX_CASES, trial_seed_indices=list(range(5)),
              reference_role="stock-seal-reference"),
        _unit("E6-FPGA", "E6", "matched_speedup", "fpga", 54, 54 * PERFORMANCE_EVENTS, 54, 50,
              mode="performance", case_ids=["real_full_4096"], warmups=4,
              measured_repetitions=50),
        _unit("E6-SEAL-EMBEDDED", "E6", "matched_speedup", "seal-embedded", 54, 0, 54, 50,
              case_ids=["real_full_4096"], warmups=4, measured_repetitions=50),
        _unit("E7-PRIOR-WORK", "E7", "prior_work_extraction", "extraction", 0, 0, 0, 0),
    ])
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    frames = 21 * sum(batch_sizes)
    units.append(_unit(
        "E8-BASE", "E8", "batch_scaling_base", "fpga", frames,
        frames * PERFORMANCE_EVENTS, frames, 20 * len(batch_sizes), mode="performance",
        max_resident_batch=128, batch_sizes=batch_sizes,
        case_ids=["real_full_4096"],
        warmup_batches=[{"batch_size": value, "frames": value} for value in batch_sizes],
        measured_passes=rotated_passes(batch_sizes, 20),
    ))
    assert_counts(units, EXPECTED_BASE_COUNTS, "base plan")
    return units


def e8_extension_units(triggered: bool) -> list[dict[str, Any]]:
    if not triggered:
        return []
    batch_sizes = [256, 512]
    frames = 21 * sum(batch_sizes)
    units = [_unit(
        "E8-EXTENSION", "E8", "batch_scaling_extension", "fpga", frames,
        frames * PERFORMANCE_EVENTS, frames, 20 * len(batch_sizes), conditional_branch=True,
        mode="performance",
        max_resident_batch=512, batch_sizes=batch_sizes,
        case_ids=["real_full_4096"],
        warmup_batches=[{"batch_size": value, "frames": value} for value in batch_sizes],
        measured_passes=rotated_passes(batch_sizes, 20),
    )]
    assert_counts(units, EXPECTED_E8_EXTENSION_COUNTS, "E8 extension")
    return units


def e4_sustained_units(batch_size: int) -> list[dict[str, Any]]:
    if batch_size < 1 or batch_size > 64:
        raise ContractError(f"B_E4 must be in [1, 64], received {batch_size}")
    full_batches, remainder = divmod(10000, batch_size)
    batches = full_batches + (1 if remainder else 0)
    units = [_unit(
        "E4-SUSTAINED", "E4", "sustained", "fpga", 10000,
        10000 * PERFORMANCE_EVENTS, 10000, batches, conditional_branch=True,
        mode="performance",
        case_ids=ALL_CASES, trial_seed_indices=list(range(5)),
        vector_seed_schedule="cartesian_cycle_case_then_seed",
        batch_size=batch_size, full_batches=full_batches,
        final_partial_batch_size=remainder, frame_index_range={"start": 0, "count": 10000},
    )]
    expected = {
        "attempts": 1, "attempt_rows": 2, "control_snapshots": 2,
        "frames": 10000, "event_rows": 270000, "correctness_rows": 10000,
        "timing_rows": math.ceil(10000 / batch_size),
    }
    assert_counts(units, expected, "E4 sustained")
    return units


def count_units(units: Iterable[dict[str, Any]]) -> dict[str, int]:
    materialized = list(units)
    attempts = sum(int(unit["child_attempts"]) for unit in materialized)
    return {
        "attempts": attempts,
        "attempt_rows": attempts * 2,
        "control_snapshots": attempts * 2,
        "frames": sum(int(unit["frames"]) for unit in materialized),
        "event_rows": sum(int(unit["event_rows"]) for unit in materialized),
        "correctness_rows": sum(int(unit["correctness_rows"]) for unit in materialized),
        "timing_rows": sum(int(unit["timing_rows"]) for unit in materialized),
    }


def assert_counts(units: Iterable[dict[str, Any]], expected: dict[str, int], label: str) -> None:
    actual = count_units(units)
    if actual != expected:
        raise ContractError(f"{label} count mismatch: expected {expected}, observed {actual}")


def assert_unique_unit_ids(units: Iterable[dict[str, Any]]) -> None:
    counts = Counter(unit["planned_unit_id"] for unit in units)
    duplicates = sorted(unit_id for unit_id, count in counts.items() if count != 1)
    if duplicates:
        raise ContractError(f"duplicate planned-unit IDs: {duplicates}")


def planned_document(document_type: str, units: list[dict[str, Any]]) -> dict[str, Any]:
    assert_unique_unit_ids(units)
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": document_type,
        "planned_units": units,
        "declared_counts": count_units(units),
    }
