#!/usr/bin/env python3
"""Materialize the predeclared E8 and E4 planning branches immutably."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common import ContractError, load_json, require_absolute, sha256_file, utc_now, write_new_json
from plan_matrix import (
    base_units,
    count_units,
    e4_sustained_units,
    e8_extension_units,
    planned_document,
)

BASE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
EXTENSION_BATCHES = [256, 512]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("e8", "e4"):
        child = subparsers.add_parser(phase)
        child.add_argument("--manifest", required=True, type=Path)
        child.add_argument("--evidence", required=True, type=Path)
    return parser.parse_args()


def combined_counts(*unit_groups: list[dict[str, Any]]) -> dict[str, int]:
    return count_units([unit for group in unit_groups for unit in group])


def load_throughput_points(path: Path, required_batches: list[int]) -> list[dict[str, Any]]:
    evidence = load_json(path)
    points = evidence.get("throughput_points")
    if not isinstance(points, list):
        raise ContractError("E8 evidence requires throughput_points[]")
    by_batch: dict[int, dict[str, Any]] = {}
    for point in points:
        batch = point.get("batch_size")
        if not isinstance(batch, int) or batch in by_batch:
            raise ContractError(f"invalid or duplicate E8 batch size: {batch!r}")
        median = point.get("median_e2e_frames_per_second")
        if not isinstance(median, (int, float)) or median <= 0:
            raise ContractError(f"batch {batch} has invalid median throughput")
        if point.get("correctness_qualified") is not True or point.get("liveness_qualified") is not True:
            raise ContractError(f"batch {batch} is not correctness- and liveness-qualified")
        by_batch[batch] = point
    missing = [batch for batch in required_batches if batch not in by_batch]
    if missing:
        raise ContractError(f"E8 evidence is missing required batches: {missing}")
    return [by_batch[batch] for batch in sorted(required_batches)]


def saturation_batch(points: list[dict[str, Any]]) -> int | None:
    medians = [float(point["median_e2e_frames_per_second"]) for point in points]
    maximum = max(medians)
    threshold = 0.95 * maximum
    for index in range(len(points) - 2):
        window = medians[index:index + 3]
        if all(value >= threshold for value in window):
            return int(points[index]["batch_size"])
    return None


def snapshot(
    phase: str,
    predecessor_sha256: str | None,
    component_sha256s: list[str],
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "document_type": "effective_plan_snapshot",
        "phase": phase,
        "predecessor_sha256": predecessor_sha256,
        "component_sha256s": component_sha256s,
        "planned_unit_ids": [unit["planned_unit_id"] for unit in units],
        "declared_counts": count_units(units),
        "created_utc": utc_now(),
    }


def finalize_e8(manifest: dict[str, Any], evidence_path: Path) -> None:
    paths = manifest["paths"]
    base_snapshot_path = require_absolute(paths["effective_plan_base"])
    base_snapshot_sha256 = sha256_file(base_snapshot_path)
    if base_snapshot_sha256 != manifest["base_effective_plan_sha256"]:
        raise ContractError("base effective-plan hash differs from immutable manifest")
    base_points = load_throughput_points(evidence_path, BASE_BATCHES)
    base_saturation = saturation_batch(base_points)
    triggered = base_saturation is None
    extension_units = e8_extension_units(triggered)
    extension_path = require_absolute(paths["planned_units_e8_extension"])
    extension_sha256 = write_new_json(
        extension_path,
        planned_document("planned_units_e8_extension", extension_units),
    )
    evidence_sha256 = sha256_file(evidence_path)
    amendment = {
        "schema_version": "1.0",
        "document_type": "manifest_amendment_e8",
        "triggered": triggered,
        "terminal_status": "applicable" if triggered else "not_applicable",
        "base_effective_plan_sha256": base_snapshot_sha256,
        "evidence_sha256": evidence_sha256,
        "extension_plan_sha256": extension_sha256,
        "declared_counts": count_units(extension_units),
        "saturation_analysis": {
            "rule": "first point and next two points all at least 95 percent of maximum median",
            "evidence_path": str(evidence_path),
            "base_batches": BASE_BATCHES,
            "base_saturation_batch": base_saturation,
            "maximum_median_e2e_frames_per_second": max(
                float(point["median_e2e_frames_per_second"]) for point in base_points
            ),
        },
        "reason": "base_saturation_not_reached" if triggered else "base_saturation_established",
        "created_utc": utc_now(),
    }
    amendment_path = require_absolute(paths["manifest_amendment_e8"])
    amendment_sha256 = write_new_json(amendment_path, amendment)
    all_units = base_units() + extension_units
    effective = snapshot(
        "base+e8",
        base_snapshot_sha256,
        [manifest["base_plan_sha256"], extension_sha256, amendment_sha256],
        all_units,
    )
    write_new_json(require_absolute(paths["effective_plan_e8"]), effective)


def finalize_e4(manifest: dict[str, Any], evidence_path: Path) -> None:
    paths = manifest["paths"]
    e8_snapshot_path = require_absolute(paths["effective_plan_e8"])
    e8_snapshot_sha256 = sha256_file(e8_snapshot_path)
    e8_amendment_path = require_absolute(paths["manifest_amendment_e8"])
    e8_amendment = load_json(e8_amendment_path)
    required_batches = BASE_BATCHES + (EXTENSION_BATCHES if e8_amendment["triggered"] else [])
    points = load_throughput_points(evidence_path, required_batches)
    selected_saturation = saturation_batch(points)
    batch_size = selected_saturation if selected_saturation is not None and selected_saturation <= 64 else 64
    sustained_units = e4_sustained_units(batch_size)
    sustained_path = require_absolute(paths["planned_units_e4_sustained"])
    sustained_sha256 = write_new_json(
        sustained_path,
        planned_document("planned_units_e4_sustained", sustained_units),
    )
    metadata = sustained_units[0]["metadata"]
    evidence_sha256 = sha256_file(evidence_path)
    amendment = {
        "schema_version": "1.0",
        "document_type": "manifest_amendment_e4",
        "terminal_status": "applicable",
        "B_E4": batch_size,
        "e8_effective_plan_sha256": e8_snapshot_sha256,
        "e8_evidence_sha256": evidence_sha256,
        "full_batches": metadata["full_batches"],
        "partial_batches": 1 if metadata["final_partial_batch_size"] else 0,
        "final_partial_batch_size": metadata["final_partial_batch_size"],
        "timing_rows": sustained_units[0]["timing_rows"],
        "sustained_plan_sha256": sustained_sha256,
        "declared_counts": count_units(sustained_units),
        "created_utc": utc_now(),
    }
    amendment_path = require_absolute(paths["manifest_amendment_e4"])
    amendment_sha256 = write_new_json(amendment_path, amendment)
    extension_document = load_json(require_absolute(paths["planned_units_e8_extension"]))
    all_units = base_units() + extension_document["planned_units"] + sustained_units
    components = [
        manifest["base_plan_sha256"],
        sha256_file(require_absolute(paths["planned_units_e8_extension"])),
        sustained_sha256,
        sha256_file(e8_amendment_path),
        amendment_sha256,
    ]
    effective = snapshot("base+e8+e4", e8_snapshot_sha256, components, all_units)
    write_new_json(require_absolute(paths["effective_plan_e4"]), effective)


def main() -> int:
    args = parse_args()
    manifest_path = require_absolute(args.manifest.resolve(), "manifest")
    evidence_path = require_absolute(args.evidence.resolve(), "evidence")
    manifest = load_json(manifest_path)
    if args.phase == "e8":
        finalize_e8(manifest, evidence_path)
    else:
        finalize_e4(manifest, evidence_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, FileExistsError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"finalize_plans: {error}", file=sys.stderr)
        raise SystemExit(2)
