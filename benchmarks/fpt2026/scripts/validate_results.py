#!/usr/bin/env python3
"""Mechanically validate a complete FPT 2026 result directory."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from common import (
    ContractError,
    canonical_json_bytes,
    load_json,
    read_jsonl,
    require_absolute,
    require_unique,
    sha256_file,
    sha256_bytes,
    sha256_tree,
)
from plan_matrix import (
    ALL_CASES,
    COMPLEX_CASES,
    EXPECTED_BASE_COUNTS,
    REAL_CASES,
    base_units,
    count_units,
    e4_sustained_units,
    e8_extension_units,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


@dataclass(frozen=True)
class SchemaIssue:
    absolute_path: tuple[str | int, ...]
    message: str


class StandardLibrarySchemaValidator:
    """Validate the bounded Draft 2020-12 keyword subset used in this project."""

    def __init__(self, schema: dict[str, Any], root: dict[str, Any] | None = None):
        self.schema = schema
        self.root = root if root is not None else schema

    def iter_errors(self, value: Any) -> list[SchemaIssue]:
        issues: list[SchemaIssue] = []
        self._validate(value, self.schema, (), issues)
        return issues

    def _resolve(self, reference: str) -> dict[str, Any]:
        if not reference.startswith("#/"):
            raise ContractError(f"only local schema references are supported: {reference}")
        current: Any = self.root
        for token in reference[2:].split("/"):
            key = token.replace("~1", "/").replace("~0", "~")
            current = current[key]
        if not isinstance(current, dict):
            raise ContractError(f"schema reference does not resolve to an object: {reference}")
        return current

    @staticmethod
    def _is_type(value: Any, expected: str) -> bool:
        return {
            "null": value is None,
            "boolean": isinstance(value, bool),
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        }.get(expected, False)

    def _branch_valid(self, value: Any, schema: dict[str, Any], path: tuple[str | int, ...]) -> bool:
        branch_issues: list[SchemaIssue] = []
        self._validate(value, schema, path, branch_issues)
        return not branch_issues

    def _validate(
        self,
        value: Any,
        schema: dict[str, Any],
        path: tuple[str | int, ...],
        issues: list[SchemaIssue],
    ) -> None:
        if "$ref" in schema:
            self._validate(value, self._resolve(schema["$ref"]), path, issues)
        if "const" in schema and value != schema["const"]:
            issues.append(SchemaIssue(path, f"value must equal {schema['const']!r}"))
        if "enum" in schema and value not in schema["enum"]:
            issues.append(SchemaIssue(path, f"value is not in {schema['enum']!r}"))

        expected_types = schema.get("type")
        if expected_types is not None:
            if isinstance(expected_types, str):
                expected_types = [expected_types]
            if not any(self._is_type(value, expected) for expected in expected_types):
                issues.append(SchemaIssue(path, f"expected type {expected_types!r}"))
                return

        if "oneOf" in schema:
            matches = sum(self._branch_valid(value, branch, path) for branch in schema["oneOf"])
            if matches != 1:
                issues.append(SchemaIssue(path, f"oneOf matched {matches} branches instead of one"))
        for branch in schema.get("allOf", []):
            self._validate(value, branch, path, issues)
        if "if" in schema:
            selected = schema.get("then") if self._branch_valid(value, schema["if"], path) else schema.get("else")
            if selected is not None:
                self._validate(value, selected, path, issues)

        if isinstance(value, dict):
            for key in schema.get("required", []):
                if key not in value:
                    issues.append(SchemaIssue(path, f"missing required property {key!r}"))
            properties = schema.get("properties", {})
            for key, child in value.items():
                if key in properties:
                    self._validate(child, properties[key], path + (key,), issues)
                elif schema.get("additionalProperties") is False:
                    issues.append(SchemaIssue(path + (key,), "additional property is prohibited"))
                elif isinstance(schema.get("additionalProperties"), dict):
                    self._validate(child, schema["additionalProperties"], path + (key,), issues)

        if isinstance(value, list):
            if len(value) < schema.get("minItems", 0):
                issues.append(SchemaIssue(path, f"array has fewer than {schema['minItems']} items"))
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                issues.append(SchemaIssue(path, f"array has more than {schema['maxItems']} items"))
            if schema.get("uniqueItems"):
                encoded = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value]
                if len(encoded) != len(set(encoded)):
                    issues.append(SchemaIssue(path, "array items are not unique"))
            if isinstance(schema.get("items"), dict):
                for index, child in enumerate(value):
                    self._validate(child, schema["items"], path + (index,), issues)

        if isinstance(value, str):
            if len(value) < schema.get("minLength", 0):
                issues.append(SchemaIssue(path, f"string is shorter than {schema['minLength']}"))
            if "pattern" in schema and re.search(schema["pattern"], value) is None:
                issues.append(SchemaIssue(path, f"string does not match {schema['pattern']!r}"))
            if schema.get("format") == "date-time":
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        raise ValueError("timezone is required")
                except ValueError:
                    issues.append(SchemaIssue(path, "string is not a timezone-aware date-time"))

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                issues.append(SchemaIssue(path, f"number is less than {schema['minimum']}"))
            if "maximum" in schema and value > schema["maximum"]:
                issues.append(SchemaIssue(path, f"number is greater than {schema['maximum']}"))
            if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
                issues.append(SchemaIssue(path, f"number is not greater than {schema['exclusiveMinimum']}"))


def schema_validator(
    schema: dict[str, Any], definition: str | None = None
) -> StandardLibrarySchemaValidator:
    selected = schema if definition is None else schema["$defs"][definition]
    return StandardLibrarySchemaValidator(selected, root=schema)


def validate_document(value: Any, validator: Any, label: str) -> None:
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.absolute_path))
    if errors:
        rendered = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors[:20]
        )
        raise ContractError(f"{label} failed schema validation: {rendered}")


def load_and_validate_json(path: Path, validator: Any, label: str) -> Any:
    value = load_json(path)
    validate_document(value, validator, label)
    return value


def load_and_validate_jsonl(path: Path, validator: Any, label: str) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    for index, record in enumerate(records, start=1):
        validate_document(record, validator, f"{label}:{index}")
    return records


def add_counts(*counts: dict[str, int]) -> dict[str, int]:
    keys = set().union(*(value.keys() for value in counts))
    return {key: sum(int(value.get(key, 0)) for value in counts) for key in keys}


def validate_artifacts(manifest: dict[str, Any]) -> None:
    for artifact in manifest["artifacts"]:
        path = require_absolute(artifact["path"], f"artifact {artifact['artifact_id']}")
        if not path.exists():
            raise ContractError(f"missing frozen artifact {artifact['artifact_id']}: {path}")
        observed = sha256_tree(path)[0] if path.is_dir() else sha256_file(path)
        if observed != artifact["sha256"]:
            raise ContractError(f"artifact hash changed for {artifact['artifact_id']}: {path}")
    configuration_path = require_absolute(manifest["paths"]["configuration"], "configuration")
    environment_path = require_absolute(manifest["paths"]["environment"], "environment")
    artifact_index_path = require_absolute(manifest["paths"]["artifact_index"], "artifact index")
    report_tree_path = require_absolute(manifest["paths"]["run_root"]) / "raw" / "report-tree.json"
    for label, path, expected in (
        ("configuration", configuration_path, manifest["configuration_sha256"]),
        ("environment", environment_path, manifest["environment_sha256"]),
        ("artifact index", artifact_index_path, manifest["artifact_index_sha256"]),
        ("report tree manifest", report_tree_path, manifest["report_tree_manifest_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ContractError(f"{label} hash differs from root manifest")
    expected_index = "".join(
        f"{artifact['sha256']}  {artifact['path']}\n"
        for artifact in sorted(manifest["artifacts"], key=lambda item: item["artifact_id"])
    )
    if artifact_index_path.read_text(encoding="utf-8") != expected_index:
        raise ContractError("artifacts.sha256 does not enumerate the frozen artifact set exactly")


def validate_planning(
    manifest: dict[str, Any],
    manifest_schema: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
    dict[str, dict[str, Any]],
    dict[str, tuple[str, str]],
]:
    paths = manifest["paths"]
    validators = {
        name: schema_validator(manifest_schema, name)
        for name in ("planned_unit_array_document", "e8_amendment", "e4_amendment", "effective_plan_snapshot")
    }
    base_path = require_absolute(paths["planned_units_base"])
    base = load_and_validate_json(base_path, validators["planned_unit_array_document"], "base planned units")
    if sha256_file(base_path) != manifest["base_plan_sha256"]:
        raise ContractError("base planned-unit hash differs from manifest")
    if base["planned_units"] != base_units() or base["declared_counts"] != EXPECTED_BASE_COUNTS:
        raise ContractError("base planned units differ from fpt2026-paper-matrix-v1")

    extension_path = require_absolute(paths["planned_units_e8_extension"])
    e8_amendment_path = require_absolute(paths["manifest_amendment_e8"])
    extension = load_and_validate_json(extension_path, validators["planned_unit_array_document"], "E8 extension")
    e8_amendment = load_and_validate_json(e8_amendment_path, validators["e8_amendment"], "E8 amendment")
    if sha256_file(extension_path) != e8_amendment["extension_plan_sha256"]:
        raise ContractError("E8 extension hash differs from amendment")
    expected_extension = e8_extension_units(bool(e8_amendment["triggered"]))
    if extension["planned_units"] != expected_extension:
        raise ContractError("E8 extension units disagree with its terminal branch")
    if extension["declared_counts"] != count_units(expected_extension):
        raise ContractError("E8 extension declared counts disagree with enumeration")
    expected_status = "applicable" if e8_amendment["triggered"] else "not_applicable"
    if e8_amendment["terminal_status"] != expected_status:
        raise ContractError("E8 terminal branch status disagrees with trigger")

    e4_path = require_absolute(paths["planned_units_e4_sustained"])
    e4_amendment_path = require_absolute(paths["manifest_amendment_e4"])
    e4 = load_and_validate_json(e4_path, validators["planned_unit_array_document"], "E4 sustained plan")
    e4_amendment = load_and_validate_json(e4_amendment_path, validators["e4_amendment"], "E4 amendment")
    if sha256_file(e4_path) != e4_amendment["sustained_plan_sha256"]:
        raise ContractError("E4 sustained-plan hash differs from amendment")
    expected_e4 = e4_sustained_units(int(e4_amendment["B_E4"]))
    if e4["planned_units"] != expected_e4:
        raise ContractError("E4 sustained units disagree with B_E4")
    if e4["declared_counts"] != count_units(expected_e4):
        raise ContractError("E4 sustained declared counts disagree with enumeration")

    base_snapshot_path = require_absolute(paths["effective_plan_base"])
    e8_snapshot_path = require_absolute(paths["effective_plan_e8"])
    e4_snapshot_path = require_absolute(paths["effective_plan_e4"])
    base_snapshot = load_and_validate_json(base_snapshot_path, validators["effective_plan_snapshot"], "base effective plan")
    e8_snapshot = load_and_validate_json(e8_snapshot_path, validators["effective_plan_snapshot"], "E8 effective plan")
    e4_snapshot = load_and_validate_json(e4_snapshot_path, validators["effective_plan_snapshot"], "E4 effective plan")
    base_snapshot_sha = sha256_file(base_snapshot_path)
    e8_snapshot_sha = sha256_file(e8_snapshot_path)
    e8_amendment_sha = sha256_file(e8_amendment_path)
    e4_amendment_sha = sha256_file(e4_amendment_path)
    if base_snapshot_sha != manifest["base_effective_plan_sha256"]:
        raise ContractError("base effective-plan hash differs from manifest")
    if base_snapshot["component_sha256s"] != [manifest["base_plan_sha256"]]:
        raise ContractError("base effective plan has the wrong component list")
    if e8_snapshot["predecessor_sha256"] != base_snapshot_sha:
        raise ContractError("E8 effective plan does not chain to base")
    if e8_amendment["base_effective_plan_sha256"] != base_snapshot_sha:
        raise ContractError("E8 amendment does not cite the base effective plan")
    expected_e8_components = [
        manifest["base_plan_sha256"], sha256_file(extension_path), e8_amendment_sha
    ]
    if e8_snapshot["component_sha256s"] != expected_e8_components:
        raise ContractError("E8 effective plan has the wrong component list")
    if e4_snapshot["predecessor_sha256"] != e8_snapshot_sha:
        raise ContractError("E4 effective plan does not chain to E8")
    if e4_amendment["e8_effective_plan_sha256"] != e8_snapshot_sha:
        raise ContractError("E4 amendment does not cite the E8 effective plan")
    expected_e4_components = expected_e8_components + [sha256_file(e4_path), e4_amendment_sha]
    if e4_snapshot["component_sha256s"] != expected_e4_components:
        raise ContractError("E4 effective plan has the wrong component list")

    all_units = base["planned_units"] + extension["planned_units"] + e4["planned_units"]
    unit_by_id = {unit["planned_unit_id"]: unit for unit in all_units}
    if len(unit_by_id) != len(all_units):
        raise ContractError("planned-unit IDs are not globally unique")
    final_counts = count_units(all_units)
    if e4_snapshot["planned_unit_ids"] != [unit["planned_unit_id"] for unit in all_units]:
        raise ContractError("final effective plan does not enumerate exact ordered unit IDs")
    if e4_snapshot["declared_counts"] != final_counts:
        raise ContractError("final effective-plan totals disagree with unit enumeration")
    expected_timing = 350 + (40 if e8_amendment["triggered"] else 0) + math.ceil(10000 / e4_amendment["B_E4"])
    if final_counts["timing_rows"] != expected_timing:
        raise ContractError("final timing total does not match branch formula")
    authorization_by_unit: dict[str, tuple[str, str]] = {}
    for unit in base["planned_units"]:
        authorization_by_unit[unit["planned_unit_id"]] = (
            base_snapshot_sha, base_snapshot["created_utc"]
        )
    for unit in extension["planned_units"]:
        authorization_by_unit[unit["planned_unit_id"]] = (
            e8_snapshot_sha, e8_snapshot["created_utc"]
        )
    e4_snapshot_sha = sha256_file(e4_snapshot_path)
    for unit in e4["planned_units"]:
        authorization_by_unit[unit["planned_unit_id"]] = (
            e4_snapshot_sha, e4_snapshot["created_utc"]
        )
    return all_units, final_counts, unit_by_id, authorization_by_unit


def validate_attempts_and_controls(
    records: dict[str, list[dict[str, Any]]],
    unit_by_id: dict[str, dict[str, Any]],
    final_counts: dict[str, int],
    authorization_by_unit: dict[str, tuple[str, str]],
    frozen_environment: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    attempts = records["attempts"]
    started = {row["attempt_id"]: row for row in attempts if row["record_type"] == "attempt_started"}
    finished = {row["attempt_id"]: row for row in attempts if row["record_type"] == "attempt_finished"}
    if len(started) + len(finished) != len(attempts):
        raise ContractError("duplicate attempt journal IDs or unknown attempt row type")
    if set(started) != set(finished):
        raise ContractError("attempt_started and attempt_finished IDs do not pair one-to-one")
    if len(started) != final_counts["attempts"] or len(attempts) != final_counts["attempt_rows"]:
        raise ContractError("attempt journal count differs from effective plan")
    planned_frequency = Counter(row["planned_unit_id"] for row in started.values())
    expected_frequency = Counter({unit_id: unit["child_attempts"] for unit_id, unit in unit_by_id.items()})
    if planned_frequency != expected_frequency:
        raise ContractError("attempts do not map one-to-one onto planned units")
    for attempt_id, row in started.items():
        unit = unit_by_id[row["planned_unit_id"]]
        if row["experiment_id"] != unit["experiment_id"]:
            raise ContractError(f"attempt {attempt_id} has wrong experiment")
        if row["planned_sample_count"] != unit["timing_rows"]:
            raise ContractError(f"attempt {attempt_id} planned sample count differs from unit")
        if row["planned_frame_count"] != unit["frames"]:
            raise ContractError(f"attempt {attempt_id} planned frame count differs from unit")
        if sha256_bytes(canonical_json_bytes(row["command"])) != row["command_sha256"]:
            raise ContractError(f"attempt {attempt_id} command digest does not resolve")
        if sha256_bytes(canonical_json_bytes(row["environment"])) != row["environment_sha256"]:
            raise ContractError(f"attempt {attempt_id} environment digest does not resolve")
        if row["environment"] != frozen_environment:
            raise ContractError(f"attempt {attempt_id} environment differs from frozen run environment")
        authorizing_sha256, authorizing_created_utc = authorization_by_unit[row["planned_unit_id"]]
        if row["effective_plan_sha256"] != authorizing_sha256:
            raise ContractError(f"attempt {attempt_id} cites the wrong phase authorizing plan")
        if datetime.fromisoformat(row["started_utc"].replace("Z", "+00:00")) < datetime.fromisoformat(
            authorizing_created_utc.replace("Z", "+00:00")
        ):
            raise ContractError(f"attempt {attempt_id} predates its authorizing plan")
        terminal = finished[attempt_id]
        if terminal["observed_frame_count"] != unit["frames"]:
            raise ContractError(f"attempt {attempt_id} frame count differs from plan")
        if terminal["observed_sample_count"] != unit["timing_rows"]:
            raise ContractError(f"attempt {attempt_id} sample count differs from plan")
        if terminal["terminal_status"] != "exited" or terminal.get("exit_code") != 0 or terminal["timed_out"]:
            raise ContractError(f"attempt {attempt_id} did not exit successfully")

    controls = records["controls"]
    require_unique(controls, "snapshot_id", "control snapshot ID")
    if len(controls) != final_counts["control_snapshots"]:
        raise ContractError("control snapshot count differs from effective plan")
    controls_by_attempt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    control_by_id = {row["snapshot_id"]: row for row in controls}
    for row in controls:
        controls_by_attempt[row["attempt_id"]].append(row)
    for attempt_id in started:
        phases = Counter(row["phase"] for row in controls_by_attempt[attempt_id])
        if phases != Counter({"pre": 1, "post": 1}):
            raise ContractError(f"attempt {attempt_id} does not have exactly pre/post controls")
        if started[attempt_id]["pre_control_snapshot_id"] not in control_by_id:
            raise ContractError(f"attempt {attempt_id} pre-control link is unresolved")
        if finished[attempt_id]["post_control_snapshot_id"] not in control_by_id:
            raise ContractError(f"attempt {attempt_id} post-control link is unresolved")
    return started, finished


def validate_samples_events_correctness(
    records: dict[str, list[dict[str, Any]]],
    started: dict[str, dict[str, Any]],
    unit_by_id: dict[str, dict[str, Any]],
    final_counts: dict[str, int],
    parameters: dict[str, Any],
) -> None:
    samples = records["samples"]
    events = records["events"]
    correctness = records["correctness"]
    require_unique(samples, "sample_id", "sample ID")
    require_unique(events, "event_record_id", "event record ID")
    require_unique(correctness, "correctness_record_id", "correctness record ID")
    if len(samples) != final_counts["timing_rows"]:
        raise ContractError("timing sample count differs from effective plan")
    if len(events) != final_counts["event_rows"]:
        raise ContractError("event count differs from effective plan")
    if len(correctness) != final_counts["correctness_rows"]:
        raise ContractError("correctness count differs from effective plan")

    event_by_id = {row["event_record_id"]: row for row in events}
    correctness_by_id = {row["correctness_record_id"]: row for row in correctness}
    samples_by_attempt = Counter(row["attempt_id"] for row in samples)
    correctness_by_attempt = Counter(row["attempt_id"] for row in correctness)
    events_by_attempt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frames_by_attempt: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event["attempt_id"] not in started:
            raise ContractError(f"event references unknown attempt {event['attempt_id']}")
        events_by_attempt[event["attempt_id"]].append(event)
        frames_by_attempt[event["attempt_id"]][event["frame_index"]].append(event)
        start_ns, end_ns = event["command_start_ns"], event["command_end_ns"]
        if start_ns is not None and end_ns is not None and end_ns < start_ns:
            raise ContractError(f"event {event['event_record_id']} has a negative interval")

    for attempt_id, start in started.items():
        unit = unit_by_id[start["planned_unit_id"]]
        attempt_events = events_by_attempt[attempt_id]
        if len(attempt_events) != unit["event_rows"]:
            raise ContractError(f"attempt {attempt_id} event count differs from planned unit")
        if samples_by_attempt[attempt_id] != unit["timing_rows"]:
            raise ContractError(f"attempt {attempt_id} enumerated timing rows differ from planned unit")
        if correctness_by_attempt[attempt_id] != unit["correctness_rows"]:
            raise ContractError(f"attempt {attempt_id} correctness rows differ from planned unit")
        if unit["backend"] == "fpga" and unit["frames"]:
            mode = unit["metadata"].get("mode")
            expected_per_frame = 51 if mode == "full_diagnostics" else 39 if mode == "one_ntt_save_stream" else 27
            if len(frames_by_attempt[attempt_id]) != unit["frames"]:
                raise ContractError(f"attempt {attempt_id} does not enumerate every FPGA frame")
            bad = [frame for frame, rows in frames_by_attempt[attempt_id].items() if len(rows) != expected_per_frame]
            if bad:
                raise ContractError(f"attempt {attempt_id} has wrong event cardinality for frames {bad[:8]}")
            input_bytes = (
                parameters["poly_modulus_degree"] // 4
            ) * parameters["pipeline_input_block_size"]
            output_bytes = parameters["poly_modulus_degree"] * parameters["uint32_size"]
            for frame_index, frame_events in frames_by_attempt[attempt_id].items():
                stage_counts = Counter(row["stage"] for row in frame_events)
                expected_stages = Counter({
                    "H2D": 1, "ENTRY": 1, "IFFT_FANOUT": 1,
                    "SCALE_REDUCE": 6, "POLY_MULT_NEG_ADD": 6,
                    "EXIT_C0": 6, "D2H": 6,
                })
                if mode == "full_diagnostics":
                    expected_stages.update({"EXIT_NTT_A": 6, "EXIT_NTT_B": 6, "D2H": 12})
                elif mode == "one_ntt_save_stream":
                    expected_stages.update({"EXIT_NTT_A": 6, "D2H": 6})
                if stage_counts != expected_stages:
                    raise ContractError(
                        f"attempt {attempt_id} frame {frame_index} has wrong stage multiset"
                    )
                transfer_counts = Counter(row["transfer_kind"] for row in frame_events)
                expected_transfers = Counter({"NONE": 20, "PACKED_INPUT": 1, "C0": 6})
                if mode == "full_diagnostics":
                    expected_transfers.update({"NONE": 12, "NTT_A": 6, "NTT_B": 6})
                elif mode == "one_ntt_save_stream":
                    expected_transfers.update({"NONE": 6, "NTT_A": 6})
                if transfer_counts != expected_transfers:
                    raise ContractError(
                        f"attempt {attempt_id} frame {frame_index} has wrong transfer multiset"
                    )
                for event in frame_events:
                    expected_bytes = input_bytes if event["transfer_kind"] == "PACKED_INPUT" else (
                        output_bytes if event["transfer_kind"] in {"C0", "NTT_A", "NTT_B"} else 0
                    )
                    if event["byte_count"] != expected_bytes:
                        raise ContractError(
                            f"event {event['event_record_id']} byte count differs from ABI formula"
                        )
                    if event["stage"] in {"H2D", "ENTRY", "IFFT_FANOUT"}:
                        if event["modulus_index"] != -1:
                            raise ContractError(
                                f"event {event['event_record_id']} must use modulus_index=-1"
                            )
                    elif event["modulus_index"] not in range(6):
                        raise ContractError(
                            f"event {event['event_record_id']} has invalid modulus index"
                        )
                modulus_stages = ["SCALE_REDUCE", "POLY_MULT_NEG_ADD", "EXIT_C0"]
                if mode in {"full_diagnostics", "one_ntt_save_stream"}:
                    modulus_stages.append("EXIT_NTT_A")
                if mode == "full_diagnostics":
                    modulus_stages.append("EXIT_NTT_B")
                for stage in modulus_stages:
                    indices = sorted(
                        row["modulus_index"] for row in frame_events if row["stage"] == stage
                    )
                    if indices != list(range(6)):
                        raise ContractError(
                            f"attempt {attempt_id} frame {frame_index} stage {stage} "
                            "does not cover each modulus exactly once"
                        )
                transfer_kinds = ["C0"]
                if mode in {"full_diagnostics", "one_ntt_save_stream"}:
                    transfer_kinds.append("NTT_A")
                if mode == "full_diagnostics":
                    transfer_kinds.append("NTT_B")
                for transfer_kind in transfer_kinds:
                    indices = sorted(
                        row["modulus_index"] for row in frame_events
                        if row["stage"] == "D2H" and row["transfer_kind"] == transfer_kind
                    )
                    if indices != list(range(6)):
                        raise ContractError(
                            f"attempt {attempt_id} frame {frame_index} transfer {transfer_kind} "
                            "does not cover each modulus exactly once"
                        )

    for sample in samples:
        if sample["attempt_id"] not in started:
            raise ContractError(f"sample {sample['sample_id']} references unknown attempt")
        if sample["status"] != "pass" or sample["correctness_summary"]["verified_after_timing"] is not True:
            raise ContractError(f"sample {sample['sample_id']} is not a post-timing correctness pass")
        if sample["frame_count_submitted"] != sample["frame_count_completed"]:
            raise ContractError(f"sample {sample['sample_id']} dropped or added frames")
        for field, value in sample["timing_ns"].items():
            if value is None and not sample["timing_unavailable_reasons"].get(field):
                raise ContractError(f"sample {sample['sample_id']} has null {field} without reason")
        timing = sample["timing_ns"]
        additive_fields = ("h2d_wall", "graph_submit_wait_wall", "d2h_wall")
        additive_available = (
            sample["backend"] == "fpga" and sample["batch_size"] == 1 and
            sample["experiment_id"] in {"E2", "E6"}
        )
        if sample["backend"] == "fpga":
            if any(field in sample["timing_unavailable_reasons"] for field in additive_fields):
                raise ContractError(f"sample {sample['sample_id']} suppresses an FPGA wall interval")
        else:
            for field in additive_fields:
                if timing[field] != 0 or not sample["timing_unavailable_reasons"].get(field):
                    raise ContractError(
                        f"sample {sample['sample_id']} must mark CPU-only {field} unavailable"
                    )
        attributed_fields = ["preparation", "pack", "unpack_and_assembly", "unattributed_wall"]
        if additive_available:
            attributed_fields.extend(additive_fields)
        attributed = sum(timing[field] for field in attributed_fields)
        if attributed != timing["application_e2e"]:
            raise ContractError(f"sample {sample['sample_id']} wall-clock breakdown does not reconcile")
        linked_event_rows = [event_by_id.get(identifier) for identifier in sample["event_record_ids"]]
        if any(event is None for event in linked_event_rows):
            raise ContractError(f"sample {sample['sample_id']} has unresolved event links")
        linked_events = [event for event in linked_event_rows if event is not None]
        expected_links = 0
        if sample["backend"] == "fpga":
            saves = int(sample["mode"]["save_ntt_s"]) + int(sample["mode"]["save_ntt_pte"])
            expected_links = sample["frame_count_submitted"] * (27 + 12 * saves)
        if len(linked_events) != expected_links:
            raise ContractError(f"sample {sample['sample_id']} has incomplete event links")
        if any(event["sample_id"] != sample["sample_id"] for event in linked_events):
            raise ContractError(f"sample {sample['sample_id']} links events owned by another phase")
        if sample["backend"] == "fpga":
            linked_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for event in linked_events:
                linked_by_frame[event["frame_index"]].append(event)
            frontiers = sample["event_frontiers"]
            if len(frontiers) != sample["frame_count_submitted"]:
                raise ContractError(f"sample {sample['sample_id']} has incomplete event frontiers")
            if {row["frame_index"] for row in frontiers} != set(linked_by_frame):
                raise ContractError(f"sample {sample['sample_id']} frontier frame indices do not resolve")
            for frontier in frontiers:
                frame_events = linked_by_frame[frontier["frame_index"]]
                bounded = {
                    "entry_end_ns": [row for row in frame_events if row["stage"] == "ENTRY"],
                    "fanout_end_ns": [row for row in frame_events if row["stage"] == "IFFT_FANOUT"],
                    "scale_end_ns": [row for row in frame_events if row["stage"] == "SCALE_REDUCE"],
                    "poly_end_ns": [row for row in frame_events if row["stage"] == "POLY_MULT_NEG_ADD"],
                    "exit_end_ns": [row for row in frame_events if row["stage"] == "EXIT_C0"],
                }
                available = all(
                    row["profiling_available"]
                    for rows in bounded.values() for row in rows
                )
                if frontier["profiling_available"] != available:
                    raise ContractError(f"sample {sample['sample_id']} frontier availability disagrees")
                if available:
                    if frontier["unavailable_reason"] is not None:
                        raise ContractError(f"sample {sample['sample_id']} available frontier has a reason")
                    for field, rows in bounded.items():
                        if frontier[field] != max(row["command_end_ns"] for row in rows):
                            raise ContractError(
                                f"sample {sample['sample_id']} {field} does not match linked events"
                            )
                else:
                    if not frontier["unavailable_reason"] or any(
                        frontier[field] is not None for field in bounded
                    ):
                        raise ContractError(f"sample {sample['sample_id']} unavailable frontier is malformed")
        elif sample["event_frontiers"]:
            raise ContractError(f"CPU sample {sample['sample_id']} must not claim FPGA event frontiers")
        linked_correctness = [correctness_by_id.get(identifier) for identifier in sample["correctness_record_ids"]]
        for linked_row in linked_correctness:
            if (linked_row is None or linked_row["passed"] is not True or
                    linked_row["sample_id"] != sample["sample_id"]):
                raise ContractError(
                    f"sample {sample['sample_id']} has unresolved or failing correctness links"
                )

    for row in correctness:
        if row["attempt_id"] not in started:
            raise ContractError(f"correctness row {row['correctness_record_id']} references unknown attempt")
        finite_fields = (
            "max_abs_error", "rms_error", "max_real_error", "max_imag_error",
            "component_max_abs_error", "threshold", "pairwise_max_abs_error",
            "pairwise_rms_error",
        )
        for field in finite_fields:
            value = row[field]
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ContractError(
                    f"correctness row {row['correctness_record_id']} has nonfinite {field}"
                )
        if row["passed"] is not True:
            raise ContractError(f"correctness row {row['correctness_record_id']} failed")
        if row["mismatch_count"] != 0 or row["nonfinite_value_count"] != 0:
            raise ContractError(
                f"passing correctness row {row['correctness_record_id']} reports failures"
            )
        if row["transport_mismatch_count"] != 0 or row["transport_passed"] is not True:
            raise ContractError(
                f"passing correctness row {row['correctness_record_id']} failed C0 transport"
            )
        if row["transport_mismatch_count"] != (
            row["noncanonical_residue_count"] + row["retained_c1_mismatch_count"]
        ):
            raise ContractError(
                f"correctness row {row['correctness_record_id']} has inconsistent C0 counts"
            )
        if row["finite_value_count"] != row["compared_value_count"]:
            raise ContractError(
                f"passing correctness row {row['correctness_record_id']} has incomplete finite coverage"
            )
        if row["max_abs_error"] is None or row["max_abs_error"] > row["threshold"]:
            raise ContractError(
                f"passing correctness row {row['correctness_record_id']} exceeds its threshold"
            )
        if row["verification_kind"] == "decrypt_decode":
            if row["requested_slot_count"] + row["inactive_slot_count"] != 4096:
                raise ContractError(
                    f"semantic row {row['correctness_record_id']} does not cover all CKKS slots"
                )
            if row["compared_value_count"] != 4096:
                raise ContractError(
                    f"semantic row {row['correctness_record_id']} has wrong comparison count"
                )

    e1 = [row for row in correctness if row["experiment_id"] == "E1"]
    c1 = [row for row in e1 if row["check_id"] == "C1"]
    c2 = [row for row in e1 if row["check_id"] == "C2"]
    fpga_test = [row for row in e1 if row["check_id"] == "FPGA-Test"]
    c3 = [row for row in e1 if row["check_id"] == "C3"]
    patterns = [
        "ntt_sparse_impulse", "ntt_alternating_ternary",
        "ntt_negative_boundary", "ntt_shake256",
    ]
    expected_c1 = Counter(
        (pattern, selector, role)
        for pattern in patterns for selector in range(6) for role in ("NTT-A", "NTT-B")
    )
    if Counter((row["pattern"], row["selector"], row["role"]) for row in c1) != expected_c1:
        raise ContractError("E1 C1 does not cover exactly four patterns, six selectors, and two roles")
    expected_c2 = Counter((pattern, selector) for pattern in patterns for selector in range(6))
    if Counter((row["pattern"], row["selector"]) for row in c2) != expected_c2:
        raise ContractError("E1 C2 does not cover exactly four patterns and six selectors")
    if Counter(row["case_id"] for row in fpga_test) != Counter(REAL_CASES):
        raise ContractError("E1 complete FPGA Test does not cover exactly the five real cases")
    if Counter(row["case_id"] for row in c3) != Counter(ALL_CASES):
        raise ContractError("E1 C3 does not cover exactly all eight semantic cases")

    e5_fpga = [row for row in correctness if row["experiment_id"] == "E5" and row["backend"] == "fpga"]
    if len(e5_fpga) != 40:
        raise ContractError("E5 must contain exactly 40 FPGA correctness rows")
    e5 = [row for row in correctness if row["experiment_id"] == "E5"]
    expected_e5 = Counter(
        [("fpga", case_id, seed) for case_id in ALL_CASES for seed in range(5)]
        + [("seal-embedded", case_id, seed) for case_id in REAL_CASES for seed in range(5)]
        + [("stock-seal-reference", case_id, seed) for case_id in COMPLEX_CASES for seed in range(5)]
    )
    if Counter((row["backend"], row["case_id"], row["trial_seed_index"]) for row in e5) != expected_e5:
        raise ContractError("E5 backend/case/trial-seed matrix is not the exact 40/25/15 split")
    reverse_links: Counter[str] = Counter()
    for row in e5_fpga:
        reference_id = row["paired_reference_correctness_record_id"]
        reference = correctness_by_id.get(reference_id)
        if reference is None:
            raise ContractError(f"E5 FPGA row {row['correctness_record_id']} has no paired reference")
        if reference["case_id"] != row["case_id"] or reference["trial_seed_index"] != row["trial_seed_index"]:
            raise ContractError(f"E5 FPGA row {row['correctness_record_id']} reference is not case/seed matched")
        if reference["paired_reference_correctness_record_id"] != row["correctness_record_id"]:
            raise ContractError(f"E5 reference {reference_id} does not link back to its FPGA row")
        if (reference["pairwise_max_abs_error"] != row["pairwise_max_abs_error"] or
                reference["pairwise_rms_error"] != row["pairwise_rms_error"]):
            raise ContractError(f"E5 pair {row['correctness_record_id']}/{reference_id} metrics disagree")
        reverse_links[reference_id] += 1
    if any(count != 1 for count in reverse_links.values()) or len(reverse_links) != 40:
        raise ContractError("E5 paired-reference links are not one-to-one")


def validate_statuses(
    statuses: list[dict[str, Any]],
    effective_sha256: str,
    units: list[dict[str, Any]],
    raw: dict[str, list[dict[str, Any]]],
) -> None:
    require_unique(statuses, "experiment_id", "experiment-status experiment")
    if {row["experiment_id"] for row in statuses} != {f"E{index}" for index in range(1, 9)}:
        raise ContractError("experiment-status rows are not exactly E1 through E8")
    for row in statuses:
        if row["status"] != "pass":
            raise ContractError(f"{row['experiment_id']} terminal status is not pass")
        if row["effective_plan_sha256"] != effective_sha256:
            raise ContractError(f"{row['experiment_id']} cites the wrong effective plan")
        experiment_id = row["experiment_id"]
        experiment_units = [unit for unit in units if unit["experiment_id"] == experiment_id]
        expected = {
            "attempts": sum(int(unit["child_attempts"]) for unit in experiment_units),
            "samples": sum(int(unit["timing_rows"]) for unit in experiment_units),
            "frames": sum(int(unit["frames"]) for unit in experiment_units),
            "events": sum(int(unit["event_rows"]) for unit in experiment_units),
            "correctness": sum(int(unit["correctness_rows"]) for unit in experiment_units),
        }
        finished_attempts = {
            item["attempt_id"] for item in raw["attempts"]
            if item.get("experiment_id") == experiment_id and
            item.get("record_type") == "attempt_finished"
        }
        experiment_events = [
            item for item in raw["events"] if item.get("experiment_id") == experiment_id
        ]
        experiment_correctness = [
            item for item in raw["correctness"] if item.get("experiment_id") == experiment_id
        ]
        frame_keys = {
            (item["attempt_id"], item["frame_index"])
            for item in (experiment_events or experiment_correctness)
        }
        observed = {
            "attempts": len(finished_attempts),
            "samples": sum(
                item.get("experiment_id") == experiment_id for item in raw["samples"]
            ),
            "frames": len(frame_keys),
            "events": len(experiment_events),
            "correctness": len(experiment_correctness),
        }
        if observed != expected or row["validated_counts"] != observed:
            raise ContractError(
                f"{experiment_id} status counts do not close: "
                f"expected {expected}, observed {observed}, declared {row['validated_counts']}"
            )
    e3 = next(row for row in statuses if row["experiment_id"] == "E3")
    report_hashes = {
        e3["report_evidence"]["report_tree_sha256"],
        e3["report_evidence"]["info_ndjson_sha256"],
        e3["report_evidence"]["quartus_ndjson_sha256"],
        e3["report_evidence"]["device_image_sha256"],
    }
    if not report_hashes.issubset(set(e3["evidence_sha256s"])):
        raise ContractError("E3 detailed report hashes are absent from evidence_sha256s")
    e7 = next(row for row in statuses if row["experiment_id"] == "E7")
    if not e7.get("source_evidence"):
        raise ContractError("E7 status lacks source_evidence")
    child_output_path = require_absolute(e7["child_output_path"], "E7 child output")
    if sha256_file(child_output_path) != e7["child_output_sha256"]:
        raise ContractError("E7 child-output hash does not resolve")
    if e7["child_output_sha256"] not in e7["evidence_sha256s"]:
        raise ContractError("E7 child-output hash is absent from evidence_sha256s")
    for evidence in e7["source_evidence"]:
        source_path = require_absolute(evidence["source_path"], "E7 source")
        if sha256_file(source_path) != evidence["source_sha256"]:
            raise ContractError(f"E7 source hash does not resolve: {source_path}")


def main() -> int:
    args = parse_args()
    run_root = require_absolute(args.run_root.resolve(), "run root")
    harness_root = Path(__file__).resolve().parent.parent
    schema_root = harness_root / "schemas"
    manifest_schema = load_json(schema_root / "manifest.schema.json")
    validators = {
        "manifest": schema_validator(manifest_schema, "root_manifest"),
        "attempts": schema_validator(load_json(schema_root / "attempt.schema.json")),
        "controls": schema_validator(load_json(schema_root / "control.schema.json")),
        "samples": schema_validator(load_json(schema_root / "sample.schema.json")),
        "events": schema_validator(load_json(schema_root / "event.schema.json")),
        "correctness": schema_validator(load_json(schema_root / "correctness.schema.json")),
        "experiment_status": schema_validator(load_json(schema_root / "experiment-status.schema.json")),
    }
    manifest = load_and_validate_json(run_root / "manifest.json", validators["manifest"], "manifest")
    if require_absolute(manifest["paths"]["run_root"]) != run_root:
        raise ContractError("manifest run_root differs from requested result directory")
    if manifest["paper_eligible"] is not True:
        raise ContractError("run is permanently marked non-paper")
    validate_artifacts(manifest)
    all_units, final_counts, unit_by_id, authorization_by_unit = validate_planning(
        manifest, manifest_schema
    )
    final_effective_path = require_absolute(manifest["paths"]["effective_plan_e4"])
    final_effective_sha256 = sha256_file(final_effective_path)

    records: dict[str, list[dict[str, Any]]] = {}
    for key, path_key in (
        ("attempts", "attempts"),
        ("controls", "controls"),
        ("samples", "samples"),
        ("events", "events"),
        ("correctness", "correctness"),
        ("experiment_status", "experiment_status"),
    ):
        records[key] = load_and_validate_jsonl(
            require_absolute(manifest["paths"][path_key]), validators[key], key
        )
    started, _ = validate_attempts_and_controls(
        records,
        unit_by_id,
        final_counts,
        authorization_by_unit,
        load_json(require_absolute(manifest["paths"]["environment"])),
    )
    validate_samples_events_correctness(
        records, started, unit_by_id, final_counts, manifest["parameters"]
    )
    validate_statuses(
        records["experiment_status"], final_effective_sha256, all_units, records
    )
    print(
        f"PASS: {len(all_units)} planned units, {final_counts['attempts']} attempts, "
        f"{final_counts['frames']} frames, {final_counts['event_rows']} events, "
        f"{final_counts['correctness_rows']} correctness rows, "
        f"{final_counts['timing_rows']} timing rows"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"validate_results: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
