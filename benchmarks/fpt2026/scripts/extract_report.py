#!/usr/bin/env python3
"""Extract E3 frequency, resources, provenance, and source association from NDJSON."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from common import (
    ContractError,
    load_json,
    read_jsonl,
    require_absolute,
    sha256_bytes,
    sha256_file,
    sha256_tree,
    write_new_json,
)

EXPECTED_SEED = 7
EXPECTED_CLOCK_MHZ = 557.00
EXPECTED_TARGET = "AGFB027R25A2E2V"
REQUIRED_REPORT_FILES = ("info.ndjson", "quartus.ndjson", "file.ndjson", "summary.ndjson")
DEVICE_SOURCE_PREFIXES = ("src/", "include/", "include_internal/", "rtl/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pre-control-snapshot-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _artifact(manifest: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    matches = [item for item in manifest.get("artifacts", []) if item.get("artifact_id") == artifact_id]
    if len(matches) != 1:
        raise ContractError(f"manifest must contain exactly one {artifact_id!r} artifact")
    return matches[0]


def _machine_json_root(report_root: Path) -> Path:
    candidates = (report_root / "resources" / "json", report_root / "json", report_root)
    for candidate in candidates:
        if all((candidate / name).is_file() for name in REQUIRED_REPORT_FILES):
            return candidate
    raise ContractError(
        f"report root lacks required machine NDJSON files {REQUIRED_REPORT_FILES}: {report_root}"
    )


def _load_ndjson(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ContractError(f"empty machine report: {path}")
    return rows


def _nodes(document: dict[str, Any], key: str) -> list[dict[str, Any]]:
    container = document.get(key)
    if not isinstance(container, dict) or not isinstance(container.get("nodes"), list):
        raise ContractError(f"missing {key}.nodes[] in machine report")
    rows = container["nodes"]
    if not all(isinstance(row, dict) for row in rows):
        raise ContractError(f"non-object row in {key}.nodes[]")
    return rows


def _only(rows: Iterable[dict[str, Any]], label: str) -> dict[str, Any]:
    materialized = list(rows)
    if len(materialized) != 1:
        raise ContractError(f"expected exactly one {label}, observed {len(materialized)}")
    return materialized[0]


def _number(value: Any, label: str) -> int | float:
    if isinstance(value, bool):
        raise ContractError(f"{label} is boolean, not numeric")
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        raise ContractError(f"{label} is not numeric: {value!r}")
    cleaned = value.replace(",", "").strip()
    try:
        return float(cleaned) if "." in cleaned else int(cleaned)
    except ValueError as exc:
        raise ContractError(f"{label} is not numeric: {value!r}") from exc


def _seed(command: str) -> int:
    matches = re.findall(r"(?:^|\s)-seed(?:=|\s+)(\d+)(?=\s|$)", command)
    if len(matches) != 1:
        raise ContractError(f"backend command must contain exactly one explicit fitter seed: {command!r}")
    return int(matches[0])


def _target(family: str, command: str) -> str:
    if EXPECTED_TARGET in family:
        return EXPECTED_TARGET
    target_match = re.search(r"(?:^|\s)-target(?:=|\s+)(\S+)", command)
    if target_match and EXPECTED_TARGET in target_match.group(1):
        return EXPECTED_TARGET
    raise ContractError(f"target {EXPECTED_TARGET} is absent from report family/command")


def _resource_row(row: dict[str, Any]) -> dict[str, int | float]:
    aliases = {"alut": "alut", "reg": "register", "alm": "alm", "dsp": "dsp", "ram": "ram"}
    result: dict[str, int | float] = {}
    for source_key, output_key in aliases.items():
        if source_key not in row:
            raise ContractError(f"resource row {row.get('name')!r} lacks {source_key}")
        result[output_key] = _number(row[source_key], f"{row.get('name')}.{source_key}")
    if "mlab" in row:
        result["mlab"] = _number(row["mlab"], f"{row.get('name')}.mlab")
    return result


def _git_blob(repository: Path, commit: str, relative_path: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ContractError(f"invalid pinned accelerator commit: {commit!r}")
    process = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        diagnostic = process.stderr.decode("utf-8", errors="replace").strip()
        raise ContractError(f"cannot read pinned source {relative_path} at {commit}: {diagnostic}")
    return process.stdout


def _report_relative_path(original: str) -> str | None:
    marker = "SYCL_CKKS_PIPELINE/"
    if marker not in original:
        return None
    relative = original.rsplit(marker, 1)[1]
    return relative if relative.startswith(DEVICE_SOURCE_PREFIXES) else None


def compare_embedded_sources(
    file_rows: list[dict[str, Any]], repository: Path, commit: str
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(file_rows, start=1):
        original = row.get("path") or row.get("absName")
        content = row.get("content")
        if not isinstance(original, str) or not isinstance(content, str):
            raise ContractError(f"file.ndjson row {row_number} lacks path/content")
        relative = _report_relative_path(original)
        if relative is None:
            continue
        if relative in seen:
            raise ContractError(f"duplicate embedded device source: {relative}")
        seen.add(relative)
        embedded = content.encode("utf-8")
        pinned = _git_blob(repository, commit, relative)
        comparisons.append(
            {
                "relative_path": relative,
                "original_report_path": original,
                "normalized_path": str(repository / relative),
                "report_row": row_number,
                "embedded_sha256": sha256_bytes(embedded),
                "pinned_sha256": sha256_bytes(pinned),
                "byte_count": len(embedded),
                "match": embedded == pinned,
            }
        )
    if not comparisons:
        raise ContractError("file.ndjson contains no accelerator device sources")
    mismatches = [item["relative_path"] for item in comparisons if not item["match"]]
    if mismatches:
        raise ContractError(f"report device sources differ from the pinned commit: {mismatches}")
    return comparisons


def _pre_control(manifest: dict[str, Any], snapshot_id: str) -> dict[str, Any]:
    controls_path = require_absolute(manifest["paths"]["controls"], "controls JSONL")
    matches = [row for row in read_jsonl(controls_path) if row.get("snapshot_id") == snapshot_id]
    control = _only(matches, f"pre-control snapshot {snapshot_id}")
    if control.get("phase") != "pre":
        raise ContractError("E3 control snapshot is not a pre-attempt observation")
    return control


def _association(manifest: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    artifacts = {
        artifact_id: _artifact(manifest, artifact_id)
        for artifact_id in ("report_root", "accelerator_archive", "gbs", "executable")
    }
    fpga = control.get("fpga")
    if not isinstance(fpga, dict):
        raise ContractError("E3 pre-control has no FPGA payload")
    programmed_hash = fpga.get("programmed_image_sha256")
    if programmed_hash != artifacts["gbs"]["sha256"]:
        raise ContractError("programmed image identity does not equal the manifest-pinned GBS")
    model = fpga.get("board_model")
    revision = fpga.get("board_revision")
    evidence_source = fpga.get("identity_evidence_source")
    if not all(isinstance(value, str) and value for value in (model, revision, evidence_source)):
        raise ContractError("E3 pre-control lacks board model, revision, or identity evidence source")
    return {
        "association_id": f"{manifest['run_id']}:seed7-report-image",
        "report_root": {"artifact_id": "report_root", "sha256": artifacts["report_root"]["sha256"]},
        "accelerator_archive": {
            "artifact_id": "accelerator_archive",
            "sha256": artifacts["accelerator_archive"]["sha256"],
        },
        "gbs": {"artifact_id": "gbs", "sha256": artifacts["gbs"]["sha256"]},
        "executable": {"artifact_id": "executable", "sha256": artifacts["executable"]["sha256"]},
        "programmed_image_sha256": programmed_hash,
        "board": {"model": model, "revision": revision, "identity_evidence_source": evidence_source},
        "resolved": True,
    }


def extract(manifest: dict[str, Any], pre_control_snapshot_id: str) -> dict[str, Any]:
    report_artifact = _artifact(manifest, "report_root")
    report_root = require_absolute(report_artifact["path"], "report root")
    if not report_root.is_dir():
        raise ContractError(f"report root is not a directory: {report_root}")
    observed_tree_hash, _ = sha256_tree(report_root)
    if observed_tree_hash != report_artifact["sha256"]:
        raise ContractError("report tree bytes differ from the frozen manifest")
    json_root = _machine_json_root(report_root)
    paths = {name: json_root / name for name in REQUIRED_REPORT_FILES}

    info_rows = _load_ndjson(paths["info.ndjson"])
    quartus_rows = _load_ndjson(paths["quartus.ndjson"])
    file_rows = _load_ndjson(paths["file.ndjson"])
    summary_rows = _load_ndjson(paths["summary.ndjson"])
    compile_node = _only(
        (node for row in info_rows for node in _nodes(row, "compileInfo")), "compileInfo node"
    )
    clock_nodes = [node for row in quartus_rows for node in _nodes(row, "quartusFitClockSummary")]
    clock_node = _only(
        (node for node in clock_nodes if node.get("name") == "Quartus Fitter: Clock Frequency (MHz)"),
        "Quartus clock summary",
    )
    resource_nodes = [
        node for row in quartus_rows for node in _nodes(row, "quartusFitResourceUsageSummary")
    ]
    entire = _only(
        (node for node in resource_nodes if node.get("name") == "Quartus Fitter: Total Used (Entire System)"),
        "entire-system resource row",
    )
    kernel_system = _only(
        (node for node in resource_nodes if node.get("name") == "Quartus Fitter: Kernel System"),
        "kernel-system resource row",
    )

    command = compile_node.get("command")
    family = compile_node.get("family")
    if not isinstance(command, str) or not isinstance(family, str):
        raise ContractError("compile node lacks command/family strings")
    seed = _seed(command)
    target = _target(family, command)
    actual_clock = float(_number(clock_node.get("kernel clock"), "actual kernel clock"))
    maximum_clock = float(_number(clock_node.get("kernel clock fmax"), "maximum kernel clock"))
    if seed != EXPECTED_SEED:
        raise ContractError(f"expected seed {EXPECTED_SEED}, observed {seed}")
    if target != EXPECTED_TARGET:
        raise ContractError(f"expected target {EXPECTED_TARGET}, observed {target}")
    if actual_clock != EXPECTED_CLOCK_MHZ:
        raise ContractError(f"expected actual clock {EXPECTED_CLOCK_MHZ:.2f} MHz, observed {actual_clock:.2f}")

    configuration_path = require_absolute(manifest["paths"]["configuration"], "configuration")
    configuration = load_json(configuration_path)
    if sha256_file(configuration_path) != manifest["configuration_sha256"]:
        raise ContractError("configuration bytes differ from the manifest hash")
    expected_toolchain = configuration["toolchain"]
    backend_version = compile_node.get("version")
    quartus_version = compile_node.get("quartus")
    if not isinstance(backend_version, str) or not backend_version.startswith(expected_toolchain["fpga_backend_version"]):
        raise ContractError("FPGA backend version differs from the frozen configuration")
    if quartus_version != expected_toolchain["quartus_version"]:
        raise ContractError("Quartus version differs from the frozen configuration")

    repository = require_absolute(manifest["repository_state"]["accelerator"]["path"], "accelerator repository")
    pinned_commit = manifest["pinned_revisions"]["accelerator_baseline_commit"]
    source_comparisons = compare_embedded_sources(file_rows, repository, pinned_commit)
    control = _pre_control(manifest, pre_control_snapshot_id)
    association = _association(manifest, control)

    kernel_inventory = [
        {
            "name": row.get("name"),
            "compiler_name": row.get("compiler_name"),
            "kernel_type": row.get("data", [None])[0] if isinstance(row.get("data"), list) and row["data"] else None,
        }
        for row in summary_rows
        if row.get("parent") == "performanceSummary" and isinstance(row.get("compiler_name"), str)
    ]
    if not kernel_inventory:
        raise ContractError("summary.ndjson contains no machine-readable kernel inventory")

    entire_resources = _resource_row(entire)
    report_evidence = {
        "report_root_path": str(report_root),
        "report_tree_sha256": observed_tree_hash,
        "info_ndjson_path": str(paths["info.ndjson"]),
        "info_ndjson_sha256": sha256_file(paths["info.ndjson"]),
        "quartus_ndjson_path": str(paths["quartus.ndjson"]),
        "quartus_ndjson_sha256": sha256_file(paths["quartus.ndjson"]),
        "source_revision": pinned_commit,
        "device_image_sha256": association["programmed_image_sha256"],
        "fpga_target": target,
        "fitter_seed": seed,
        "kernel_clock_mhz": actual_clock,
        "resources": {
            "alut": int(entire_resources["alut"]),
            "registers": int(entire_resources["register"]),
            "alm": int(entire_resources["alm"]),
            "dsp": int(entire_resources["dsp"]),
            "ram": int(entire_resources["ram"]),
        },
    }
    source_evidence = [
        {
            "source_path": str(paths["info.ndjson"]),
            "source_sha256": sha256_file(paths["info.ndjson"]),
            "locator": "NDJSON object 1: compileInfo.nodes[0]",
            "payload": {
                "build_time": compile_node.get("time"),
                "backend_version": backend_version,
                "quartus_version": quartus_version,
                "family": family,
                "backend_command": command,
                "fitter_seed": seed,
            },
        },
        {
            "source_path": str(paths["quartus.ndjson"]),
            "source_sha256": sha256_file(paths["quartus.ndjson"]),
            "locator": "NDJSON object 1: quartusFitClockSummary/quartusFitResourceUsageSummary",
            "payload": {
                "actual_kernel_clock_mhz": actual_clock,
                "maximum_kernel_clock_mhz": maximum_clock,
                "entire_system": _resource_row(entire),
                "kernel_system": _resource_row(kernel_system),
            },
        },
        {
            "source_path": str(paths["file.ndjson"]),
            "source_sha256": sha256_file(paths["file.ndjson"]),
            "locator": "all accelerator device-source rows after absolute-prefix normalization",
            "payload": {"pinned_commit": pinned_commit, "comparisons": source_comparisons},
        },
        {
            "source_path": str(paths["summary.ndjson"]),
            "source_sha256": sha256_file(paths["summary.ndjson"]),
            "locator": "rows with parent=performanceSummary",
            "payload": {"kernel_inventory": kernel_inventory},
        },
        {
            "source_path": str(configuration_path),
            "source_sha256": sha256_file(configuration_path),
            "locator": "toolchain.seed7_host_compiler_version",
            "payload": {"host_compiler_version": expected_toolchain["seed7_host_compiler_version"]},
        },
    ]
    return {
        "schema_version": "1.0",
        "record_type": "report_extraction",
        "run_id": manifest["run_id"],
        "experiment_id": "E3",
        "report_evidence": report_evidence,
        "report": {
            "build_time": compile_node.get("time"),
            "source_build_path": compile_node.get("reports_directory"),
            "fpga_backend_version": backend_version,
            "host_compiler_version": expected_toolchain["seed7_host_compiler_version"],
            "quartus_version": quartus_version,
            "target_family": family,
            "target_part": target,
            "backend_command": command,
            "fitter_seed": seed,
            "actual_kernel_clock_mhz": actual_clock,
            "maximum_kernel_clock_mhz": maximum_clock,
            "entire_system_resources": entire_resources,
            "kernel_system_resources": _resource_row(kernel_system),
            "kernel_resources": [
                {"name": node.get("name"), "type": node.get("type"), "resources": _resource_row(node)}
                for node in resource_nodes
                if node.get("type") == "kernel"
            ],
            "kernel_inventory": kernel_inventory,
        },
        "association": association,
        "source_evidence": source_evidence,
    }


def main() -> int:
    args = parse_args()
    manifest_path = require_absolute(args.manifest.resolve(), "manifest")
    output_path = require_absolute(args.output.resolve(), "output")
    manifest = load_json(manifest_path)
    write_new_json(output_path, extract(manifest, args.pre_control_snapshot_id))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ContractError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as error:
        print(f"extract_report: {error}", file=sys.stderr)
        raise SystemExit(2)
