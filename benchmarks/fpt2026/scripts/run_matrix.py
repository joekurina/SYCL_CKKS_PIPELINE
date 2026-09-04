#!/usr/bin/env python3
"""Run the immutable FPT 2026 matrix with external child watchdogs.

This module is orchestration only.  It launches an already-frozen benchmark
executable or one of the two read-only extraction children; it contains no build,
synthesis, link, FPGA-programming, or benchmark implementation path.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from common import (
    ContractError,
    append_jsonl,
    canonical_json_bytes,
    load_json,
    read_jsonl,
    require_absolute,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_new_json,
)
from finalize_plans import finalize_e4, finalize_e8
from summarize_results import e8_evidence_document

SCRIPT_ROOT = Path(__file__).resolve().parent
EXTRACT_REPORT = SCRIPT_ROOT / "extract_report.py"
EXTRACT_PRIOR_WORK = SCRIPT_ROOT / "extract_prior_work.py"
FORBIDDEN_COMMAND_FRAGMENTS = (
    "-xsprofile",
    "-reuse-exe",
    "-fsycl-link",
    "quartus_sh",
    "quartus_fit",
    "quartus_asm",
    "quartus_pgm",
    "aocl program",
    "jtagconfig",
)
@dataclass(frozen=True)
class Authorization:
    phase: str
    snapshot_path: Path
    snapshot_sha256: str
    snapshot: dict[str, Any]
    units: tuple[dict[str, Any], ...]
    component_paths: tuple[Path, ...]


@dataclass(frozen=True)
class AttemptOutcome:
    attempt_id: str
    terminal_status: str
    exit_code: int | None
    signal_number: int | None
    timed_out: bool
    observed_samples: int
    observed_frames: int

    @property
    def passed(self) -> bool:
        return self.terminal_status == "exited" and self.exit_code == 0 and not self.timed_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--terminate-grace-seconds",
        type=float,
        default=5.0,
        help="fixed grace after watchdog SIGTERM before SIGKILL",
    )
    return parser.parse_args()


def _artifact(manifest: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    matches = [item for item in manifest.get("artifacts", []) if item.get("artifact_id") == artifact_id]
    if len(matches) != 1:
        raise ContractError(f"manifest must contain exactly one {artifact_id!r} artifact")
    return matches[0]


def _fsync_existing(path: Path) -> None:
    if not path.is_file():
        raise ContractError(f"authorizing evidence is missing: {path}")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _document_units(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = load_json(path)
    units = document.get("planned_units")
    if not isinstance(units, list) or not all(isinstance(unit, dict) for unit in units):
        raise ContractError(f"invalid planned-unit document: {path}")
    identifiers = [unit.get("planned_unit_id") for unit in units]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ContractError(f"planned unit lacks an ID: {path}")
    if len(set(identifiers)) != len(identifiers):
        raise ContractError(f"duplicate planned-unit ID in {path}")
    return document, units


def _authorize(
    manifest: dict[str, Any],
    *,
    phase: str,
    snapshot_key: str,
    unit_path_keys: Iterable[str],
    amendment_path_keys: Iterable[str],
    predecessor_key: str | None,
) -> Authorization:
    paths = manifest["paths"]
    snapshot_path = require_absolute(paths[snapshot_key], f"{phase} effective plan")
    _fsync_existing(snapshot_path)
    snapshot = load_json(snapshot_path)
    snapshot_sha256 = sha256_file(snapshot_path)
    if snapshot.get("document_type") != "effective_plan_snapshot":
        raise ContractError(f"{phase} authorizer is not an effective-plan snapshot")
    if predecessor_key is None:
        if snapshot.get("predecessor_sha256") is not None:
            raise ContractError("base effective plan unexpectedly has a predecessor")
        if snapshot_sha256 != manifest["base_effective_plan_sha256"]:
            raise ContractError("base effective-plan hash differs from the immutable manifest")
    else:
        predecessor_path = require_absolute(paths[predecessor_key], f"{phase} predecessor")
        _fsync_existing(predecessor_path)
        if snapshot.get("predecessor_sha256") != sha256_file(predecessor_path):
            raise ContractError(f"{phase} effective plan has a broken predecessor hash")

    units: list[dict[str, Any]] = []
    components: list[Path] = []
    component_hashes: list[str] = []
    for path_key in unit_path_keys:
        path = require_absolute(paths[path_key], f"{phase} planned units")
        _fsync_existing(path)
        _, document_units = _document_units(path)
        units.extend(document_units)
        components.append(path)
        component_hashes.append(sha256_file(path))
    for path_key in amendment_path_keys:
        path = require_absolute(paths[path_key], f"{phase} amendment")
        _fsync_existing(path)
        components.append(path)
        component_hashes.append(sha256_file(path))
    declared_components = snapshot.get("component_sha256s")
    if not isinstance(declared_components, list) or not all(
        digest in declared_components for digest in component_hashes
    ):
        raise ContractError(f"{phase} effective plan does not hash every referenced component")
    planned_ids = [unit["planned_unit_id"] for unit in units]
    if snapshot.get("planned_unit_ids") != planned_ids:
        raise ContractError(f"{phase} effective plan does not authorize the exact ordered units")
    if len(set(planned_ids)) != len(planned_ids):
        raise ContractError(f"{phase} effective plan repeats a planned-unit ID")
    return Authorization(
        phase=phase,
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot_sha256,
        snapshot=snapshot,
        units=tuple(units),
        component_paths=tuple(components),
    )


def authorize_base(manifest: dict[str, Any]) -> Authorization:
    authorization = _authorize(
        manifest,
        phase="base",
        snapshot_key="effective_plan_base",
        unit_path_keys=("planned_units_base",),
        amendment_path_keys=(),
        predecessor_key=None,
    )
    if sha256_file(authorization.component_paths[0]) != manifest["base_plan_sha256"]:
        raise ContractError("base unit-array hash differs from the manifest")
    return authorization


def authorize_e8(manifest: dict[str, Any]) -> Authorization:
    return _authorize(
        manifest,
        phase="base+e8",
        snapshot_key="effective_plan_e8",
        unit_path_keys=("planned_units_base", "planned_units_e8_extension"),
        amendment_path_keys=("manifest_amendment_e8",),
        predecessor_key="effective_plan_base",
    )


def authorize_e4(manifest: dict[str, Any]) -> Authorization:
    return _authorize(
        manifest,
        phase="base+e8+e4",
        snapshot_key="effective_plan_e4",
        unit_path_keys=(
            "planned_units_base",
            "planned_units_e8_extension",
            "planned_units_e4_sustained",
        ),
        amendment_path_keys=("manifest_amendment_e8", "manifest_amendment_e4"),
        predecessor_key="effective_plan_e8",
    )


def require_authorized_unit(authorization: Authorization, unit: dict[str, Any]) -> None:
    matches = [candidate for candidate in authorization.units if candidate.get("planned_unit_id") == unit.get("planned_unit_id")]
    if len(matches) != 1 or matches[0] != unit:
        raise ContractError(
            f"unit {unit.get('planned_unit_id')!r} is not byte-for-byte authorized by "
            f"effective plan {authorization.snapshot_sha256}"
        )
    _fsync_existing(authorization.snapshot_path)
    if sha256_file(authorization.snapshot_path) != authorization.snapshot_sha256:
        raise ContractError("effective plan changed after authorization")


def _driver_path(manifest: dict[str, Any]) -> Path:
    artifact = _artifact(manifest, "executable")
    path = require_absolute(artifact["path"], "benchmark executable")
    if not path.is_file() or sha256_file(path) != artifact["sha256"]:
        raise ContractError("benchmark executable is missing or differs from the manifest hash")
    return path


def _common_driver_command(driver: Path, subcommand: str, manifest_path: Path, run_root: Path) -> list[str]:
    return [
        str(driver),
        subcommand,
        "--profile",
        "paper",
        "--manifest",
        str(manifest_path),
        "--output",
        str(run_root),
    ]


def command_for_unit(
    manifest: dict[str, Any], manifest_path: Path, unit: dict[str, Any], attempt_id: str
) -> tuple[list[str], Path, Path | None]:
    run_root = require_absolute(manifest["paths"]["run_root"], "run root")
    unit_id = unit["planned_unit_id"]
    extraction_output: Path | None = None
    if unit_id == "E3-REPORT":
        extraction_output = run_root / "raw" / "extractions" / f"{attempt_id}.json"
        command = [
            sys.executable,
            str(EXTRACT_REPORT),
            "--manifest",
            str(manifest_path),
            "--pre-control-snapshot-id",
            f"{attempt_id}:control-pre",
            "--output",
            str(extraction_output),
        ]
        return command, SCRIPT_ROOT, extraction_output
    if unit_id == "E7-PRIOR-WORK":
        extraction_output = run_root / "raw" / "extractions" / f"{attempt_id}.json"
        command = [
            sys.executable,
            str(EXTRACT_PRIOR_WORK),
            "--manifest",
            str(manifest_path),
            "--output",
            str(extraction_output),
        ]
        return command, SCRIPT_ROOT, extraction_output

    driver = _driver_path(manifest)
    metadata = unit.get("metadata", {})
    experiment = unit["experiment_id"]
    backend = unit["backend"]
    if unit_id == "E1-C1-C2":
        command = _common_driver_command(driver, "correctness", manifest_path, run_root) + [
            "--backend", "fpga", "--suite", "c1-c2", "--save-mode", "full"
        ]
    elif unit_id == "E1-FPGA-TEST":
        command = _common_driver_command(driver, "correctness", manifest_path, run_root) + [
            "--backend", "fpga", "--suite", "fpga-test", "--save-mode", "full"
        ]
    elif unit_id == "E1-C3":
        command = _common_driver_command(driver, "correctness", manifest_path, run_root) + [
            "--backend", "fpga", "--suite", "standalone-semantic"
        ]
    elif unit_id == "E2-LATENCY":
        command = _common_driver_command(driver, "latency", manifest_path, run_root) + [
            "--backend", "fpga", "--case", "real_full_4096", "--warmup", "4",
            "--repetitions", "50",
        ]
    elif unit_id.startswith("E4-PROCESS-"):
        command = _common_driver_command(driver, "cold-start", manifest_path, run_root) + [
            "--backend", "fpga", "--case", "real_short_mixed", "--warmup", "1",
            "--repetitions", "1",
        ]
    elif unit_id == "E4-SUSTAINED":
        command = _common_driver_command(driver, "robustness", manifest_path, run_root) + [
            "--backend", "fpga", "--frames", "10000", "--batch", str(metadata["batch_size"]),
        ]
    elif unit_id.startswith("E5-"):
        driver_backend = "stock-seal" if backend == "stock-seal" else backend
        command = _common_driver_command(driver, "correctness", manifest_path, run_root) + [
            "--backend", driver_backend, "--suite", "semantic-matrix", "--trial-seeds", "0,1,2,3,4",
        ]
    elif unit_id in {"E6-FPGA", "E6-SEAL-EMBEDDED"}:
        command = _common_driver_command(driver, "latency", manifest_path, run_root) + [
            "--backend", backend, "--case", "real_full_4096", "--warmup", "4",
            "--repetitions", "50",
        ]
    elif unit_id in {"E8-BASE", "E8-EXTENSION"}:
        batches = ",".join(str(value) for value in metadata["batch_sizes"])
        command = _common_driver_command(driver, "throughput", manifest_path, run_root) + [
            "--backend", "fpga", "--batches", batches, "--repetitions", "20",
        ]
    else:
        raise ContractError(f"no frozen child command for planned unit {unit_id}")

    command.extend(["--attempt-id", attempt_id, "--planned-unit-id", unit_id])
    if unit_id == "E1-FPGA-TEST":
        working_directory = require_absolute(
            manifest["repository_state"]["seal_embedded"]["path"], "SEAL-Embedded repository"
        ) / "device"
    else:
        working_directory = driver.parent
    return command, working_directory, extraction_output


def assert_safe_child_command(
    command: list[str], manifest: dict[str, Any], extraction_output: Path | None
) -> None:
    if not command or not Path(command[0]).is_absolute():
        raise ContractError("child executable path must be absolute")
    lowered = " ".join(command).lower()
    forbidden = [token for token in FORBIDDEN_COMMAND_FRAGMENTS if token in lowered]
    if forbidden:
        raise ContractError(f"orchestrator refuses build/synthesis/programming command tokens: {forbidden}")
    executable = Path(command[0]).resolve()
    driver = Path(_artifact(manifest, "executable")["path"]).resolve()
    if executable == driver:
        if extraction_output is not None:
            raise ContractError("benchmark executable cannot own extraction output")
        return
    if executable != Path(sys.executable).resolve() or len(command) < 2:
        raise ContractError(f"child executable is outside the fixed allowlist: {executable}")
    script = Path(command[1]).resolve()
    if script not in {EXTRACT_REPORT.resolve(), EXTRACT_PRIOR_WORK.resolve()}:
        raise ContractError(f"Python child is outside the extraction allowlist: {script}")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _cpu_model() -> str | None:
    text = _read_text(Path("/proc/cpuinfo"))
    if text is None:
        return None
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"model name", "Hardware"}:
            return value.strip()
    return None


def _sysfs_values(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path("/").glob(pattern.lstrip("/"))):
        value = _read_text(path)
        if value is not None:
            rows.append({"path": str(path), "value": value})
    return rows


def _processes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        candidates = sorted(
            (path for path in Path("/proc").iterdir() if path.name.isdigit()),
            key=lambda path: int(path.name),
        )
    except OSError:
        return rows
    for path in candidates:
        name = _read_text(path / "comm")
        if name is not None:
            rows.append({"pid": int(path.name), "name": name})
    return rows


def capture_control(
    manifest: dict[str, Any], attempt_id: str, snapshot_id: str, phase: str, backend: str
) -> dict[str, Any]:
    notes: list[str] = []
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        affinity = []
        notes.append(f"cpu_affinity_unavailable:{type(error).__name__}")
    configuration = load_json(require_absolute(manifest["paths"]["configuration"], "configuration"))
    gbs_hash = _artifact(manifest, "gbs")["sha256"]
    programmed_hash = os.environ.get("FPT2026_PROGRAMMED_IMAGE_SHA256")
    board_model = os.environ.get("FPT2026_FPGA_BOARD_MODEL")
    board_revision = os.environ.get("FPT2026_FPGA_BOARD_REVISION")
    identity_source = os.environ.get("FPT2026_FPGA_IDENTITY_EVIDENCE_SOURCE")
    fpga_available = all((programmed_hash, board_model, board_revision, identity_source))
    if backend == "fpga" and not fpga_available:
        notes.append("fpga_runtime_identity_incomplete")
    if programmed_hash is not None and programmed_hash != gbs_hash:
        notes.append("programmed_image_differs_from_manifest_gbs")
    return {
        "schema_version": "1.0",
        "record_type": "control_snapshot",
        "run_id": manifest["run_id"],
        "snapshot_id": snapshot_id,
        "attempt_id": attempt_id,
        "phase": phase,
        "captured_utc": utc_now(),
        "hostname": socket.gethostname(),
        "processes": _processes(),
        "thermal": {
            "captured_monotonic_ns": time.monotonic_ns(),
            "thermal_zones": _sysfs_values("/sys/class/thermal/thermal_zone*/temp"),
        },
        "power": {
            "cpu_affinity": affinity,
            "cpu_model": _cpu_model(),
            "governors": _sysfs_values("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"),
            "current_frequency_khz": _sysfs_values("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"),
            "minimum_frequency_khz": _sysfs_values("/sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq"),
            "maximum_frequency_khz": _sysfs_values("/sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq"),
        },
        "fpga": {
            "required_for_backend": backend == "fpga",
            "capture_status": "available" if fpga_available else "unavailable",
            "unavailable_reason": None if fpga_available else "runtime_identity_environment_not_complete",
            "target_part": configuration["toolchain"]["fpga_target"],
            "board_model": board_model,
            "board_revision": board_revision,
            "identity_evidence_source": identity_source,
            "programmed_image_sha256": programmed_hash,
            "manifest_gbs_sha256": gbs_hash,
            "temperature": os.environ.get("FPT2026_FPGA_TEMPERATURE"),
            "health": os.environ.get("FPT2026_FPGA_HEALTH"),
        },
        "notes": notes,
    }


def _child_environment(manifest: dict[str, Any]) -> tuple[dict[str, str], str]:
    """Load the one immutable environment used verbatim by every official child."""
    environment_path = require_absolute(manifest["paths"]["environment"], "frozen environment")
    environment = load_json(environment_path)
    if not isinstance(environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ContractError("frozen child environment must be a string-to-string object")
    environment_sha256 = sha256_bytes(canonical_json_bytes(environment))
    if environment_sha256 != manifest["environment_sha256"]:
        raise ContractError("frozen child environment differs from the root manifest hash")
    if sha256_file(environment_path) != environment_sha256:
        raise ContractError("frozen environment file is not canonical or has changed")
    return environment, environment_sha256


def _timeout_for_unit(manifest: dict[str, Any], unit: dict[str, Any]) -> float:
    timeouts = manifest["timeouts"]
    unit_id = unit["planned_unit_id"]
    if unit["backend"] == "extraction":
        key = "extraction"
    elif unit_id.startswith("E8-") or unit_id == "E4-SUSTAINED":
        key = "batch"
    else:
        key = "single_frame"
    value = float(timeouts[key])
    if value <= 0:
        raise ContractError(f"non-positive frozen timeout for {unit_id}")
    return value


def _terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _scan_rows(path: Path, attempt_id: str) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed = 0
    if not path.exists():
        return records, malformed
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict) and value.get("attempt_id") == attempt_id:
                records.append(value)
    return records, malformed


def observe_attempt(manifest: dict[str, Any], unit: dict[str, Any], attempt_id: str) -> tuple[int, int, str | None]:
    paths = manifest["paths"]
    samples, malformed_samples = _scan_rows(require_absolute(paths["samples"]), attempt_id)
    events, malformed_events = _scan_rows(require_absolute(paths["events"]), attempt_id)
    correctness, malformed_correctness = _scan_rows(require_absolute(paths["correctness"]), attempt_id)
    if malformed_samples + malformed_events + malformed_correctness:
        last_progress = "malformed_partial_jsonl"
    else:
        last_progress = None
    if events:
        frames = len({int(row["frame_index"]) for row in events if isinstance(row.get("frame_index"), int)})
        last_progress = str(events[-1].get("event_record_id") or last_progress)
    else:
        frames = len({int(row["frame_index"]) for row in correctness if isinstance(row.get("frame_index"), int)})
        if correctness:
            last_progress = str(correctness[-1].get("correctness_record_id") or last_progress)
    progress_path = require_absolute(paths["run_root"]) / "raw" / "progress.jsonl"
    progress, malformed_progress = _scan_rows(progress_path, attempt_id)
    if progress:
        last_progress = str(progress[-1].get("progress_id") or last_progress)
        reported_frames = progress[-1].get("frame_count_completed")
        if isinstance(reported_frames, int) and reported_frames >= frames:
            frames = reported_frames
    elif malformed_progress:
        last_progress = "malformed_partial_progress_jsonl"
    return len(samples), frames, last_progress


def _journal_state(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not path.exists():
        return {}, {}
    rows = read_jsonl(path)
    started: dict[str, dict[str, Any]] = {}
    finished: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = started if row.get("record_type") == "attempt_started" else finished if row.get("record_type") == "attempt_finished" else None
        if target is None:
            raise ContractError(f"unknown attempt record type {row.get('record_type')!r}")
        attempt_id = row.get("attempt_id")
        if not isinstance(attempt_id, str) or attempt_id in target:
            raise ContractError(f"duplicate or invalid attempt row for {attempt_id!r}")
        target[attempt_id] = row
    return started, finished


def run_attempt(
    manifest: dict[str, Any],
    manifest_path: Path,
    unit: dict[str, Any],
    authorization: Authorization,
    terminate_grace_seconds: float,
) -> tuple[AttemptOutcome, Path | None]:
    require_authorized_unit(authorization, unit)
    run_root = require_absolute(manifest["paths"]["run_root"], "run root")
    attempts_path = require_absolute(manifest["paths"]["attempts"], "attempt journal")
    controls_path = require_absolute(manifest["paths"]["controls"], "controls journal")
    attempt_id = f"{manifest['run_id']}:{unit['planned_unit_id']}:attempt-00"
    started_existing, finished_existing = _journal_state(attempts_path)
    if attempt_id in started_existing or attempt_id in finished_existing:
        if attempt_id in started_existing and attempt_id in finished_existing:
            terminal = finished_existing[attempt_id]
            outcome = AttemptOutcome(
                attempt_id=attempt_id,
                terminal_status=terminal["terminal_status"],
                exit_code=terminal.get("exit_code"),
                signal_number=terminal.get("signal"),
                timed_out=bool(terminal["timed_out"]),
                observed_samples=int(terminal["observed_sample_count"]),
                observed_frames=int(terminal["observed_frame_count"]),
            )
            extraction = run_root / "raw" / "extractions" / f"{attempt_id}.json" if unit["backend"] == "extraction" else None
            return outcome, extraction
        raise ContractError(f"unterminated existing attempt cannot be reused: {attempt_id}")

    command, working_directory, extraction_output = command_for_unit(manifest, manifest_path, unit, attempt_id)
    assert_safe_child_command(command, manifest, extraction_output)
    working_directory = require_absolute(working_directory.resolve(), "child working directory")
    if not working_directory.is_dir():
        raise ContractError(f"child working directory is missing: {working_directory}")
    timeout_seconds = _timeout_for_unit(manifest, unit)
    if terminate_grace_seconds <= 0:
        raise ContractError("terminate grace must be positive")
    pre_snapshot_id = f"{attempt_id}:control-pre"
    post_snapshot_id = f"{attempt_id}:control-post"
    environment, environment_sha256 = _child_environment(manifest)
    log_root = run_root / "logs" / "stdout-stderr"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{attempt_id}.stdout"
    stderr_path = log_root / f"{attempt_id}.stderr"

    process: subprocess.Popen[Any] | None = None
    terminal_status = "launch_failed"
    exit_code: int | None = None
    signal_number: int | None = None
    timed_out = False
    interrupted: BaseException | None = None
    with stdout_path.open("xb") as stdout_stream, stderr_path.open("xb") as stderr_stream:
        pre_control = capture_control(
            manifest, attempt_id, pre_snapshot_id, "pre", unit["backend"]
        )
        append_jsonl(controls_path, pre_control)
        started = {
            "schema_version": "1.0",
            "record_type": "attempt_started",
            "run_id": manifest["run_id"],
            "attempt_id": attempt_id,
            "experiment_id": unit["experiment_id"],
            "planned_unit_id": unit["planned_unit_id"],
            "effective_plan_sha256": authorization.snapshot_sha256,
            "command": command,
            "command_sha256": sha256_bytes(canonical_json_bytes(command)),
            "environment": environment,
            "environment_sha256": environment_sha256,
            "working_directory": str(working_directory),
            "planned_sample_count": int(unit["timing_rows"]),
            "planned_frame_count": int(unit["frames"]),
            "timeout_seconds": timeout_seconds,
            "pre_control_snapshot_id": pre_snapshot_id,
            "started_utc": utc_now(),
        }
        append_jsonl(attempts_path, started)
        try:
            process = subprocess.Popen(
                command,
                cwd=working_directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                start_new_session=True,
                shell=False,
            )
            try:
                return_code = process.wait(timeout=timeout_seconds)
                if return_code < 0:
                    terminal_status = "signaled"
                    signal_number = -return_code
                else:
                    terminal_status = "exited"
                    exit_code = return_code
            except subprocess.TimeoutExpired:
                timed_out = True
                terminal_status = "timed_out"
                _terminate_process_group(process, terminate_grace_seconds)
                return_code = process.returncode
                if return_code is not None and return_code < 0:
                    signal_number = -return_code
        except OSError:
            terminal_status = "launch_failed"
        except BaseException as error:
            interrupted = error
            if process is not None:
                _terminate_process_group(process, terminate_grace_seconds)
                return_code = process.returncode
                terminal_status = "signaled" if return_code is not None and return_code < 0 else "exited"
                signal_number = -return_code if return_code is not None and return_code < 0 else None
                exit_code = return_code if return_code is not None and return_code >= 0 else None
        finally:
            stdout_stream.flush()
            stderr_stream.flush()
            os.fsync(stdout_stream.fileno())
            os.fsync(stderr_stream.fileno())

    observed_samples, observed_frames, last_progress_id = observe_attempt(manifest, unit, attempt_id)
    post_control = capture_control(
        manifest, attempt_id, post_snapshot_id, "post", unit["backend"]
    )
    append_jsonl(controls_path, post_control)
    finished = {
        "schema_version": "1.0",
        "record_type": "attempt_finished",
        "run_id": manifest["run_id"],
        "attempt_id": attempt_id,
        "experiment_id": unit["experiment_id"],
        "terminal_status": terminal_status,
        "exit_code": exit_code,
        "signal": signal_number,
        "timed_out": timed_out,
        "observed_sample_count": observed_samples,
        "observed_frame_count": observed_frames,
        "last_progress_id": last_progress_id,
        "post_control_snapshot_id": post_snapshot_id,
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
        "finished_utc": utc_now(),
    }
    append_jsonl(attempts_path, finished)
    if interrupted is not None:
        raise interrupted
    return (
        AttemptOutcome(
            attempt_id=attempt_id,
            terminal_status=terminal_status,
            exit_code=exit_code,
            signal_number=signal_number,
            timed_out=timed_out,
            observed_samples=observed_samples,
            observed_frames=observed_frames,
        ),
        extraction_output,
    )


def _phase_units(authorization: Authorization, planned_ids: set[str]) -> list[dict[str, Any]]:
    units = [unit for unit in authorization.units if unit["planned_unit_id"] in planned_ids]
    if {unit["planned_unit_id"] for unit in units} != planned_ids:
        missing = sorted(planned_ids - {unit["planned_unit_id"] for unit in units})
        raise ContractError(f"effective plan is missing requested units: {missing}")
    return units


def _run_units(
    manifest: dict[str, Any],
    manifest_path: Path,
    authorization: Authorization,
    units: Iterable[dict[str, Any]],
    terminate_grace_seconds: float,
    extraction_results: dict[str, Path],
) -> None:
    for unit in units:
        outcome, extraction_output = run_attempt(
            manifest, manifest_path, unit, authorization, terminate_grace_seconds
        )
        if extraction_output is not None:
            extraction_results[unit["experiment_id"]] = extraction_output
        if not outcome.passed:
            raise ContractError(
                f"attempt {outcome.attempt_id} ended as {outcome.terminal_status} "
                f"with exit={outcome.exit_code} signal={outcome.signal_number}"
            )
        if outcome.observed_samples != int(unit["timing_rows"]):
            raise ContractError(f"attempt {outcome.attempt_id} emitted the wrong timing-row count")
        if outcome.observed_frames != int(unit["frames"]):
            raise ContractError(f"attempt {outcome.attempt_id} emitted the wrong frame count")


def _samples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    path = require_absolute(manifest["paths"]["samples"], "samples")
    return read_jsonl(path) if path.exists() else []


def _write_e8_evidence(manifest: dict[str, Any], name: str, required_batches: list[int]) -> Path:
    run_root = require_absolute(manifest["paths"]["run_root"], "run root")
    evidence = e8_evidence_document(_samples(manifest), required_batches=required_batches)
    path = run_root / "planning" / name
    write_new_json(path, evidence)
    return path


def _validate_extraction(path: Path, experiment_id: str, run_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"{experiment_id} extraction child produced no output: {path}")
    result = load_json(path)
    expected_type = "report_extraction" if experiment_id == "E3" else "prior_work_extraction"
    if result.get("record_type") != expected_type or result.get("experiment_id") != experiment_id:
        raise ContractError(f"{experiment_id} extraction output has the wrong type")
    if result.get("run_id") != run_id:
        raise ContractError(f"{experiment_id} extraction output has the wrong run ID")
    evidence = result.get("source_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ContractError(f"{experiment_id} extraction output lacks source_evidence[]")
    for row in evidence:
        if not isinstance(row, dict) or not {"source_path", "source_sha256", "locator"}.issubset(row):
            raise ContractError(f"{experiment_id} extraction source evidence is incomplete")
        source = require_absolute(row["source_path"], f"{experiment_id} source")
        if not source.is_file() or sha256_file(source) != row["source_sha256"]:
            raise ContractError(f"{experiment_id} source hash does not resolve: {source}")
        if not isinstance(row["locator"], str) or not row["locator"]:
            raise ContractError(f"{experiment_id} source locator is empty")
    if experiment_id == "E3":
        report_evidence = result.get("report_evidence")
        required_report_fields = {
            "report_root_path", "report_tree_sha256", "info_ndjson_path",
            "info_ndjson_sha256", "quartus_ndjson_path", "quartus_ndjson_sha256",
            "source_revision", "device_image_sha256", "fpga_target", "fitter_seed",
            "kernel_clock_mhz", "resources",
        }
        if not isinstance(report_evidence, dict) or not required_report_fields.issubset(report_evidence):
            raise ContractError("E3 extraction lacks the complete report_evidence payload")
        if result.get("association", {}).get("resolved") is not True:
            raise ContractError("E3 report/image association is unresolved")
    else:
        required_e7_fields = {
            "parameter_set", "operation_boundary", "platform", "frequency", "latency",
            "throughput", "classification",
        }
        for row in evidence:
            if not required_e7_fields.issubset(row):
                raise ContractError("E7 source evidence lacks contextual quantity fields")
            for quantity_name in ("frequency", "latency", "throughput"):
                quantity = row[quantity_name]
                if not isinstance(quantity, dict) or set(quantity) != {"value", "unit", "unavailable_reason"}:
                    raise ContractError(f"E7 {quantity_name} is not a complete quantity object")
        if result.get("direct_speedup_claim_permitted") is not False:
            raise ContractError("E7 extraction improperly permits a cross-platform speedup claim")
    return result


def _append_extraction_statuses(
    manifest: dict[str, Any], extraction_results: dict[str, Path], final_effective_sha256: str
) -> None:
    status_path = require_absolute(manifest["paths"]["experiment_status"], "experiment status")
    existing = read_jsonl(status_path) if status_path.exists() else []
    existing_ids = {row.get("experiment_id") for row in existing}
    for experiment_id in ("E3", "E7"):
        path = extraction_results.get(experiment_id)
        if path is None:
            raise ContractError(f"missing owned {experiment_id} extraction child output")
        result = _validate_extraction(path, experiment_id, manifest["run_id"])
        output_sha256 = sha256_file(path)
        evidence_hashes = [output_sha256]
        for row in result["source_evidence"]:
            if row["source_sha256"] not in evidence_hashes:
                evidence_hashes.append(row["source_sha256"])
        if experiment_id == "E3":
            report_evidence = result["report_evidence"]
            for field in (
                "report_tree_sha256",
                "info_ndjson_sha256",
                "quartus_ndjson_sha256",
                "device_image_sha256",
            ):
                if report_evidence[field] not in evidence_hashes:
                    evidence_hashes.append(report_evidence[field])
        if experiment_id in existing_ids:
            raise ContractError(f"orchestrator-owned {experiment_id} status already exists")
        status = {
                "schema_version": "1.0",
                "record_type": "experiment_status",
                "run_id": manifest["run_id"],
                "experiment_id": experiment_id,
                "status": "pass",
                "effective_plan_sha256": final_effective_sha256,
                "validated_counts": {
                    "attempts": 1,
                    "samples": 0,
                    "frames": 0,
                    "events": 0,
                    "correctness": 0,
                },
                "evidence_sha256s": evidence_hashes,
                "reason": None,
                "finished_utc": utc_now(),
        }
        if experiment_id == "E3":
            status["report_evidence"] = result["report_evidence"]
        else:
            status["child_output_path"] = str(path)
            status["child_output_sha256"] = output_sha256
            status["source_evidence"] = result["source_evidence"]
        append_jsonl(status_path, status)
        existing_ids.add(experiment_id)


def main() -> int:
    args = parse_args()
    manifest_path = require_absolute(args.manifest.resolve(), "manifest")
    manifest = load_json(manifest_path)
    run_root = require_absolute(manifest["paths"]["run_root"], "run root")
    if manifest_path != run_root / "manifest.json":
        raise ContractError("manifest path does not match its immutable run root")
    if manifest.get("paper_eligible") is not True:
        raise ContractError("official matrix refuses a non-paper manifest")

    extraction_results: dict[str, Path] = {}
    base_authorization = authorize_base(manifest)
    e7_units = [unit for unit in base_authorization.units if unit["planned_unit_id"] == "E7-PRIOR-WORK"]
    if len(e7_units) != 1:
        raise ContractError("base plan must own exactly one E7 extraction child")
    _run_units(
        manifest,
        manifest_path,
        base_authorization,
        base_authorization.units,
        args.terminate_grace_seconds,
        extraction_results,
    )

    base_batches = [1, 2, 4, 8, 16, 32, 64, 128]
    base_evidence_path = _write_e8_evidence(
        manifest, "e8-base-throughput-evidence.json", base_batches
    )
    finalize_e8(manifest, base_evidence_path)
    e8_authorization = authorize_e8(manifest)
    base_ids = {unit["planned_unit_id"] for unit in base_authorization.units}
    extension_ids = {
        unit["planned_unit_id"] for unit in e8_authorization.units if unit["planned_unit_id"] not in base_ids
    }
    _run_units(
        manifest,
        manifest_path,
        e8_authorization,
        _phase_units(e8_authorization, extension_ids),
        args.terminate_grace_seconds,
        extraction_results,
    )

    e8_amendment = load_json(require_absolute(manifest["paths"]["manifest_amendment_e8"]))
    all_batches = base_batches + ([256, 512] if e8_amendment["triggered"] else [])
    complete_evidence_path = _write_e8_evidence(
        manifest, "e8-complete-throughput-evidence.json", all_batches
    )
    finalize_e4(manifest, complete_evidence_path)
    e4_authorization = authorize_e4(manifest)
    e8_ids = {unit["planned_unit_id"] for unit in e8_authorization.units}
    e4_ids = {
        unit["planned_unit_id"] for unit in e4_authorization.units if unit["planned_unit_id"] not in e8_ids
    }
    _run_units(
        manifest,
        manifest_path,
        e4_authorization,
        _phase_units(e4_authorization, e4_ids),
        args.terminate_grace_seconds,
        extraction_results,
    )
    final_effective_sha256 = sha256_file(
        require_absolute(manifest["paths"]["effective_plan_e4"], "final effective plan")
    )
    _append_extraction_statuses(manifest, extraction_results, final_effective_sha256)
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
        print(f"run_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
