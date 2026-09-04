#!/usr/bin/env python3
"""Freeze a new immutable FPT 2026 run manifest and base plan.

This preparatory command never launches the benchmark driver. It refuses missing
artifacts and dirty paper repositories unless the explicit diagnostic override is
used; the override permanently marks the run non-paper.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (
    ContractError,
    canonical_json_bytes,
    load_json,
    require_absolute,
    sha256_bytes,
    sha256_file,
    sha256_tree,
    utc_now,
    write_new_bytes,
    write_new_json,
)
from plan_matrix import MATRIX_ID, base_units, planned_document

PLAN_PATH = Path("/home/joe/Documents/Obsidian/School/Thesis/FPT 2026 Paper/Benchmarks/Benchmark_Plan.md")
DEFAULT_ALOHA_PAPER = Path("/home/joe/Documents/Obsidian/School/Thesis/Aloha-HE/Aloha-HE Paper.pdf")
ENVIRONMENT_WHITELIST = [
    "HOME", "USER", "LANG", "LC_ALL", "TMPDIR",
    "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH",
    "ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER",
    "CL_CONTEXT_MPSIM_DEVICE_INTELFPGA", "INTELFPGA_SIM_DEVICE_SPEC_DIR",
    "INTELFPGAOCLSDKROOT", "AOCL_BOARD_PACKAGE_ROOT", "QUARTUS_ROOTDIR",
    "LM_LICENSE_FILE", "OCL_ICD_VENDORS", "ZE_AFFINITY_MASK",
    "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "MKL_NUM_THREADS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--accelerator-archive", required=True, type=Path)
    parser.add_argument("--gbs", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--benchmark-key-compact", required=True, type=Path)
    parser.add_argument("--benchmark-key-seal", required=True, type=Path)
    parser.add_argument("--ciphertext-context", required=True, type=Path)
    parser.add_argument("--cmake-cache", required=True, type=Path)
    parser.add_argument("--toolchain-record", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--aloha-paper", type=Path, default=DEFAULT_ALOHA_PAPER)
    parser.add_argument("--timeout", action="append", default=[], metavar="NAME=SECONDS")
    parser.add_argument("--allow-dirty-diagnostic", action="store_true")
    return parser.parse_args()


def parse_timeouts(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for item in values:
        name, separator, raw_value = item.partition("=")
        if not separator or not name:
            raise ContractError(f"invalid timeout assignment: {item!r}")
        value = float(raw_value)
        if value <= 0:
            raise ContractError(f"timeout must be positive: {item!r}")
        parsed[name] = value
    required = {"pilot", "single_frame", "batch", "extraction"}
    missing = sorted(required - parsed.keys())
    if missing:
        raise ContractError(f"missing fixed timeouts: {', '.join(missing)}")
    return parsed


def git_text(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments], cwd=repository, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return process.stdout.strip()


def git_is_ancestor(repository: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    process = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode not in (0, 1):
        raise ContractError(
            f"cannot test Git ancestry in {repository}: {process.stderr.strip()}"
        )
    return process.returncode == 0


def freeze_repository(repository: Path, allow_dirty: bool) -> dict[str, Any]:
    require_absolute(repository, "repository")
    if not (repository / ".git").exists():
        raise ContractError(f"not a Git worktree: {repository}")
    status = git_text(repository, "status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise ContractError(f"paper run refuses dirty repository: {repository}")
    return {
        "path": str(repository),
        "head": git_text(repository, "rev-parse", "HEAD"),
        "branch": git_text(repository, "branch", "--show-current"),
        "dirty": dirty,
        "status_sha256": sha256_bytes((status + "\n").encode("utf-8")),
        "status": status,
    }


def freeze_file(artifact_id: str, path: Path) -> dict[str, Any]:
    source = require_absolute(path, artifact_id)
    if not source.is_file():
        raise ContractError(f"required artifact is not a file: {source}")
    return {
        "artifact_id": artifact_id,
        "path": str(source),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def validate_ciphertext_context(context: dict[str, Any], config: dict[str, Any]) -> None:
    required = {
        "special_prime", "key_context_moduli", "key_parms_id_words",
        "first_parms_id_words", "ciphertext_size", "ciphertext_modulus_count",
        "ciphertext_is_ntt_form", "pipeline_input_block_size", "uint32_size",
        "key_parms_id_bytes_sha256", "first_parms_id_bytes_sha256",
        "key_pair_verification",
    }
    missing = sorted(required - context.keys())
    if missing:
        raise ContractError(f"ciphertext context is missing: {', '.join(missing)}")
    data_moduli = config["ckks"]["data_moduli"]
    special_prime = context["special_prime"]
    if not isinstance(special_prime, int) or special_prime <= 0:
        raise ContractError("ciphertext context special_prime must be a positive runtime value")
    if context["key_context_moduli"] != data_moduli + [special_prime]:
        raise ContractError("ciphertext context modulus chain differs from configured data primes")
    for field in ("key_parms_id_words", "first_parms_id_words"):
        words = context[field]
        if (not isinstance(words, list) or len(words) != 4 or
                any(not isinstance(word, int) or word < 0 for word in words) or
                not any(words)):
            raise ContractError(f"ciphertext context {field} must contain four non-placeholder words")
    if context["ciphertext_size"] != 2 or context["ciphertext_modulus_count"] != 6:
        raise ContractError("ciphertext context must describe a two-polynomial, six-limb ciphertext")
    if context["ciphertext_is_ntt_form"] is not True:
        raise ContractError("ciphertext context must be in NTT form")
    if not isinstance(context["pipeline_input_block_size"], int) or context["pipeline_input_block_size"] <= 0:
        raise ContractError("pipeline_input_block_size must be a positive runtime sizeof value")
    if context["uint32_size"] != 4:
        raise ContractError("uint32_size must be exactly four bytes")
    verification = context["key_pair_verification"]
    if (not isinstance(verification, dict) or verification.get("status") != "pass" or
            not verification.get("attempt_id")):
        raise ContractError("compact/stock benchmark key-pair verification is not a recorded pass")


def publish_base_plan(run_root: Path) -> tuple[Path, str, Path, str, list[dict[str, Any]]]:
    planning = run_root / "planning"
    units = base_units()
    units_path = planning / "planned_units-base.json"
    units_sha256 = write_new_json(units_path, planned_document("planned_units_base", units))
    effective = {
        "schema_version": "1.0",
        "document_type": "effective_plan_snapshot",
        "phase": "base",
        "predecessor_sha256": None,
        "component_sha256s": [units_sha256],
        "planned_unit_ids": [unit["planned_unit_id"] for unit in units],
        "declared_counts": planned_document("planned_units_base", units)["declared_counts"],
        "created_utc": utc_now(),
    }
    effective_path = planning / "effective-plan-00-base.json"
    effective_sha256 = write_new_json(effective_path, effective)
    return units_path, units_sha256, effective_path, effective_sha256, units


def main() -> int:
    args = parse_args()
    config_path = require_absolute(args.config.resolve(), "config")
    output = require_absolute(args.output.resolve(), "output")
    if output.exists():
        raise FileExistsError(f"run output must not already exist: {output}")
    config = load_json(config_path)
    if config.get("campaign_id") != MATRIX_ID:
        raise ContractError(f"configuration campaign_id must be {MATRIX_ID}")
    timeouts = parse_timeouts(args.timeout)

    paths = config.get("paths", {})
    accelerator_repository = require_absolute(paths["accelerator_repository"], "accelerator repository")
    seal_repository = require_absolute(paths["seal_embedded_repository"], "SEAL-Embedded repository")
    accelerator_git = freeze_repository(accelerator_repository, args.allow_dirty_diagnostic)
    seal_git = freeze_repository(seal_repository, args.allow_dirty_diagnostic)
    if accelerator_git["branch"] != "8k_benchmarks":
        raise ContractError(f"accelerator branch must be 8k_benchmarks, observed {accelerator_git['branch']!r}")
    accelerator_baseline = config["pinned_revisions"]["accelerator_baseline_commit"]
    if not git_is_ancestor(accelerator_repository, accelerator_baseline):
        raise ContractError(
            f"accelerator implementation is not descended from baseline {accelerator_baseline}"
        )
    accelerator_git["baseline_commit"] = accelerator_baseline
    accelerator_git["baseline_is_ancestor"] = True
    expected_seal = config["pinned_revisions"]["seal_embedded_commit"]
    if seal_git["head"] != expected_seal:
        raise ContractError(f"SEAL-Embedded HEAD differs from pinned commit {expected_seal}")

    required_files = {
        "accelerator_archive": args.accelerator_archive,
        "gbs": args.gbs,
        "executable": args.executable,
        "benchmark_key_compact": args.benchmark_key_compact,
        "benchmark_key_seal": args.benchmark_key_seal,
        "ciphertext_context": args.ciphertext_context,
        "cmake_cache": args.cmake_cache,
        "toolchain_record": args.toolchain_record,
        "aloha_he_paper": args.aloha_paper,
        "benchmark_plan": PLAN_PATH,
    }
    artifacts = [freeze_file(name, require_absolute(path.resolve(), name)) for name, path in required_files.items()]
    by_artifact_id = {artifact["artifact_id"]: artifact for artifact in artifacts}
    for artifact_id, expected_sha256 in config["expected_artifact_sha256"].items():
        observed_sha256 = by_artifact_id[artifact_id]["sha256"]
        if observed_sha256 != expected_sha256:
            raise ContractError(
                f"{artifact_id} is not the frozen seed-7 artifact: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
    ciphertext_context = load_json(
        require_absolute(args.ciphertext_context.resolve(), "ciphertext context")
    )
    validate_ciphertext_context(ciphertext_context, config)
    report_root = require_absolute(args.report_root.resolve(), "report root")
    report_tree_sha256, report_entries = sha256_tree(report_root)
    artifacts.append({
        "artifact_id": "report_root",
        "path": str(report_root),
        "sha256": report_tree_sha256,
        "size_bytes": sum(int(entry["size_bytes"]) for entry in report_entries),
    })

    for directory in ("planning", "raw", "logs", "summaries", "diagnostics"):
        (output / directory).mkdir(parents=True, exist_ok=False)
    created_utc = utc_now()
    config_snapshot = output / "configuration.json"
    configuration_sha256 = write_new_json(config_snapshot, config)
    report_tree_path = output / "raw" / "report-tree.json"
    report_tree_manifest_sha256 = write_new_json(report_tree_path, report_entries)
    units_path, units_sha256, effective_path, effective_sha256, units = publish_base_plan(output)

    environment_path = output / "environment.json"
    environment = {name: os.environ[name] for name in ENVIRONMENT_WHITELIST if name in os.environ}
    environment_sha256 = write_new_json(environment_path, environment)
    artifact_index_path = output / "artifacts.sha256"
    artifact_index = "".join(
        f"{artifact['sha256']}  {artifact['path']}\n"
        for artifact in sorted(artifacts, key=lambda item: item["artifact_id"])
    )
    write_new_bytes(artifact_index_path, artifact_index.encode("utf-8"))
    artifact_index_sha256 = sha256_file(artifact_index_path)

    commands_path = output / "commands.txt"
    command_line = " ".join(shlex.quote(value) for value in sys.argv) + "\n"
    write_new_bytes(commands_path, command_line.encode("utf-8"))

    paper_eligible = not accelerator_git["dirty"] and not seal_git["dirty"]
    manifest = {
        "schema_version": "1.0",
        "document_type": "root_manifest",
        "run_id": output.name,
        "campaign_id": MATRIX_ID,
        "created_utc": created_utc,
        "paper_eligible": paper_eligible,
        "paths": {
            "run_root": str(output),
            "configuration": str(config_snapshot),
            "environment": str(environment_path),
            "artifact_index": str(artifact_index_path),
            "planned_units_base": str(units_path),
            "planned_units_e8_extension": str(output / "planning" / "planned_units-e8-extension.json"),
            "planned_units_e4_sustained": str(output / "planning" / "planned_units-e4-sustained.json"),
            "manifest_amendment_e8": str(output / "planning" / "manifest-amendment-e8.json"),
            "manifest_amendment_e4": str(output / "planning" / "manifest-amendment-e4.json"),
            "effective_plan_base": str(effective_path),
            "effective_plan_e8": str(output / "planning" / "effective-plan-01-e8.json"),
            "effective_plan_e4": str(output / "planning" / "effective-plan-02-e4.json"),
            "attempts": str(output / "raw" / "attempts.jsonl"),
            "controls": str(output / "raw" / "controls.jsonl"),
            "samples": str(output / "raw" / "samples.jsonl"),
            "events": str(output / "raw" / "events.jsonl"),
            "correctness": str(output / "raw" / "correctness.jsonl"),
            "experiment_status": str(output / "raw" / "experiment-status.jsonl"),
        },
        "pinned_revisions": {
            "accelerator_head": accelerator_git["head"],
            "accelerator_branch": accelerator_git["branch"],
            "accelerator_baseline_commit": config["pinned_revisions"]["accelerator_baseline_commit"],
            "seal_embedded_head": seal_git["head"],
            "microsoft_seal_commit": config["pinned_revisions"]["microsoft_seal_commit"],
        },
        "repository_state": {"accelerator": accelerator_git, "seal_embedded": seal_git},
        "artifacts": artifacts,
        "report_tree_manifest_sha256": report_tree_manifest_sha256,
        "environment_sha256": environment_sha256,
        "artifact_index_sha256": artifact_index_sha256,
        "configuration_sha256": configuration_sha256,
        "base_plan_sha256": units_sha256,
        "base_effective_plan_sha256": effective_sha256,
        "timeouts": timeouts,
        "parameters": {**config["ckks"], **ciphertext_context},
        "planned_counts": planned_document("planned_units_base", units)["declared_counts"],
        "branch_state": {"e8_extension": "pending", "e4_sustained": "pending"},
    }
    manifest_path = output / "manifest.json"
    write_new_json(manifest_path, manifest)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, FileExistsError, KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"prepare_run: {error}", file=sys.stderr)
        raise SystemExit(2)
