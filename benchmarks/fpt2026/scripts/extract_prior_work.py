#!/usr/bin/env python3
"""Extract the contextual E7 table from the manifest-pinned Aloha-HE paper.

The extractor deliberately has no network or discovery path.  It accepts only the
paper artifact frozen into the run manifest, verifies both the manifest pin and
the bytes on disk, and emits locator-bearing evidence.  The values below are the
small audited table needed by the benchmark plan; they are not a performance
baseline for this harness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common import ContractError, load_json, require_absolute, sha256_file, write_new_json

EXPECTED_PAPER_PATH = Path(
    "/home/joe/Documents/Obsidian/School/Thesis/Aloha-HE/Aloha-HE Paper.pdf"
)
EXPECTED_PAPER_SHA256 = "a7cdcf0e1254c138d37c25426448a694749a6cd4b69f54ac1ce10a87ccd32ec9"
SOURCE_TITLE = "Aloha-HE: A Low-Area Hardware Accelerator for Client-Side Operations in Homomorphic Encryption"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _artifact(manifest: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    matches = [item for item in manifest.get("artifacts", []) if item.get("artifact_id") == artifact_id]
    if len(matches) != 1:
        raise ContractError(f"manifest must contain exactly one {artifact_id!r} artifact")
    return matches[0]


def pinned_paper(manifest: dict[str, Any]) -> tuple[Path, str]:
    """Return the sole authorized E7 source after validating every pin."""
    artifact = _artifact(manifest, "aloha_he_paper")
    source = require_absolute(artifact["path"], "Aloha-HE paper")
    manifest_hash = artifact.get("sha256")
    if source != EXPECTED_PAPER_PATH:
        raise ContractError(
            f"E7 refuses an unplanned source path: expected {EXPECTED_PAPER_PATH}, observed {source}"
        )
    if manifest_hash != EXPECTED_PAPER_SHA256:
        raise ContractError("manifest Aloha-HE paper hash differs from the approved source pin")
    if not source.is_file():
        raise ContractError(f"pinned Aloha-HE paper is missing: {source}")
    observed_hash = sha256_file(source)
    if observed_hash != manifest_hash:
        raise ContractError("Aloha-HE paper bytes differ from the manifest hash")
    return source, observed_hash


def _context_payload(
    *,
    platform: str,
    device: str,
    frequency_mhz: int,
    encryption_ms: float,
    decryption_ms: float,
    ntt_ms: float,
    locator_row: str,
) -> dict[str, Any]:
    return {
        "parameter_set": {
            "scheme": "CKKS",
            "poly_modulus_degree": 8192,
            "coefficient_modulus": "3 x 54-bit primes",
            "comparison_caveat": (
                "The FPT 2026 harness instead uses six 30-bit primes on Intel Agilex 7; "
                "this is contextual evidence, not a matched E6 parameter set."
            ),
        },
        "operation_boundary": (
            "CKKS plaintext-to-ciphertext: encoding + encryption + software + hardware + "
            "data transfer on the authors' complete system; not application_e2e_v1"
        ),
        "platform": f"{platform}; {device}; Xilinx Vivado 2019.1",
        "frequency": {"value": frequency_mhz, "unit": "MHz", "unavailable_reason": None},
        "latency": {"value": encryption_ms, "unit": "ms", "unavailable_reason": None},
        "throughput": {
            "value": None,
            "unit": "frames_per_second",
            "unavailable_reason": "the cited table reports single-operation latency, not sustained throughput",
        },
        "classification": "measured",
    }


def extract(manifest: dict[str, Any]) -> dict[str, Any]:
    source, source_hash = pinned_paper(manifest)
    contexts = [
        (
            "p. 5, Section IV.B and Table I, rows 'Our — Kintex-7 @200MHz'",
            _context_payload(
                platform="Genesys2 Kintex-7 FPGA with MicroBlaze soft CPU",
                device="xc7k325tffg900-2",
                frequency_mhz=200,
                encryption_ms=1.87,
                decryption_ms=0.87,
                ntt_ms=0.267,
                locator_row="Our — Kintex-7 @200MHz",
            ),
        ),
        (
            "p. 5, Section IV.B and Table I, rows 'Our — ZYNQ-7000 @130MHz'",
            _context_payload(
                platform="PYNQ-Z2 ZYNQ-7000 SoC with dual-core ARM CPU",
                device="xc7z020clg400-1",
                frequency_mhz=130,
                encryption_ms=3.04,
                decryption_ms=1.30,
                ntt_ms=0.410,
                locator_row="Our — ZYNQ-7000 @130MHz",
            ),
        ),
    ]
    evidence = [
        {
            "source_path": str(source),
            "source_sha256": source_hash,
            "locator": locator,
            **payload,
        }
        for locator, payload in contexts
    ]
    for row in evidence:
        required = {
            "source_path",
            "source_sha256",
            "locator",
            "parameter_set",
            "operation_boundary",
            "platform",
            "frequency",
            "latency",
            "throughput",
            "classification",
        }
        if not required.issubset(row):
            raise ContractError(f"incomplete prior-work payload at {row['locator']}")
    return {
        "schema_version": "1.0",
        "record_type": "prior_work_extraction",
        "run_id": manifest["run_id"],
        "experiment_id": "E7",
        "context_only": True,
        "direct_speedup_claim_permitted": False,
        "source_evidence": evidence,
    }


def main() -> int:
    args = parse_args()
    manifest_path = require_absolute(args.manifest.resolve(), "manifest")
    output_path = require_absolute(args.output.resolve(), "output")
    manifest = load_json(manifest_path)
    write_new_json(output_path, extract(manifest))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, FileExistsError, FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"extract_prior_work: {error}", file=sys.stderr)
        raise SystemExit(2)
