#!/usr/bin/env python3
"""Source and pure-contract tests for the FPT 2026 Python harness.

These are definitions only.  This file intentionally has no unittest.main entry
point and was not executed while the source-only harness was authored.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = HARNESS_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import extract_prior_work  # noqa: E402
import extract_report  # noqa: E402
import plan_matrix  # noqa: E402
import run_matrix  # noqa: E402
import summarize_results  # noqa: E402
from common import ContractError, canonical_json_bytes, sha256_bytes, write_new_json  # noqa: E402


class MatrixSourceContractTests(unittest.TestCase):
    def test_frozen_base_counts_and_unique_e7_child(self) -> None:
        units = plan_matrix.base_units()
        self.assertEqual(plan_matrix.count_units(units), plan_matrix.EXPECTED_BASE_COUNTS)
        e7 = [unit for unit in units if unit["planned_unit_id"] == "E7-PRIOR-WORK"]
        self.assertEqual(len(e7), 1)
        self.assertEqual(e7[0]["backend"], "extraction")
        self.assertEqual(e7[0]["child_attempts"], 1)

    def test_e1_uses_the_four_named_production_path_patterns(self) -> None:
        unit = next(unit for unit in plan_matrix.base_units() if unit["planned_unit_id"] == "E1-C1-C2")
        self.assertEqual(
            unit["metadata"]["patterns"],
            [
                "ntt_sparse_impulse",
                "ntt_alternating_ternary",
                "ntt_negative_boundary",
                "ntt_shake256",
            ],
        )

    def test_conditional_counts_are_exact(self) -> None:
        self.assertEqual(plan_matrix.e8_extension_units(False), [])
        self.assertEqual(
            plan_matrix.count_units(plan_matrix.e8_extension_units(True)),
            plan_matrix.EXPECTED_E8_EXTENSION_COUNTS,
        )
        sustained = plan_matrix.e4_sustained_units(64)
        self.assertEqual(sustained[0]["frames"], 10_000)
        self.assertEqual(sustained[0]["timing_rows"], 157)
        self.assertEqual(sustained[0]["metadata"]["final_partial_batch_size"], 16)


class OrchestratorSourceContractTests(unittest.TestCase):
    def test_child_process_is_external_and_watchdog_bounded(self) -> None:
        source = inspect.getsource(run_matrix.run_attempt)
        self.assertIn("subprocess.Popen", source)
        self.assertIn("start_new_session=True", source)
        self.assertIn("shell=False", source)
        self.assertIn("process.wait(timeout=timeout_seconds)", source)
        watchdog = inspect.getsource(run_matrix._terminate_process_group)
        self.assertIn("signal.SIGTERM", watchdog)
        self.assertIn("signal.SIGKILL", watchdog)

    def test_pre_control_precedes_start_and_post_control_precedes_finish(self) -> None:
        source = inspect.getsource(run_matrix.run_attempt)
        pre_append = source.index("append_jsonl(controls_path, pre_control)")
        started_append = source.index("append_jsonl(attempts_path, started)")
        launch = source.index("subprocess.Popen")
        post_append = source.index("append_jsonl(controls_path, post_control)")
        finished_append = source.index("append_jsonl(attempts_path, finished)")
        self.assertLess(pre_append, started_append)
        self.assertLess(started_append, launch)
        self.assertLess(post_append, finished_append)

    def test_attempt_start_material_and_digests_are_present(self) -> None:
        source = inspect.getsource(run_matrix.run_attempt)
        self.assertIn('"command": command', source)
        self.assertIn('"command_sha256": sha256_bytes(canonical_json_bytes(command))', source)
        self.assertIn('"environment": environment', source)
        self.assertIn('"environment_sha256": environment_sha256', source)

    def test_unlisted_or_mutated_unit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            snapshot_path = root / "effective.json"
            snapshot = {
                "schema_version": "1.0",
                "document_type": "effective_plan_snapshot",
                "phase": "base",
                "predecessor_sha256": None,
                "component_sha256s": ["0" * 64],
                "planned_unit_ids": ["authorized"],
                "declared_counts": {
                    "attempts": 1,
                    "attempt_rows": 2,
                    "control_snapshots": 2,
                    "frames": 0,
                    "event_rows": 0,
                    "correctness_rows": 0,
                    "timing_rows": 0,
                },
                "created_utc": "2026-01-01T00:00:00Z",
            }
            digest = write_new_json(snapshot_path.resolve(), snapshot)
            authorized = {
                "planned_unit_id": "authorized",
                "experiment_id": "E7",
                "backend": "extraction",
                "child_attempts": 1,
                "frames": 0,
                "event_rows": 0,
                "correctness_rows": 0,
                "timing_rows": 0,
                "metadata": {"phase": "prior_work_extraction"},
            }
            authorization = run_matrix.Authorization(
                phase="base",
                snapshot_path=snapshot_path,
                snapshot_sha256=digest,
                snapshot=snapshot,
                units=(authorized,),
                component_paths=(),
            )
            run_matrix.require_authorized_unit(authorization, authorized)
            mutated = dict(authorized)
            mutated["frames"] = 1
            with self.assertRaises(ContractError):
                run_matrix.require_authorized_unit(authorization, mutated)

    def test_synthesis_and_programming_tokens_are_rejected(self) -> None:
        manifest = {
            "artifacts": [
                {"artifact_id": "executable", "path": "/tmp/fpt2026_benchmark", "sha256": "0" * 64}
            ]
        }
        with self.assertRaises(ContractError):
            run_matrix.assert_safe_child_command(
                ["/tmp/fpt2026_benchmark", "quartus_pgm", "image.gbs"], manifest, None
            )
        with self.assertRaises(ContractError):
            run_matrix.assert_safe_child_command(
                ["/tmp/fpt2026_benchmark", "compile", "-Xsprofile"], manifest, None
            )

    def test_only_two_python_extraction_children_are_allowlisted(self) -> None:
        source = inspect.getsource(run_matrix.assert_safe_child_command)
        self.assertIn("EXTRACT_REPORT.resolve()", source)
        self.assertIn("EXTRACT_PRIOR_WORK.resolve()", source)
        self.assertNotIn("shell=True", source)

    def test_e7_status_carries_exact_child_output_path_and_hash(self) -> None:
        source = inspect.getsource(run_matrix._append_extraction_statuses)
        self.assertIn('status["child_output_path"] = str(path)', source)
        self.assertIn('status["child_output_sha256"] = output_sha256', source)
        self.assertIn('status["source_evidence"] = result["source_evidence"]', source)

    def test_e3_status_carries_machine_report_evidence(self) -> None:
        source = inspect.getsource(run_matrix._append_extraction_statuses)
        self.assertIn('status["report_evidence"] = result["report_evidence"]', source)
        for field in (
            "report_tree_sha256",
            "info_ndjson_sha256",
            "quartus_ndjson_sha256",
            "device_image_sha256",
        ):
            self.assertIn(field, source)


class ReportExtractorContractTests(unittest.TestCase):
    def test_seed_parser_requires_one_explicit_seed(self) -> None:
        self.assertEqual(extract_report._seed("aoc -hardware -seed=7 -o output"), 7)
        with self.assertRaises(ContractError):
            extract_report._seed("aoc -hardware -o output")
        with self.assertRaises(ContractError):
            extract_report._seed("aoc -seed=7 -seed=3")

    def test_resource_parser_preserves_machine_values(self) -> None:
        parsed = extract_report._resource_row(
            {"name": "system", "alut": "690164", "reg": "1,376,060", "alm": "481176", "dsp": "2,384", "ram": "5,115"}
        )
        self.assertEqual(
            parsed,
            {"alut": 690164, "register": 1376060, "alm": 481176, "dsp": 2384, "ram": 5115},
        )

    def test_embedded_source_paths_only_normalize_accelerator_prefix(self) -> None:
        self.assertEqual(
            extract_report._report_relative_path(
                "/remote/build/SYCL_CKKS_PIPELINE/include_internal/SYCL_ntt.h"
            ),
            "include_internal/SYCL_ntt.h",
        )
        self.assertIsNone(extract_report._report_relative_path("/opt/intel/include/sycl.hpp"))

    def test_report_evidence_contains_current_status_schema_fields(self) -> None:
        source = inspect.getsource(extract_report.extract)
        for field in (
            "report_root_path",
            "report_tree_sha256",
            "info_ndjson_path",
            "info_ndjson_sha256",
            "quartus_ndjson_path",
            "quartus_ndjson_sha256",
            "source_revision",
            "device_image_sha256",
            "fpga_target",
            "fitter_seed",
            "kernel_clock_mhz",
            "resources",
        ):
            self.assertIn(f'"{field}"', source)

    def test_report_extractor_reads_machine_ndjson_not_html(self) -> None:
        source = inspect.getsource(extract_report)
        self.assertIn("info.ndjson", source)
        self.assertIn("quartus.ndjson", source)
        self.assertIn("file.ndjson", source)
        self.assertNotIn(".html", source.lower())


class PriorWorkExtractorContractTests(unittest.TestCase):
    def test_unpinned_source_path_is_rejected(self) -> None:
        manifest = {
            "artifacts": [
                {
                    "artifact_id": "aloha_he_paper",
                    "path": "/tmp/unapproved.pdf",
                    "sha256": extract_prior_work.EXPECTED_PAPER_SHA256,
                }
            ]
        }
        with self.assertRaises(ContractError):
            extract_prior_work.pinned_paper(manifest)

    def test_context_rows_match_e7_status_quantity_contract(self) -> None:
        row = extract_prior_work._context_payload(
            platform="platform",
            device="device",
            frequency_mhz=200,
            encryption_ms=1.87,
            decryption_ms=0.87,
            ntt_ms=0.267,
            locator_row="row",
        )
        self.assertIsInstance(row["parameter_set"], dict)
        self.assertIsInstance(row["operation_boundary"], str)
        self.assertIsInstance(row["platform"], str)
        self.assertEqual(row["classification"], "measured")
        for field in ("frequency", "latency", "throughput"):
            self.assertEqual(set(row[field]), {"value", "unit", "unavailable_reason"})
        self.assertIsNone(row["throughput"]["value"])
        self.assertTrue(row["throughput"]["unavailable_reason"])

    def test_cross_platform_speedup_is_explicitly_forbidden(self) -> None:
        source = inspect.getsource(extract_prior_work.extract)
        self.assertIn('"direct_speedup_claim_permitted": False', source)
        self.assertIn('"context_only": True', source)


class SummaryStatisticsContractTests(unittest.TestCase):
    def test_descriptive_statistics_include_all_required_quantities(self) -> None:
        stats = summarize_results.descriptive_statistics([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(
            set(stats),
            {"min", "p5", "p25", "median", "p75", "p95", "max", "mad", "iqr"},
        )
        self.assertAlmostEqual(stats["p5"], 1.15)
        self.assertAlmostEqual(stats["p25"], 1.75)
        self.assertAlmostEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["p75"], 3.25)
        self.assertAlmostEqual(stats["p95"], 3.85)
        self.assertAlmostEqual(stats["mad"], 1.0)
        self.assertAlmostEqual(stats["iqr"], 1.5)

    def test_bootstrap_is_percentile_10000_and_deterministic(self) -> None:
        first = summarize_results.percentile_bootstrap_ci(
            [1.0, 2.0, 3.0],
            summarize_results.median,
            analysis_seed=1234,
            label="test-group",
        )
        second = summarize_results.percentile_bootstrap_ci(
            [1.0, 2.0, 3.0],
            summarize_results.median,
            analysis_seed=1234,
            label="test-group",
        )
        self.assertEqual(first, second)
        self.assertEqual(first["algorithm"], "percentile-bootstrap-v1")
        self.assertEqual(first["resamples"], 10_000)
        self.assertEqual(first["confidence"], 0.95)
        self.assertIsNotNone(first["lower"])
        self.assertIsNotNone(first["upper"])

    def test_ci_requires_at_least_two_observations(self) -> None:
        ci = summarize_results.percentile_bootstrap_ci(
            [1.0],
            summarize_results.median,
            analysis_seed=7,
            label="singleton",
        )
        self.assertIsNone(ci["lower"])
        self.assertIsNone(ci["upper"])
        self.assertEqual(ci["observation_count"], 1)
        self.assertEqual(ci["unavailable_reason"], "requires_at_least_2_observations")

    def test_outliers_are_flagged_not_removed(self) -> None:
        rows = [
            {
                "experiment_id": "E2",
                "backend": "fpga",
                "boundary_id": "application_e2e_v1",
                "phase": "warm_measured",
                "case_id": "real_full_4096",
                "batch_size": 1,
                "attempt_id": "attempt",
                "status": "pass",
                "frame_count_submitted": 1,
                "frame_count_completed": 1,
                "correctness_summary": {"verified_after_timing": True},
                "timing_ns": {field: None for field in summarize_results.TIMING_FIELDS},
            }
            for _ in range(5)
        ]
        for row, value in zip(rows, [10, 10, 10, 10, 10_000]):
            row["timing_ns"]["application_e2e"] = value
        summaries = summarize_results.latency_summary_rows(rows, {}, analysis_seed=99)
        application = next(row for row in summaries if row["metric"] == "application_e2e")
        self.assertEqual(application["observation_count"], 5)
        self.assertEqual(application["successful"], 5)
        self.assertGreaterEqual(application["tukey_outlier_count"], 1)

    def test_saturation_needs_candidate_plus_two_larger_points(self) -> None:
        two_points = [
            {"batch_size": 1, "median_e2e_frames_per_second": 100.0},
            {"batch_size": 2, "median_e2e_frames_per_second": 100.0},
        ]
        self.assertIsNone(summarize_results.saturation_batch(two_points))
        three_points = two_points + [
            {"batch_size": 4, "median_e2e_frames_per_second": 99.0}
        ]
        self.assertEqual(summarize_results.saturation_batch(three_points), 1)

    def test_e8_evidence_requires_exactly_twenty_verified_observations(self) -> None:
        samples = []
        for repetition in range(20):
            samples.append(
                {
                    "sample_id": f"sample-{repetition}",
                    "experiment_id": "E8",
                    "backend": "fpga",
                    "batch_size": 1,
                    "frame_count_submitted": 1,
                    "frame_count_completed": 1,
                    "status": "pass",
                    "correctness_summary": {"verified_after_timing": True},
                    "timing_ns": {"application_e2e": 1_000_000, "graph_device": 500_000},
                }
            )
        evidence = summarize_results.e8_evidence_document(samples, required_batches=[1])
        self.assertEqual(evidence["throughput_points"][0]["observation_count"], 20)
        with self.assertRaises(ContractError):
            summarize_results.e8_evidence_document(samples[:-1], required_batches=[1])

    def test_command_and_environment_digest_definition_is_canonical(self) -> None:
        command = ["/abs/program", "latency"]
        environment = {"PATH": "/bin", "FPT2026_RUN_ID": "run"}
        self.assertEqual(
            sha256_bytes(canonical_json_bytes(command)),
            sha256_bytes(canonical_json_bytes(list(command))),
        )
        self.assertEqual(
            sha256_bytes(canonical_json_bytes(environment)),
            sha256_bytes(canonical_json_bytes(dict(reversed(list(environment.items()))))),
        )
