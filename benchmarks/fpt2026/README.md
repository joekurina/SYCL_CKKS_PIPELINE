# FPT 2026 CKKS Benchmark Harness

This directory contains the source, configuration, schemas, orchestration, validation, and analysis support for the FPT 2026 end-to-end CKKS benchmark campaign.

The implementation follows:

`/home/joe/Documents/Obsidian/School/Thesis/FPT 2026 Paper/Benchmarks/Benchmark_Plan.md`

## Scope

- `include/` and `src/`: standalone C/C++ benchmark driver and the audited C boundary for SEAL-Embedded.
- `scripts/`: run preparation, orchestration, validation, report extraction, prior-work extraction, and summaries.
- `schemas/`: JSON Schema Draft 2020-12 contracts for immutable planning and JSONL records.
- `configs/fpt2026-paper.json`: pinned paper configuration without runtime-derived hashes.
- `tests/`: source and contract tests. Test execution is deliberately deferred.

The accelerator API is additive. The legacy `SYCL_encrypt` entry point remains available, while `SYCL_ckks_benchmark.h` exposes a process-lifetime benchmark session API with explicit event records.

## Safety state

No configuration, compilation, linking, emulator execution, simulator execution, FPGA programming, or benchmark execution was performed while creating this harness. Runtime-derived artifact hashes, control snapshots, attempts, samples, events, correctness records, experiment status, amendments, and effective-plan snapshots must be created only by a separately authorized run.

## Non-negotiable evidence rules

- All frozen manifest paths are absolute.
- Every child attempt has one pre-control and one post-control snapshot.
- Completion text is not a correctness verdict.
- `[FPGA Test] PASS` is emitted only after active numerical verification.
- Official results use the uninstrumented seed-7 image; optional profiler diagnostics are isolated from E1-E8.
- A profiled executable is never a `-reuse-exe` anchor.
- Performance-mode FPGA frames resolve exactly 27 bounded/copy event rows; one diagnostic NTT stream resolves 39; full diagnostics resolves 51.
- E2 contains exactly 50 measured repetitions.
- E4 sustained contains exactly 10,000 resident frames.
- E5 has 80 semantic rows: 40 FPGA, 25 unaccelerated SEAL-Embedded, and 15 stock-SEAL complex references.
- `application_e2e_ns` uses one monotonic wall clock across CPU and FPGA backends.

## Status

The harness is source-complete only after static review of every file. Building and all runtime validation remain intentionally pending explicit authorization.
