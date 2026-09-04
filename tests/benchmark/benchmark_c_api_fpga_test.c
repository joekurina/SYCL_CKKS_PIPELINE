#include <stddef.h>
#include <stdint.h>

#include "sycl_ckks_accelerator/SYCL_ckks_benchmark.h"

_Static_assert(SYCL_BENCHMARK_ABI_VERSION == 1u, "unexpected benchmark ABI version");
_Static_assert(sizeof(SYCLBenchmarkStatus) == sizeof(uint32_t), "status must stay open and fixed-width");
_Static_assert(sizeof(SYCLBenchmarkStage) == sizeof(uint32_t), "stage must stay open and fixed-width");
_Static_assert(sizeof(SYCLBenchmarkTransferKind) == sizeof(uint32_t), "transfer kind must stay open and fixed-width");
_Static_assert(sizeof(SYCLBenchmarkProfilingUnavailableReason) == sizeof(uint32_t), "profiling reason must stay fixed-width");

_Static_assert(offsetof(SYCLBenchmarkConfig, abi_version) == 0, "config ABI prefix moved");
_Static_assert(offsetof(SYCLBenchmarkConfig, struct_size) > offsetof(SYCLBenchmarkConfig, abi_version), "config size prefix moved");
_Static_assert(offsetof(SYCLBenchmarkConfig, enable_profiling) > offsetof(SYCLBenchmarkConfig, max_frames), "config field order changed");
_Static_assert(sizeof(SYCLBenchmarkConfig) >= offsetof(SYCLBenchmarkConfig, enable_profiling) + sizeof(uint32_t), "config is truncated");

_Static_assert(offsetof(SYCLBenchmarkTiming, abi_version) == 0, "timing ABI prefix moved");
_Static_assert(offsetof(SYCLBenchmarkTiming, struct_size) > offsetof(SYCLBenchmarkTiming, abi_version), "timing size prefix moved");
_Static_assert(offsetof(SYCLBenchmarkTiming, d2h_profiling_available) > offsetof(SYCLBenchmarkTiming, first_entry_start_ns), "timing field order changed");
_Static_assert(offsetof(SYCLBenchmarkTiming, additive_wall_breakdown_available) > offsetof(SYCLBenchmarkTiming, d2h_profiling_available), "timing availability field order changed");
_Static_assert(offsetof(SYCLBenchmarkTiming, graph_submit_wait_wall_ns) > offsetof(SYCLBenchmarkTiming, additive_wall_breakdown_available), "timing wall field order changed");
_Static_assert(sizeof(SYCLBenchmarkTiming) >= offsetof(SYCLBenchmarkTiming, graph_submit_wait_wall_ns) + sizeof(uint64_t), "timing is truncated");

_Static_assert(offsetof(SYCLBenchmarkEventRecord, abi_version) == 0, "event ABI prefix moved");
_Static_assert(offsetof(SYCLBenchmarkEventRecord, struct_size) > offsetof(SYCLBenchmarkEventRecord, abi_version), "event size prefix moved");
_Static_assert(offsetof(SYCLBenchmarkEventRecord, unavailable_reason) > offsetof(SYCLBenchmarkEventRecord, command_end_ns), "event field order changed");
_Static_assert(sizeof(SYCLBenchmarkEventRecord) >= offsetof(SYCLBenchmarkEventRecord, unavailable_reason) + sizeof(uint32_t), "event record is truncated");

typedef int (*create_signature)(
    const SYCLBenchmarkConfig *, const double *, const uint32_t *, const uint32_t *,
    SYCLBenchmarkSession **, size_t *, char *, size_t);
typedef int (*batch_signature)(
    SYCLBenchmarkSession *, size_t, const complex_double *, const int8_t *,
    const uint32_t *const *, const uint32_t *const *, uint32_t **, uint32_t **,
    uint32_t **, uint32_t **, SYCLBenchmarkTiming *, SYCLBenchmarkEventRecord *,
    size_t, size_t *, char *, size_t);
typedef int (*close_signature)(SYCLBenchmarkSession *, char *, size_t);

/* Source-only FPGA Test: assigning each symbol enforces the pure-C ABI. */
int main(void)
{
    create_signature create_api = &SYCL_benchmark_session_create;
    batch_signature batch_api = &SYCL_benchmark_encrypt_batch;
    close_signature close_api = &SYCL_benchmark_session_close;

    if (SYCL_BENCHMARK_STAGE_H2D != 0u ||
        SYCL_BENCHMARK_STAGE_D2H != 8u ||
        SYCL_BENCHMARK_TRANSFER_NONE != 0u ||
        SYCL_BENCHMARK_TRANSFER_NTT_B != 4u) {
        return 1;
    }
    return (create_api && batch_api && close_api) ? 0 : 2;
}
