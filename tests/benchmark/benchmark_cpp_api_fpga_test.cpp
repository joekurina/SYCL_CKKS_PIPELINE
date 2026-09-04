#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "sycl_ckks_accelerator/SYCL_ckks_benchmark.h"

using CreateSignature = int (*)(
    const SYCLBenchmarkConfig *, const double *, const uint32_t *, const uint32_t *,
    SYCLBenchmarkSession **, size_t *, char *, size_t);
using BatchSignature = int (*)(
    SYCLBenchmarkSession *, size_t, const complex_double *, const int8_t *,
    const uint32_t *const *, const uint32_t *const *, uint32_t **, uint32_t **,
    uint32_t **, uint32_t **, SYCLBenchmarkTiming *, SYCLBenchmarkEventRecord *,
    size_t, size_t *, char *, size_t);
using CloseSignature = int (*)(SYCLBenchmarkSession *, char *, size_t);

static_assert(std::is_same_v<decltype(&SYCL_benchmark_session_create), CreateSignature>);
static_assert(std::is_same_v<decltype(&SYCL_benchmark_encrypt_batch), BatchSignature>);
static_assert(std::is_same_v<decltype(&SYCL_benchmark_session_close), CloseSignature>);
static_assert(std::is_standard_layout_v<SYCLBenchmarkConfig>);
static_assert(std::is_standard_layout_v<SYCLBenchmarkTiming>);
static_assert(std::is_standard_layout_v<SYCLBenchmarkEventRecord>);
static_assert(std::is_trivially_copyable_v<SYCLBenchmarkConfig>);
static_assert(std::is_trivially_copyable_v<SYCLBenchmarkTiming>);
static_assert(std::is_trivially_copyable_v<SYCLBenchmarkEventRecord>);
static_assert(sizeof(SYCLBenchmarkStage) == sizeof(uint32_t));
static_assert(sizeof(SYCLBenchmarkTransferKind) == sizeof(uint32_t));
static_assert(offsetof(SYCLBenchmarkConfig, abi_version) == 0);
static_assert(offsetof(SYCLBenchmarkTiming, abi_version) == 0);
static_assert(offsetof(SYCLBenchmarkEventRecord, abi_version) == 0);

/* Source-only FPGA Test: no queue, service, emulator, or hardware is used. */
int main()
{
    SYCLBenchmarkConfig config{};
    config.abi_version = SYCL_BENCHMARK_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.n = SYCL_POLY_N;
    config.num_moduli = SYCL_NUM_MODULI;
    config.max_frames = 2;

    SYCLBenchmarkTiming timing{};
    timing.abi_version = SYCL_BENCHMARK_ABI_VERSION;
    timing.struct_size = sizeof(timing);

    SYCLBenchmarkEventRecord record{};
    record.abi_version = SYCL_BENCHMARK_ABI_VERSION;
    record.struct_size = sizeof(record);

    return (config.max_frames * 27u == 54u &&
            timing.struct_size == sizeof(SYCLBenchmarkTiming) &&
            record.modulus_index == 0)
               ? 0
               : 1;
}
