#include "sycl_ckks_accelerator/SYCL_ckks_sym.h"
#include "sycl_ckks_accelerator/SYCL_ckks_benchmark.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "SYCL_common.h"
#include "SYCL_data_types.h"
#include "SYCL_pipes.h"
#include "SYCL_entry.h"
#include "SYCL_pipeline_exit.h"
#include "SYCL_ntt.h"
#include "SYCL_ifft.h"
#include "SYCL_scale_and_reduce.h"
#include "SYCL_poly_mult_neg_add.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace sycl;
using namespace sycl_ckks;

static_assert(POLY_N == SYCL_POLY_N, "public and internal polynomial degrees differ");
static_assert(NUM_MODULI == SYCL_NUM_MODULI, "public and internal modulus counts differ");
static_assert(NUM_PHYSICAL_PIPELINES == SYCL_NUM_PHYSICAL_PIPELINES,
              "public and internal physical-pipeline counts differ");

struct ModulusParams {
    double scale;
    uint32_t mod_value;
    uint32_t const_ratio[2];
    uint8_t modulus_selector;
    bool save_ntt_s;
    bool save_ntt_pte;
};

enum class ProcessLifecycle : uint32_t {
    UNUSED = 0,
    LEGACY_ACTIVE = 1,
    LEGACY_USED = 2,
    SESSION_ACTIVE = 3,
    CLOSED = 4,
    FAILED = 5
};

static std::atomic<ProcessLifecycle> process_lifecycle{ProcessLifecycle::UNUSED};

#ifdef SYCL_CKKS_BENCHMARK_PRIVATE_TEST
static std::atomic<SYCLBenchmarkPrivatePostGuardHook> private_post_guard_hook{nullptr};
static std::atomic<void*> private_post_guard_context{nullptr};
#endif

static void notify_post_guard_winner() noexcept
{
#ifdef SYCL_CKKS_BENCHMARK_PRIVATE_TEST
    SYCLBenchmarkPrivatePostGuardHook hook =
        private_post_guard_hook.load(std::memory_order_acquire);
    if (hook) {
        try {
            hook(private_post_guard_context.load(std::memory_order_acquire));
        } catch (...) {
            // A private test hook cannot alter the production lifecycle path.
        }
    }
#endif
}

static bool try_claim_process_lifecycle(ProcessLifecycle claimed_state)
{
    ProcessLifecycle expected = ProcessLifecycle::UNUSED;
    if (!process_lifecycle.compare_exchange_strong(
            expected,
            claimed_state,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return false;
    }
    notify_post_guard_winner();
    return true;
}

static const char* lifecycle_name(ProcessLifecycle state)
{
    switch (state) {
    case ProcessLifecycle::UNUSED: return "UNUSED";
    case ProcessLifecycle::LEGACY_ACTIVE: return "LEGACY_ACTIVE";
    case ProcessLifecycle::LEGACY_USED: return "LEGACY_USED";
    case ProcessLifecycle::SESSION_ACTIVE: return "SESSION_ACTIVE";
    case ProcessLifecycle::CLOSED: return "CLOSED";
    case ProcessLifecycle::FAILED: return "FAILED";
    }
    return "UNKNOWN";
}

static void pack_input(
    size_t n,
    const complex_double* encoding_buffer,
    const int8_t* error_samples,
    const uint32_t* const* secret_keys,
    const uint32_t* const* c1_polys,
    std::vector<PipelineInputBlock>& input_blocks)
{
    size_t num_blocks = n / LANES;
    input_blocks.resize(num_blocks);

    for (size_t blk = 0; blk < num_blocks; ++blk) {
        PipelineInputBlock& block = input_blocks[blk];
        pack_encoding_to_block(encoding_buffer, blk, block.encoding);
        pack_error_to_block(error_samples, blk, block.error);

        for (size_t p = 0; p < NUM_MODULI; ++p) {
            pack_scalar_to_block(secret_keys[p], blk, block.secret_key[p]);
            pack_scalar_to_block(c1_polys[p], blk, block.c1[p]);
        }
    }
}

static void unpack_u32_blocks(
    size_t n,
    const std::vector<u32x4>& blocks,
    uint32_t* out)
{
    size_t num_blocks = n / LANES;
    for (size_t blk = 0; blk < num_blocks; ++blk) {
        unpack_block_to_scalar(blocks[blk], blk, out);
    }
}

static std::vector<event> run_pipeline(
    queue& q,
    buffer<PipelineInputBlock, 1>& input_buf,
    std::array<buffer<u32x4, 1>, NUM_MODULI>& c0_bufs,
    std::array<buffer<u32x4, 1>, NUM_MODULI>& ntt_s_bufs,
    std::array<buffer<u32x4, 1>, NUM_MODULI>& ntt_pte_bufs,
    const std::array<ModulusParams, NUM_MODULI>& mod_params)
{
    try {
        std::vector<event> events;

        events.push_back(q.submit([&](handler& h) { ExitC0Kernel<0> kernel(c0_bufs[0]); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ExitC0Kernel<1> kernel(c0_bufs[1]); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ExitC0Kernel<2> kernel(c0_bufs[2]); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ExitC0Kernel<3> kernel(c0_bufs[3]); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ExitC0Kernel<4> kernel(c0_bufs[4]); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ExitC0Kernel<5> kernel(c0_bufs[5]); kernel(h); }));

        if (mod_params[0].save_ntt_s) events.push_back(q.submit([&](handler& h) { ExitNTTASKernel<0> kernel(ntt_s_bufs[0]); kernel(h); }));
        if (mod_params[1].save_ntt_s) events.push_back(q.submit([&](handler& h) { ExitNTTASKernel<1> kernel(ntt_s_bufs[1]); kernel(h); }));
        if (mod_params[2].save_ntt_s) events.push_back(q.submit([&](handler& h) { ExitNTTASKernel<2> kernel(ntt_s_bufs[2]); kernel(h); }));
        if (mod_params[3].save_ntt_s) events.push_back(q.submit([&](handler& h) { ExitNTTASKernel<3> kernel(ntt_s_bufs[3]); kernel(h); }));
        if (mod_params[4].save_ntt_s) events.push_back(q.submit([&](handler& h) { ExitNTTASKernel<4> kernel(ntt_s_bufs[4]); kernel(h); }));
        if (mod_params[5].save_ntt_s) events.push_back(q.submit([&](handler& h) { ExitNTTASKernel<5> kernel(ntt_s_bufs[5]); kernel(h); }));

        if (mod_params[0].save_ntt_pte) events.push_back(q.submit([&](handler& h) { ExitNTTBKernel<0> kernel(ntt_pte_bufs[0]); kernel(h); }));
        if (mod_params[1].save_ntt_pte) events.push_back(q.submit([&](handler& h) { ExitNTTBKernel<1> kernel(ntt_pte_bufs[1]); kernel(h); }));
        if (mod_params[2].save_ntt_pte) events.push_back(q.submit([&](handler& h) { ExitNTTBKernel<2> kernel(ntt_pte_bufs[2]); kernel(h); }));
        if (mod_params[3].save_ntt_pte) events.push_back(q.submit([&](handler& h) { ExitNTTBKernel<3> kernel(ntt_pte_bufs[3]); kernel(h); }));
        if (mod_params[4].save_ntt_pte) events.push_back(q.submit([&](handler& h) { ExitNTTBKernel<4> kernel(ntt_pte_bufs[4]); kernel(h); }));
        if (mod_params[5].save_ntt_pte) events.push_back(q.submit([&](handler& h) { ExitNTTBKernel<5> kernel(ntt_pte_bufs[5]); kernel(h); }));

        events.push_back(q.submit([&](handler& h) { PolyMultNegAddKernel<0> kernel(mod_params[0].mod_value, mod_params[0].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { PolyMultNegAddKernel<1> kernel(mod_params[1].mod_value, mod_params[1].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { PolyMultNegAddKernel<2> kernel(mod_params[2].mod_value, mod_params[2].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { PolyMultNegAddKernel<3> kernel(mod_params[3].mod_value, mod_params[3].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { PolyMultNegAddKernel<4> kernel(mod_params[4].mod_value, mod_params[4].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { PolyMultNegAddKernel<5> kernel(mod_params[5].mod_value, mod_params[5].const_ratio); kernel(h); }));

        // Imported NTT and IFFT service kernels are deliberately persistent;
        // their events are never collected into the waited vector.
        q.submit([&](handler& h) { NTTKernelA<0> kernel(mod_params[0].modulus_selector, mod_params[0].save_ntt_s); kernel(h); });
        q.submit([&](handler& h) { NTTKernelA<1> kernel(mod_params[1].modulus_selector, mod_params[1].save_ntt_s); kernel(h); });
        q.submit([&](handler& h) { NTTKernelA<2> kernel(mod_params[2].modulus_selector, mod_params[2].save_ntt_s); kernel(h); });
        q.submit([&](handler& h) { NTTKernelA<3> kernel(mod_params[3].modulus_selector, mod_params[3].save_ntt_s); kernel(h); });
        q.submit([&](handler& h) { NTTKernelA<4> kernel(mod_params[4].modulus_selector, mod_params[4].save_ntt_s); kernel(h); });
        q.submit([&](handler& h) { NTTKernelA<5> kernel(mod_params[5].modulus_selector, mod_params[5].save_ntt_s); kernel(h); });
        q.submit([&](handler& h) { NTTKernelB<0> kernel(mod_params[0].modulus_selector, mod_params[0].save_ntt_pte); kernel(h); });
        q.submit([&](handler& h) { NTTKernelB<1> kernel(mod_params[1].modulus_selector, mod_params[1].save_ntt_pte); kernel(h); });
        q.submit([&](handler& h) { NTTKernelB<2> kernel(mod_params[2].modulus_selector, mod_params[2].save_ntt_pte); kernel(h); });
        q.submit([&](handler& h) { NTTKernelB<3> kernel(mod_params[3].modulus_selector, mod_params[3].save_ntt_pte); kernel(h); });
        q.submit([&](handler& h) { NTTKernelB<4> kernel(mod_params[4].modulus_selector, mod_params[4].save_ntt_pte); kernel(h); });
        q.submit([&](handler& h) { NTTKernelB<5> kernel(mod_params[5].modulus_selector, mod_params[5].save_ntt_pte); kernel(h); });

        events.push_back(q.submit([&](handler& h) { ScaleAndReduceKernel<0> kernel(mod_params[0].scale, mod_params[0].mod_value, mod_params[0].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ScaleAndReduceKernel<1> kernel(mod_params[1].scale, mod_params[1].mod_value, mod_params[1].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ScaleAndReduceKernel<2> kernel(mod_params[2].scale, mod_params[2].mod_value, mod_params[2].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ScaleAndReduceKernel<3> kernel(mod_params[3].scale, mod_params[3].mod_value, mod_params[3].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ScaleAndReduceKernel<4> kernel(mod_params[4].scale, mod_params[4].mod_value, mod_params[4].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ScaleAndReduceKernel<5> kernel(mod_params[5].scale, mod_params[5].mod_value, mod_params[5].const_ratio); kernel(h); }));

        q.submit([&](handler& h) { IFFTKernel kernel; kernel(h); });
        events.push_back(q.submit([&](handler& h) { IFFTFanoutKernel<> kernel; kernel(h); }));
        events.push_back(q.submit([&](handler& h) { EntryKernel<> kernel(input_buf); kernel(h); }));

        return events;
    } catch (std::exception const& e) {
        std::cerr << "[SYCL_encrypt] Pipeline exception: " << e.what() << std::endl;
        std::exit(1);
    }
    return {};
}

namespace {

constexpr size_t BENCHMARK_BASE_EVENTS_PER_FRAME = 27;
constexpr size_t BENCHMARK_SAVE_EVENTS_PER_FRAME = 12;
constexpr uint64_t BENCHMARK_SESSION_MAGIC = UINT64_C(0x5359434c42454e43);
constexpr std::array<uint32_t, NUM_MODULI> BENCHMARK_MODULI = {
    1053818881u, 1054015489u, 1054212097u,
    1055260673u, 1056178177u, 1056440321u};

using BenchmarkClock = std::chrono::steady_clock;

static uint64_t elapsed_ns(BenchmarkClock::time_point begin, BenchmarkClock::time_point end)
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

static void set_error_text(char* error_message, size_t capacity, const char* message)
{
    if (!error_message || capacity == 0) {
        return;
    }
    std::snprintf(error_message, capacity, "%s", message ? message : "unknown error");
}

static int benchmark_error(
    SYCLBenchmarkStatus status,
    char* error_message,
    size_t error_message_capacity,
    const char* message)
{
    set_error_text(error_message, error_message_capacity, message);
    return static_cast<int>(status);
}

static size_t benchmark_events_per_frame(const SYCLBenchmarkConfig& config)
{
    return BENCHMARK_BASE_EVENTS_PER_FRAME +
           BENCHMARK_SAVE_EVENTS_PER_FRAME * static_cast<size_t>(config.save_ntt_s) +
           BENCHMARK_SAVE_EVENTS_PER_FRAME * static_cast<size_t>(config.save_ntt_pte);
}

static int validate_benchmark_config(
    const SYCLBenchmarkConfig* config,
    const double* scales,
    const uint32_t* mod_values,
    const uint32_t* const_ratios,
    size_t& required_capacity,
    std::array<ModulusParams, NUM_MODULI>& mod_params,
    char* error_message,
    size_t error_message_capacity)
{
    if (!config) {
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT,
            error_message,
            error_message_capacity,
            "benchmark config is null");
    }
    if (config->struct_size < sizeof(SYCLBenchmarkConfig)) {
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_ABI_MISMATCH,
            error_message,
            error_message_capacity,
            "benchmark config structure is too small");
    }
    if (config->abi_version != SYCL_BENCHMARK_ABI_VERSION) {
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_ABI_MISMATCH,
            error_message,
            error_message_capacity,
            "benchmark config ABI version mismatch");
    }
    if (config->n != POLY_N || config->n != SYCL_POLY_N) {
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT,
            error_message,
            error_message_capacity,
            "benchmark session requires polynomial degree 8192");
    }
    if (config->num_moduli != NUM_MODULI ||
        config->num_moduli != SYCL_NUM_MODULI) {
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT,
            error_message,
            error_message_capacity,
            "benchmark session requires six moduli");
    }
    if (config->max_frames == 0) {
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT,
            error_message,
            error_message_capacity,
            "benchmark max_frames must be nonzero");
    }
    if (config->save_ntt_s > 1 || config->save_ntt_pte > 1 ||
        config->enable_profiling > 1) {
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT,
            error_message,
            error_message_capacity,
            "benchmark mode fields must be zero or one");
    }
    if (!scales || !mod_values || !const_ratios) {
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT,
            error_message,
            error_message_capacity,
            "benchmark modulus parameter array is null");
    }

    const size_t events_per_frame = benchmark_events_per_frame(*config);
    if (config->max_frames > std::numeric_limits<size_t>::max() / events_per_frame ||
        config->max_frames > std::numeric_limits<size_t>::max() / config->n) {
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT,
            error_message,
            error_message_capacity,
            "benchmark max_frames overflows the frame-major layout");
    }
    required_capacity = config->max_frames * events_per_frame;

    for (size_t p = 0; p < NUM_MODULI; ++p) {
        if (mod_values[p] != BENCHMARK_MODULI[p]) {
            return benchmark_error(
                SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT,
                error_message,
                error_message_capacity,
                "benchmark moduli must contain the exact ordered six-prime 8K chain");
        }
        uint8_t selector = get_modulus_selector(mod_values[p]);
        if (selector == 0xff || selector != p) {
            return benchmark_error(
                SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT,
                error_message,
                error_message_capacity,
                "benchmark modulus is not supported by the six-channel 8K NTT RTL");
        }

        uint32_t expected_cr0 = 0;
        uint32_t expected_cr1 = 0;
        if (!get_barrett_constants(mod_values[p], expected_cr0, expected_cr1) ||
            expected_cr0 != const_ratios[p * 2] ||
            expected_cr1 != const_ratios[p * 2 + 1]) {
            return benchmark_error(
                SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT,
                error_message,
                error_message_capacity,
                "benchmark Barrett constants do not match the modulus");
        }

        mod_params[p].scale = scales[p];
        mod_params[p].mod_value = mod_values[p];
        mod_params[p].const_ratio[0] = const_ratios[p * 2];
        mod_params[p].const_ratio[1] = const_ratios[p * 2 + 1];
        mod_params[p].modulus_selector = selector;
        mod_params[p].save_ntt_s = config->save_ntt_s != 0;
        mod_params[p].save_ntt_pte = config->save_ntt_pte != 0;
    }

    return SYCL_BENCHMARK_STATUS_SUCCESS;
}

static queue make_benchmark_queue()
{
#if FPGA_SIMULATOR
    auto selector = ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = ext::intel::fpga_selector_v;
#else
    auto selector = ext::intel::fpga_emulator_selector_v;
#endif
    // A default SYCL queue is out-of-order. in_order must not be requested here:
    // a persistent service command would otherwise block every later command.
    return queue{selector, property::queue::enable_profiling()};
}

struct BenchmarkFrameBuffers {
    std::vector<PipelineInputBlock> packed_input;
    std::array<std::vector<u32x4>, NUM_MODULI> c0_host;
    std::array<std::vector<u32x4>, NUM_MODULI> ntt_s_host;
    std::array<std::vector<u32x4>, NUM_MODULI> ntt_pte_host;

    buffer<PipelineInputBlock, 1> input_buf;
    std::array<buffer<u32x4, 1>, NUM_MODULI> c0_bufs;
    std::array<buffer<u32x4, 1>, NUM_MODULI> ntt_s_bufs;
    std::array<buffer<u32x4, 1>, NUM_MODULI> ntt_pte_bufs;

    BenchmarkFrameBuffers()
        : packed_input(NUM_BLOCKS),
          input_buf(range<1>(NUM_BLOCKS)),
          c0_bufs{
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS))},
          ntt_s_bufs{
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS))},
          ntt_pte_bufs{
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS)),
              buffer<u32x4, 1>(range<1>(NUM_BLOCKS))}
    {
        // Range-only buffers have no host final-data destination. Explicit copy
        // commands below are therefore the only benchmark-path transfers.
        for (size_t p = 0; p < NUM_MODULI; ++p) {
            c0_host[p].resize(NUM_BLOCKS);
            ntt_s_host[p].resize(NUM_BLOCKS);
            ntt_pte_host[p].resize(NUM_BLOCKS);
        }
    }
};

struct PreviousBoundedEvents {
    std::optional<event> entry;
    std::optional<event> ifft_fanout;
    std::array<std::optional<event>, NUM_MODULI> scale_reduce;
    std::array<std::optional<event>, NUM_MODULI> poly_mult_neg_add;
    std::array<std::optional<event>, NUM_MODULI> exit_c0;
    std::array<std::optional<event>, NUM_MODULI> exit_ntt_s;
    std::array<std::optional<event>, NUM_MODULI> exit_ntt_pte;
};

struct BenchmarkSessionRuntime {
    queue q;
    SYCLBenchmarkConfig config;
    std::array<ModulusParams, NUM_MODULI> mod_params;
    std::vector<std::unique_ptr<BenchmarkFrameBuffers>> frames;
    PreviousBoundedEvents previous;

    BenchmarkSessionRuntime(
        const SYCLBenchmarkConfig& session_config,
        const std::array<ModulusParams, NUM_MODULI>& params)
        : q(make_benchmark_queue()), config(session_config), mod_params(params)
    {
        frames.reserve(config.max_frames);
        for (size_t frame = 0; frame < config.max_frames; ++frame) {
            frames.emplace_back(std::make_unique<BenchmarkFrameBuffers>());
        }
    }
};

struct SubmittedFrameEvents {
    event h2d;
    event entry;
    event ifft_fanout;
    std::array<event, NUM_MODULI> scale_reduce;
    std::array<event, NUM_MODULI> poly_mult_neg_add;
    std::array<event, NUM_MODULI> exit_c0;
    std::array<event, NUM_MODULI> exit_ntt_s;
    std::array<event, NUM_MODULI> exit_ntt_pte;
    std::array<event, NUM_MODULI> d2h_c0;
    std::array<event, NUM_MODULI> d2h_ntt_s;
    std::array<event, NUM_MODULI> d2h_ntt_pte;
};

struct NamedBenchmarkEvent {
    event command;
    size_t frame_index;
    int32_t modulus_index;
    SYCLBenchmarkStage stage;
    SYCLBenchmarkTransferKind transfer_kind;
    uint64_t byte_count;
};

struct EventProfiling {
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
    uint32_t available = 0;
    uint32_t unavailable_reason = SYCL_BENCHMARK_PROFILING_DISABLED;
};

} // namespace

struct SYCLBenchmarkSession {
    uint64_t magic = BENCHMARK_SESSION_MAGIC;
    std::mutex mutex;
    BenchmarkSessionRuntime* runtime = nullptr;
    bool closed = false;
    bool failed = false;
};

namespace {

static void submit_persistent_services_once(BenchmarkSessionRuntime& runtime)
{
    queue& q = runtime.q;
    const auto& mod_params = runtime.mod_params;

    // Persistent service events are intentionally discarded and are never
    // waited. Only bounded and explicit-copy events are retained by a batch.
    q.submit([&](handler& h) { IFFTKernel kernel; kernel(h); });
    q.submit([&](handler& h) { NTTKernelA<0> kernel(mod_params[0].modulus_selector, mod_params[0].save_ntt_s); kernel(h); });
    q.submit([&](handler& h) { NTTKernelA<1> kernel(mod_params[1].modulus_selector, mod_params[1].save_ntt_s); kernel(h); });
    q.submit([&](handler& h) { NTTKernelA<2> kernel(mod_params[2].modulus_selector, mod_params[2].save_ntt_s); kernel(h); });
    q.submit([&](handler& h) { NTTKernelA<3> kernel(mod_params[3].modulus_selector, mod_params[3].save_ntt_s); kernel(h); });
    q.submit([&](handler& h) { NTTKernelA<4> kernel(mod_params[4].modulus_selector, mod_params[4].save_ntt_s); kernel(h); });
    q.submit([&](handler& h) { NTTKernelA<5> kernel(mod_params[5].modulus_selector, mod_params[5].save_ntt_s); kernel(h); });
    q.submit([&](handler& h) { NTTKernelB<0> kernel(mod_params[0].modulus_selector, mod_params[0].save_ntt_pte); kernel(h); });
    q.submit([&](handler& h) { NTTKernelB<1> kernel(mod_params[1].modulus_selector, mod_params[1].save_ntt_pte); kernel(h); });
    q.submit([&](handler& h) { NTTKernelB<2> kernel(mod_params[2].modulus_selector, mod_params[2].save_ntt_pte); kernel(h); });
    q.submit([&](handler& h) { NTTKernelB<3> kernel(mod_params[3].modulus_selector, mod_params[3].save_ntt_pte); kernel(h); });
    q.submit([&](handler& h) { NTTKernelB<4> kernel(mod_params[4].modulus_selector, mod_params[4].save_ntt_pte); kernel(h); });
    q.submit([&](handler& h) { NTTKernelB<5> kernel(mod_params[5].modulus_selector, mod_params[5].save_ntt_pte); kernel(h); });
}

template <typename Submitter>
static event submit_after_previous(
    queue& q,
    const std::optional<event>& previous,
    Submitter&& submitter)
{
    return q.submit([&](handler& h) {
        if (previous) {
            h.depends_on(*previous);
        }
        submitter(h);
    });
}

static SubmittedFrameEvents submit_one_bounded_frame(
    BenchmarkSessionRuntime& runtime,
    BenchmarkFrameBuffers& frame)
{
    queue& q = runtime.q;
    auto& previous = runtime.previous;
    const auto& params = runtime.mod_params;
    SubmittedFrameEvents submitted;

    // Preserve the original graph's consumer-before-producer submission order.
    submitted.exit_c0[0] = submit_after_previous(q, previous.exit_c0[0], [&](handler& h) { ExitC0Kernel<0> kernel(frame.c0_bufs[0]); kernel(h); });
    submitted.exit_c0[1] = submit_after_previous(q, previous.exit_c0[1], [&](handler& h) { ExitC0Kernel<1> kernel(frame.c0_bufs[1]); kernel(h); });
    submitted.exit_c0[2] = submit_after_previous(q, previous.exit_c0[2], [&](handler& h) { ExitC0Kernel<2> kernel(frame.c0_bufs[2]); kernel(h); });
    submitted.exit_c0[3] = submit_after_previous(q, previous.exit_c0[3], [&](handler& h) { ExitC0Kernel<3> kernel(frame.c0_bufs[3]); kernel(h); });
    submitted.exit_c0[4] = submit_after_previous(q, previous.exit_c0[4], [&](handler& h) { ExitC0Kernel<4> kernel(frame.c0_bufs[4]); kernel(h); });
    submitted.exit_c0[5] = submit_after_previous(q, previous.exit_c0[5], [&](handler& h) { ExitC0Kernel<5> kernel(frame.c0_bufs[5]); kernel(h); });

    if (runtime.config.save_ntt_s) {
        submitted.exit_ntt_s[0] = submit_after_previous(q, previous.exit_ntt_s[0], [&](handler& h) { ExitNTTASKernel<0> kernel(frame.ntt_s_bufs[0]); kernel(h); });
        submitted.exit_ntt_s[1] = submit_after_previous(q, previous.exit_ntt_s[1], [&](handler& h) { ExitNTTASKernel<1> kernel(frame.ntt_s_bufs[1]); kernel(h); });
        submitted.exit_ntt_s[2] = submit_after_previous(q, previous.exit_ntt_s[2], [&](handler& h) { ExitNTTASKernel<2> kernel(frame.ntt_s_bufs[2]); kernel(h); });
        submitted.exit_ntt_s[3] = submit_after_previous(q, previous.exit_ntt_s[3], [&](handler& h) { ExitNTTASKernel<3> kernel(frame.ntt_s_bufs[3]); kernel(h); });
        submitted.exit_ntt_s[4] = submit_after_previous(q, previous.exit_ntt_s[4], [&](handler& h) { ExitNTTASKernel<4> kernel(frame.ntt_s_bufs[4]); kernel(h); });
        submitted.exit_ntt_s[5] = submit_after_previous(q, previous.exit_ntt_s[5], [&](handler& h) { ExitNTTASKernel<5> kernel(frame.ntt_s_bufs[5]); kernel(h); });
    }
    if (runtime.config.save_ntt_pte) {
        submitted.exit_ntt_pte[0] = submit_after_previous(q, previous.exit_ntt_pte[0], [&](handler& h) { ExitNTTBKernel<0> kernel(frame.ntt_pte_bufs[0]); kernel(h); });
        submitted.exit_ntt_pte[1] = submit_after_previous(q, previous.exit_ntt_pte[1], [&](handler& h) { ExitNTTBKernel<1> kernel(frame.ntt_pte_bufs[1]); kernel(h); });
        submitted.exit_ntt_pte[2] = submit_after_previous(q, previous.exit_ntt_pte[2], [&](handler& h) { ExitNTTBKernel<2> kernel(frame.ntt_pte_bufs[2]); kernel(h); });
        submitted.exit_ntt_pte[3] = submit_after_previous(q, previous.exit_ntt_pte[3], [&](handler& h) { ExitNTTBKernel<3> kernel(frame.ntt_pte_bufs[3]); kernel(h); });
        submitted.exit_ntt_pte[4] = submit_after_previous(q, previous.exit_ntt_pte[4], [&](handler& h) { ExitNTTBKernel<4> kernel(frame.ntt_pte_bufs[4]); kernel(h); });
        submitted.exit_ntt_pte[5] = submit_after_previous(q, previous.exit_ntt_pte[5], [&](handler& h) { ExitNTTBKernel<5> kernel(frame.ntt_pte_bufs[5]); kernel(h); });
    }

    submitted.poly_mult_neg_add[0] = submit_after_previous(q, previous.poly_mult_neg_add[0], [&](handler& h) { PolyMultNegAddKernel<0> kernel(params[0].mod_value, params[0].const_ratio); kernel(h); });
    submitted.poly_mult_neg_add[1] = submit_after_previous(q, previous.poly_mult_neg_add[1], [&](handler& h) { PolyMultNegAddKernel<1> kernel(params[1].mod_value, params[1].const_ratio); kernel(h); });
    submitted.poly_mult_neg_add[2] = submit_after_previous(q, previous.poly_mult_neg_add[2], [&](handler& h) { PolyMultNegAddKernel<2> kernel(params[2].mod_value, params[2].const_ratio); kernel(h); });
    submitted.poly_mult_neg_add[3] = submit_after_previous(q, previous.poly_mult_neg_add[3], [&](handler& h) { PolyMultNegAddKernel<3> kernel(params[3].mod_value, params[3].const_ratio); kernel(h); });
    submitted.poly_mult_neg_add[4] = submit_after_previous(q, previous.poly_mult_neg_add[4], [&](handler& h) { PolyMultNegAddKernel<4> kernel(params[4].mod_value, params[4].const_ratio); kernel(h); });
    submitted.poly_mult_neg_add[5] = submit_after_previous(q, previous.poly_mult_neg_add[5], [&](handler& h) { PolyMultNegAddKernel<5> kernel(params[5].mod_value, params[5].const_ratio); kernel(h); });

    submitted.scale_reduce[0] = submit_after_previous(q, previous.scale_reduce[0], [&](handler& h) { ScaleAndReduceKernel<0> kernel(params[0].scale, params[0].mod_value, params[0].const_ratio); kernel(h); });
    submitted.scale_reduce[1] = submit_after_previous(q, previous.scale_reduce[1], [&](handler& h) { ScaleAndReduceKernel<1> kernel(params[1].scale, params[1].mod_value, params[1].const_ratio); kernel(h); });
    submitted.scale_reduce[2] = submit_after_previous(q, previous.scale_reduce[2], [&](handler& h) { ScaleAndReduceKernel<2> kernel(params[2].scale, params[2].mod_value, params[2].const_ratio); kernel(h); });
    submitted.scale_reduce[3] = submit_after_previous(q, previous.scale_reduce[3], [&](handler& h) { ScaleAndReduceKernel<3> kernel(params[3].scale, params[3].mod_value, params[3].const_ratio); kernel(h); });
    submitted.scale_reduce[4] = submit_after_previous(q, previous.scale_reduce[4], [&](handler& h) { ScaleAndReduceKernel<4> kernel(params[4].scale, params[4].mod_value, params[4].const_ratio); kernel(h); });
    submitted.scale_reduce[5] = submit_after_previous(q, previous.scale_reduce[5], [&](handler& h) { ScaleAndReduceKernel<5> kernel(params[5].scale, params[5].mod_value, params[5].const_ratio); kernel(h); });

    submitted.ifft_fanout = submit_after_previous(q, previous.ifft_fanout, [&](handler& h) { IFFTFanoutKernel<> kernel; kernel(h); });
    submitted.entry = submit_after_previous(q, previous.entry, [&](handler& h) { EntryKernel<> kernel(frame.input_buf); kernel(h); });

    previous.entry = submitted.entry;
    previous.ifft_fanout = submitted.ifft_fanout;
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        previous.scale_reduce[p] = submitted.scale_reduce[p];
        previous.poly_mult_neg_add[p] = submitted.poly_mult_neg_add[p];
        previous.exit_c0[p] = submitted.exit_c0[p];
        if (runtime.config.save_ntt_s) {
            previous.exit_ntt_s[p] = submitted.exit_ntt_s[p];
        }
        if (runtime.config.save_ntt_pte) {
            previous.exit_ntt_pte[p] = submitted.exit_ntt_pte[p];
        }
    }

    return submitted;
}

static void pack_benchmark_frame(
    size_t n,
    size_t frame_index,
    const complex_double* encoding_buffers,
    const int8_t* error_samples,
    const uint32_t* const* secret_keys,
    const uint32_t* const* uniform_polys,
    BenchmarkFrameBuffers& frame)
{
    const size_t frame_offset = frame_index * n;
    std::array<const uint32_t*, NUM_MODULI> frame_secret_keys;
    std::array<const uint32_t*, NUM_MODULI> frame_uniform_polys;
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        frame_secret_keys[p] = secret_keys[p] + frame_offset;
        frame_uniform_polys[p] = uniform_polys[p] + frame_offset;
    }
    pack_input(
        n,
        encoding_buffers + frame_offset,
        error_samples + frame_offset,
        frame_secret_keys.data(),
        frame_uniform_polys.data(),
        frame.packed_input);
}

static event copy_batch_to_device(queue& q, BenchmarkFrameBuffers& frame)
{
    return q.submit([&](handler& h) {
        auto destination = frame.input_buf.get_access<access::mode::write>(h);
        h.copy(frame.packed_input.data(), destination);
    });
}

template <size_t P>
static event copy_c0_to_host(
    queue& q,
    BenchmarkFrameBuffers& frame,
    const event& exit_event)
{
    return q.submit([&](handler& h) {
        h.depends_on(exit_event);
        auto source = frame.c0_bufs[P].get_access<access::mode::read>(h);
        h.copy(source, frame.c0_host[P].data());
    });
}

template <size_t P>
static event copy_ntt_s_to_host(
    queue& q,
    BenchmarkFrameBuffers& frame,
    const event& exit_event)
{
    return q.submit([&](handler& h) {
        h.depends_on(exit_event);
        auto source = frame.ntt_s_bufs[P].get_access<access::mode::read>(h);
        h.copy(source, frame.ntt_s_host[P].data());
    });
}

template <size_t P>
static event copy_ntt_pte_to_host(
    queue& q,
    BenchmarkFrameBuffers& frame,
    const event& exit_event)
{
    return q.submit([&](handler& h) {
        h.depends_on(exit_event);
        auto source = frame.ntt_pte_bufs[P].get_access<access::mode::read>(h);
        h.copy(source, frame.ntt_pte_host[P].data());
    });
}

static void copy_batch_to_host(
    BenchmarkSessionRuntime& runtime,
    BenchmarkFrameBuffers& frame,
    SubmittedFrameEvents& events)
{
    queue& q = runtime.q;
    events.d2h_c0[0] = copy_c0_to_host<0>(q, frame, events.exit_c0[0]);
    events.d2h_c0[1] = copy_c0_to_host<1>(q, frame, events.exit_c0[1]);
    events.d2h_c0[2] = copy_c0_to_host<2>(q, frame, events.exit_c0[2]);
    events.d2h_c0[3] = copy_c0_to_host<3>(q, frame, events.exit_c0[3]);
    events.d2h_c0[4] = copy_c0_to_host<4>(q, frame, events.exit_c0[4]);
    events.d2h_c0[5] = copy_c0_to_host<5>(q, frame, events.exit_c0[5]);

    if (runtime.config.save_ntt_s) {
        events.d2h_ntt_s[0] = copy_ntt_s_to_host<0>(q, frame, events.exit_ntt_s[0]);
        events.d2h_ntt_s[1] = copy_ntt_s_to_host<1>(q, frame, events.exit_ntt_s[1]);
        events.d2h_ntt_s[2] = copy_ntt_s_to_host<2>(q, frame, events.exit_ntt_s[2]);
        events.d2h_ntt_s[3] = copy_ntt_s_to_host<3>(q, frame, events.exit_ntt_s[3]);
        events.d2h_ntt_s[4] = copy_ntt_s_to_host<4>(q, frame, events.exit_ntt_s[4]);
        events.d2h_ntt_s[5] = copy_ntt_s_to_host<5>(q, frame, events.exit_ntt_s[5]);
    }
    if (runtime.config.save_ntt_pte) {
        events.d2h_ntt_pte[0] = copy_ntt_pte_to_host<0>(q, frame, events.exit_ntt_pte[0]);
        events.d2h_ntt_pte[1] = copy_ntt_pte_to_host<1>(q, frame, events.exit_ntt_pte[1]);
        events.d2h_ntt_pte[2] = copy_ntt_pte_to_host<2>(q, frame, events.exit_ntt_pte[2]);
        events.d2h_ntt_pte[3] = copy_ntt_pte_to_host<3>(q, frame, events.exit_ntt_pte[3]);
        events.d2h_ntt_pte[4] = copy_ntt_pte_to_host<4>(q, frame, events.exit_ntt_pte[4]);
        events.d2h_ntt_pte[5] = copy_ntt_pte_to_host<5>(q, frame, events.exit_ntt_pte[5]);
    }
}

static void append_named_event(
    std::vector<NamedBenchmarkEvent>& named,
    const event& command,
    size_t frame_index,
    int32_t modulus_index,
    SYCLBenchmarkStage stage,
    SYCLBenchmarkTransferKind transfer_kind,
    uint64_t byte_count)
{
    named.push_back(NamedBenchmarkEvent{
        command,
        frame_index,
        modulus_index,
        stage,
        transfer_kind,
        byte_count});
}

static void append_named_frame_events(
    const SYCLBenchmarkConfig& config,
    size_t frame_index,
    const SubmittedFrameEvents& events,
    std::vector<NamedBenchmarkEvent>& named)
{
    const uint64_t input_bytes = static_cast<uint64_t>(NUM_BLOCKS) * sizeof(PipelineInputBlock);
    const uint64_t output_bytes = static_cast<uint64_t>(NUM_BLOCKS) * sizeof(u32x4);

    append_named_event(named, events.h2d, frame_index, -1, SYCL_BENCHMARK_STAGE_H2D, SYCL_BENCHMARK_TRANSFER_PACKED_INPUT, input_bytes);
    append_named_event(named, events.entry, frame_index, -1, SYCL_BENCHMARK_STAGE_ENTRY, SYCL_BENCHMARK_TRANSFER_NONE, 0);
    append_named_event(named, events.ifft_fanout, frame_index, -1, SYCL_BENCHMARK_STAGE_IFFT_FANOUT, SYCL_BENCHMARK_TRANSFER_NONE, 0);
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        append_named_event(named, events.scale_reduce[p], frame_index, static_cast<int32_t>(p), SYCL_BENCHMARK_STAGE_SCALE_REDUCE, SYCL_BENCHMARK_TRANSFER_NONE, 0);
    }
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        append_named_event(named, events.poly_mult_neg_add[p], frame_index, static_cast<int32_t>(p), SYCL_BENCHMARK_STAGE_POLY_MULT_NEG_ADD, SYCL_BENCHMARK_TRANSFER_NONE, 0);
    }
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        append_named_event(named, events.exit_c0[p], frame_index, static_cast<int32_t>(p), SYCL_BENCHMARK_STAGE_EXIT_C0, SYCL_BENCHMARK_TRANSFER_NONE, 0);
    }
    if (config.save_ntt_s) {
        for (size_t p = 0; p < NUM_MODULI; ++p) {
            append_named_event(named, events.exit_ntt_s[p], frame_index, static_cast<int32_t>(p), SYCL_BENCHMARK_STAGE_EXIT_NTT_A, SYCL_BENCHMARK_TRANSFER_NONE, 0);
        }
    }
    if (config.save_ntt_pte) {
        for (size_t p = 0; p < NUM_MODULI; ++p) {
            append_named_event(named, events.exit_ntt_pte[p], frame_index, static_cast<int32_t>(p), SYCL_BENCHMARK_STAGE_EXIT_NTT_B, SYCL_BENCHMARK_TRANSFER_NONE, 0);
        }
    }
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        append_named_event(named, events.d2h_c0[p], frame_index, static_cast<int32_t>(p), SYCL_BENCHMARK_STAGE_D2H, SYCL_BENCHMARK_TRANSFER_C0, output_bytes);
    }
    if (config.save_ntt_s) {
        for (size_t p = 0; p < NUM_MODULI; ++p) {
            append_named_event(named, events.d2h_ntt_s[p], frame_index, static_cast<int32_t>(p), SYCL_BENCHMARK_STAGE_D2H, SYCL_BENCHMARK_TRANSFER_NTT_A, output_bytes);
        }
    }
    if (config.save_ntt_pte) {
        for (size_t p = 0; p < NUM_MODULI; ++p) {
            append_named_event(named, events.d2h_ntt_pte[p], frame_index, static_cast<int32_t>(p), SYCL_BENCHMARK_STAGE_D2H, SYCL_BENCHMARK_TRANSFER_NTT_B, output_bytes);
        }
    }
}

static EventProfiling query_event_profiling(const event& command, bool enabled)
{
    EventProfiling result;
    if (!enabled) {
        return result;
    }

    try {
        result.start_ns = command.get_profiling_info<info::event_profiling::command_start>();
        result.end_ns = command.get_profiling_info<info::event_profiling::command_end>();
        if (result.end_ns < result.start_ns) {
            result.start_ns = 0;
            result.end_ns = 0;
            result.unavailable_reason = SYCL_BENCHMARK_PROFILING_QUERY_FAILED;
            return result;
        }
        result.available = 1;
        result.unavailable_reason = SYCL_BENCHMARK_PROFILING_AVAILABLE;
    } catch (...) {
        result.start_ns = 0;
        result.end_ns = 0;
        result.available = 0;
        result.unavailable_reason = SYCL_BENCHMARK_PROFILING_QUERY_FAILED;
    }
    return result;
}

static void collect_named_profiling(
    const std::vector<NamedBenchmarkEvent>& named,
    bool profiling_enabled,
    SYCLBenchmarkEventRecord* records)
{
    for (size_t i = 0; i < named.size(); ++i) {
        const EventProfiling profiling =
            query_event_profiling(named[i].command, profiling_enabled);
        records[i].abi_version = SYCL_BENCHMARK_ABI_VERSION;
        records[i].struct_size = sizeof(SYCLBenchmarkEventRecord);
        records[i].frame_index = named[i].frame_index;
        records[i].modulus_index = named[i].modulus_index;
        records[i].stage = named[i].stage;
        records[i].transfer_kind = named[i].transfer_kind;
        records[i].byte_count = named[i].byte_count;
        records[i].command_start_ns = profiling.start_ns;
        records[i].command_end_ns = profiling.end_ns;
        records[i].profiling_available = profiling.available;
        records[i].unavailable_reason = profiling.unavailable_reason;
    }
}

static void initialize_timing(SYCLBenchmarkTiming& timing)
{
    timing.abi_version = SYCL_BENCHMARK_ABI_VERSION;
    timing.struct_size = sizeof(SYCLBenchmarkTiming);
    timing.pack_wall_ns = 0;
    timing.h2d_wall_ns = 0;
    timing.h2d_device_ns = 0;
    timing.graph_device_ns = 0;
    timing.d2h_device_ns = 0;
    timing.d2h_wall_ns = 0;
    timing.unpack_wall_ns = 0;
    timing.accelerator_api_wall_ns = 0;
    timing.first_entry_start_ns = 0;
    timing.last_c0_exit_end_ns = 0;
    timing.h2d_profiling_available = 0;
    timing.graph_profiling_available = 0;
    timing.d2h_profiling_available = 0;
    timing.additive_wall_breakdown_available = 0;
    timing.graph_submit_wait_wall_ns = 0;
}

static void aggregate_timing_from_records(
    const SYCLBenchmarkEventRecord* records,
    size_t count,
    SYCLBenchmarkTiming& timing)
{
    uint64_t first_h2d_start = std::numeric_limits<uint64_t>::max();
    uint64_t last_h2d_end = 0;
    uint64_t first_d2h_start = std::numeric_limits<uint64_t>::max();
    uint64_t last_d2h_end = 0;
    uint64_t first_entry_start = std::numeric_limits<uint64_t>::max();
    uint64_t last_c0_exit_end = 0;
    bool h2d_found = false;
    bool d2h_found = false;
    bool entry_found = false;
    bool c0_found = false;
    bool h2d_available = true;
    bool d2h_available = true;
    bool graph_available = true;

    for (size_t i = 0; i < count; ++i) {
        const auto& record = records[i];
        if (record.stage == SYCL_BENCHMARK_STAGE_H2D) {
            h2d_found = true;
            h2d_available = h2d_available && record.profiling_available != 0;
            if (record.profiling_available) {
                first_h2d_start = std::min(first_h2d_start, record.command_start_ns);
                last_h2d_end = std::max(last_h2d_end, record.command_end_ns);
            }
        } else if (record.stage == SYCL_BENCHMARK_STAGE_D2H) {
            d2h_found = true;
            d2h_available = d2h_available && record.profiling_available != 0;
            if (record.profiling_available) {
                first_d2h_start = std::min(first_d2h_start, record.command_start_ns);
                last_d2h_end = std::max(last_d2h_end, record.command_end_ns);
            }
        } else if (record.stage == SYCL_BENCHMARK_STAGE_ENTRY) {
            entry_found = true;
            graph_available = graph_available && record.profiling_available != 0;
            if (record.profiling_available) {
                first_entry_start = std::min(first_entry_start, record.command_start_ns);
            }
        } else if (record.stage == SYCL_BENCHMARK_STAGE_EXIT_C0) {
            c0_found = true;
            graph_available = graph_available && record.profiling_available != 0;
            if (record.profiling_available) {
                last_c0_exit_end = std::max(last_c0_exit_end, record.command_end_ns);
            }
        }
    }

    if (h2d_found && h2d_available && last_h2d_end >= first_h2d_start) {
        timing.h2d_profiling_available = 1;
        timing.h2d_device_ns = last_h2d_end - first_h2d_start;
    }
    if (d2h_found && d2h_available && last_d2h_end >= first_d2h_start) {
        timing.d2h_profiling_available = 1;
        timing.d2h_device_ns = last_d2h_end - first_d2h_start;
    }
    if (entry_found && c0_found && graph_available &&
        last_c0_exit_end >= first_entry_start) {
        timing.graph_profiling_available = 1;
        timing.first_entry_start_ns = first_entry_start;
        timing.last_c0_exit_end_ns = last_c0_exit_end;
        timing.graph_device_ns = last_c0_exit_end - first_entry_start;
    }
}

static int validate_batch_arguments(
    SYCLBenchmarkSession* session,
    size_t frame_count,
    const complex_double* encoding_buffers,
    const int8_t* error_samples,
    const uint32_t* const* secret_keys,
    const uint32_t* const* uniform_polys,
    uint32_t** c0_outputs,
    uint32_t** c1_outputs,
    uint32_t** s_save,
    uint32_t** ntt_pte_outputs,
    SYCLBenchmarkTiming* timing,
    SYCLBenchmarkEventRecord* event_records,
    size_t event_record_capacity,
    size_t* event_records_written,
    size_t& required_records,
    char* error_message,
    size_t error_message_capacity)
{
    if (!session || session->magic != BENCHMARK_SESSION_MAGIC || !session->runtime) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "benchmark session is invalid");
    }
    if (!event_records_written) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "event_records_written is null");
    }
    *event_records_written = 0;
    if (session->closed || session->failed ||
        process_lifecycle.load(std::memory_order_acquire) != ProcessLifecycle::SESSION_ACTIVE) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_LIFECYCLE_ERROR, error_message, error_message_capacity, "benchmark session is not active");
    }
    if (frame_count == 0 || frame_count > session->runtime->config.max_frames) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "frame_count is outside the preallocated session capacity");
    }
    if (!encoding_buffers || !error_samples || !secret_keys || !uniform_polys ||
        !c0_outputs || !c1_outputs) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "required frame-major batch array is null");
    }
    if (session->runtime->config.save_ntt_s && !s_save) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "NTT-A output array is required by the session mode");
    }
    if (session->runtime->config.save_ntt_pte && !ntt_pte_outputs) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "NTT-B output array is required by the session mode");
    }
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        if (!secret_keys[p] || !uniform_polys[p] || !c0_outputs[p] || !c1_outputs[p]) {
            return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "required per-modulus frame-major array is null");
        }
        if (session->runtime->config.save_ntt_s && !s_save[p]) {
            return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "required per-modulus NTT-A output is null");
        }
        if (session->runtime->config.save_ntt_pte && !ntt_pte_outputs[p]) {
            return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "required per-modulus NTT-B output is null");
        }
    }
    if (!timing) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "benchmark timing output is null");
    }
    if (timing->struct_size < sizeof(SYCLBenchmarkTiming) ||
        timing->abi_version != SYCL_BENCHMARK_ABI_VERSION) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_ABI_MISMATCH, error_message, error_message_capacity, "benchmark timing ABI or structure size is invalid");
    }

    required_records = frame_count * benchmark_events_per_frame(session->runtime->config);
    if (!event_records || event_record_capacity < required_records) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "event-record capacity is insufficient");
    }
    for (size_t i = 0; i < required_records; ++i) {
        if (event_records[i].struct_size < sizeof(SYCLBenchmarkEventRecord) ||
            event_records[i].abi_version != SYCL_BENCHMARK_ABI_VERSION) {
            return benchmark_error(SYCL_BENCHMARK_STATUS_ABI_MISMATCH, error_message, error_message_capacity, "event-record ABI or structure size is invalid");
        }
    }
    return SYCL_BENCHMARK_STATUS_SUCCESS;
}

static void mark_session_failed(SYCLBenchmarkSession& session)
{
    session.failed = true;
    ProcessLifecycle expected = ProcessLifecycle::SESSION_ACTIVE;
    process_lifecycle.compare_exchange_strong(
        expected,
        ProcessLifecycle::FAILED,
        std::memory_order_acq_rel,
        std::memory_order_acquire);
}

} // namespace

static int benchmark_session_create_impl(
    const SYCLBenchmarkConfig* config,
    const double* scales,
    const uint32_t* mod_values,
    const uint32_t* const_ratios,
    SYCLBenchmarkSession** session,
    size_t* required_event_record_capacity,
    char* error_message,
    size_t error_message_capacity)
{
    set_error_text(error_message, error_message_capacity, "");
    if (!session || !required_event_record_capacity) {
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT,
            error_message,
            error_message_capacity,
            "session and required-event-capacity outputs are required");
    }
    *session = nullptr;
    *required_event_record_capacity = 0;

    size_t required_capacity = 0;
    std::array<ModulusParams, NUM_MODULI> mod_params;
    int status = validate_benchmark_config(
        config,
        scales,
        mod_values,
        const_ratios,
        required_capacity,
        mod_params,
        error_message,
        error_message_capacity);
    if (status != SYCL_BENCHMARK_STATUS_SUCCESS) {
        return status;
    }
    *required_event_record_capacity = required_capacity;

    if (!try_claim_process_lifecycle(ProcessLifecycle::SESSION_ACTIVE)) {
        char message[160];
        std::snprintf(
            message,
            sizeof(message),
            "benchmark session is prohibited in process lifecycle state %s",
            lifecycle_name(process_lifecycle.load(std::memory_order_acquire)));
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_LIFECYCLE_ERROR,
            error_message,
            error_message_capacity,
            message);
    }

    SYCLBenchmarkSession* handle = nullptr;
    BenchmarkSessionRuntime* runtime = nullptr;
    bool persistent_submission_started = false;
    try {
        handle = new SYCLBenchmarkSession();
        SYCLBenchmarkConfig stored_config = *config;
        stored_config.abi_version = SYCL_BENCHMARK_ABI_VERSION;
        stored_config.struct_size = sizeof(SYCLBenchmarkConfig);
        runtime = new BenchmarkSessionRuntime(stored_config, mod_params);
        handle->runtime = runtime;
        persistent_submission_started = true;
        submit_persistent_services_once(*runtime);
        *session = handle;
        return SYCL_BENCHMARK_STATUS_SUCCESS;
    } catch (const std::exception& exception) {
        process_lifecycle.store(ProcessLifecycle::FAILED, std::memory_order_release);
        if (!persistent_submission_started) {
            delete runtime;
            delete handle;
        }
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_RUNTIME_ERROR,
            error_message,
            error_message_capacity,
            exception.what());
    } catch (...) {
        process_lifecycle.store(ProcessLifecycle::FAILED, std::memory_order_release);
        if (!persistent_submission_started) {
            delete runtime;
            delete handle;
        }
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_RUNTIME_ERROR,
            error_message,
            error_message_capacity,
            "unknown exception while creating benchmark session");
    }
}

extern "C" int SYCL_benchmark_session_create(
    const SYCLBenchmarkConfig* config,
    const double* scales,
    const uint32_t* mod_values,
    const uint32_t* const_ratios,
    SYCLBenchmarkSession** session,
    size_t* required_event_record_capacity,
    char* error_message,
    size_t error_message_capacity)
{
    try {
        return benchmark_session_create_impl(
            config,
            scales,
            mod_values,
            const_ratios,
            session,
            required_event_record_capacity,
            error_message,
            error_message_capacity);
    } catch (const std::exception& exception) {
        ProcessLifecycle expected = ProcessLifecycle::SESSION_ACTIVE;
        process_lifecycle.compare_exchange_strong(expected, ProcessLifecycle::FAILED);
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_RUNTIME_ERROR,
            error_message,
            error_message_capacity,
            exception.what());
    } catch (...) {
        ProcessLifecycle expected = ProcessLifecycle::SESSION_ACTIVE;
        process_lifecycle.compare_exchange_strong(expected, ProcessLifecycle::FAILED);
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_RUNTIME_ERROR,
            error_message,
            error_message_capacity,
            "unknown exception at benchmark create C ABI boundary");
    }
}

static int benchmark_encrypt_batch_impl(
    SYCLBenchmarkSession* session,
    size_t frame_count,
    const complex_double* encoding_buffers,
    const int8_t* error_samples,
    const uint32_t* const* secret_keys,
    const uint32_t* const* uniform_polys,
    uint32_t** c0_outputs,
    uint32_t** c1_outputs,
    uint32_t** s_save,
    uint32_t** ntt_pte_outputs,
    SYCLBenchmarkTiming* timing,
    SYCLBenchmarkEventRecord* event_records,
    size_t event_record_capacity,
    size_t* event_records_written,
    char* error_message,
    size_t error_message_capacity)
{
    const BenchmarkClock::time_point api_start = BenchmarkClock::now();
    set_error_text(error_message, error_message_capacity, "");
    if (!session || session->magic != BENCHMARK_SESSION_MAGIC) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "benchmark session is invalid");
    }

    std::lock_guard<std::mutex> lock(session->mutex);
    size_t required_records = 0;
    int status = validate_batch_arguments(
        session,
        frame_count,
        encoding_buffers,
        error_samples,
        secret_keys,
        uniform_polys,
        c0_outputs,
        c1_outputs,
        s_save,
        ntt_pte_outputs,
        timing,
        event_records,
        event_record_capacity,
        event_records_written,
        required_records,
        error_message,
        error_message_capacity);
    if (status != SYCL_BENCHMARK_STATUS_SUCCESS) {
        return status;
    }

    initialize_timing(*timing);
    try {
        BenchmarkSessionRuntime& runtime = *session->runtime;
        std::vector<SubmittedFrameEvents> submitted(frame_count);
        std::vector<NamedBenchmarkEvent> named;
        named.reserve(required_records);

        const BenchmarkClock::time_point pack_start = BenchmarkClock::now();
        for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
            pack_benchmark_frame(
                runtime.config.n,
                frame_index,
                encoding_buffers,
                error_samples,
                secret_keys,
                uniform_polys,
                *runtime.frames[frame_index]);
        }
        const BenchmarkClock::time_point pack_end = BenchmarkClock::now();
        timing->pack_wall_ns = elapsed_ns(pack_start, pack_end);

        for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
            submitted[frame_index].h2d =
                copy_batch_to_device(runtime.q, *runtime.frames[frame_index]);
        }

        if (frame_count == 1) {
            const BenchmarkClock::time_point h2d_wait_start = BenchmarkClock::now();
            submitted[0].h2d.wait_and_throw();
            timing->h2d_wall_ns = elapsed_ns(h2d_wait_start, BenchmarkClock::now());
        }

        const BenchmarkClock::time_point graph_start = BenchmarkClock::now();
        for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
            SubmittedFrameEvents bounded =
                submit_one_bounded_frame(runtime, *runtime.frames[frame_index]);
            bounded.h2d = submitted[frame_index].h2d;
            submitted[frame_index] = std::move(bounded);
        }

        if (frame_count == 1) {
            for (size_t p = 0; p < NUM_MODULI; ++p) {
                submitted[0].exit_c0[p].wait_and_throw();
                if (runtime.config.save_ntt_s) {
                    submitted[0].exit_ntt_s[p].wait_and_throw();
                }
                if (runtime.config.save_ntt_pte) {
                    submitted[0].exit_ntt_pte[p].wait_and_throw();
                }
            }
            timing->graph_submit_wait_wall_ns =
                elapsed_ns(graph_start, BenchmarkClock::now());
        }

        for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
            copy_batch_to_host(
                runtime,
                *runtime.frames[frame_index],
                submitted[frame_index]);
        }
        const BenchmarkClock::time_point d2h_wait_start = BenchmarkClock::now();

        // For B>1 the complete batch is submitted before any wait. B=1 uses
        // disjoint synchronous waits above for the additive E2 wall breakdown.
        for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
            for (size_t p = 0; p < NUM_MODULI; ++p) {
                submitted[frame_index].d2h_c0[p].wait_and_throw();
                if (runtime.config.save_ntt_s) {
                    submitted[frame_index].d2h_ntt_s[p].wait_and_throw();
                }
                if (runtime.config.save_ntt_pte) {
                    submitted[frame_index].d2h_ntt_pte[p].wait_and_throw();
                }
            }
        }
        if (frame_count == 1) {
            timing->d2h_wall_ns = elapsed_ns(d2h_wait_start, BenchmarkClock::now());
            timing->additive_wall_breakdown_available = 1;
        }

        const BenchmarkClock::time_point unpack_start = BenchmarkClock::now();
        for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
            const size_t frame_offset = frame_index * runtime.config.n;
            BenchmarkFrameBuffers& frame = *runtime.frames[frame_index];
            for (size_t p = 0; p < NUM_MODULI; ++p) {
                unpack_u32_blocks(
                    runtime.config.n,
                    frame.c0_host[p],
                    c0_outputs[p] + frame_offset);
                std::memcpy(
                    c1_outputs[p] + frame_offset,
                    uniform_polys[p] + frame_offset,
                    runtime.config.n * sizeof(uint32_t));
                if (runtime.config.save_ntt_s) {
                    unpack_u32_blocks(
                        runtime.config.n,
                        frame.ntt_s_host[p],
                        s_save[p] + frame_offset);
                }
                if (runtime.config.save_ntt_pte) {
                    unpack_u32_blocks(
                        runtime.config.n,
                        frame.ntt_pte_host[p],
                        ntt_pte_outputs[p] + frame_offset);
                }
            }
        }
        const BenchmarkClock::time_point unpack_end = BenchmarkClock::now();
        timing->unpack_wall_ns = elapsed_ns(unpack_start, unpack_end);
        timing->accelerator_api_wall_ns = elapsed_ns(api_start, unpack_end);

        for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
            append_named_frame_events(
                runtime.config,
                frame_index,
                submitted[frame_index],
                named);
        }
        if (named.size() != required_records) {
            throw std::logic_error("internal benchmark event count mismatch");
        }
        collect_named_profiling(
            named,
            runtime.config.enable_profiling != 0,
            event_records);
        aggregate_timing_from_records(event_records, required_records, *timing);
        *event_records_written = required_records;
        return SYCL_BENCHMARK_STATUS_SUCCESS;
    } catch (const std::exception& exception) {
        mark_session_failed(*session);
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_RUNTIME_ERROR,
            error_message,
            error_message_capacity,
            exception.what());
    } catch (...) {
        mark_session_failed(*session);
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_RUNTIME_ERROR,
            error_message,
            error_message_capacity,
            "unknown exception while submitting benchmark batch");
    }
}

extern "C" int SYCL_benchmark_encrypt_batch(
    SYCLBenchmarkSession* session,
    size_t frame_count,
    const complex_double* encoding_buffers,
    const int8_t* error_samples,
    const uint32_t* const* secret_keys,
    const uint32_t* const* uniform_polys,
    uint32_t** c0_outputs,
    uint32_t** c1_outputs,
    uint32_t** s_save,
    uint32_t** ntt_pte_outputs,
    SYCLBenchmarkTiming* timing,
    SYCLBenchmarkEventRecord* event_records,
    size_t event_record_capacity,
    size_t* event_records_written,
    char* error_message,
    size_t error_message_capacity)
{
    try {
        return benchmark_encrypt_batch_impl(
            session,
            frame_count,
            encoding_buffers,
            error_samples,
            secret_keys,
            uniform_polys,
            c0_outputs,
            c1_outputs,
            s_save,
            ntt_pte_outputs,
            timing,
            event_records,
            event_record_capacity,
            event_records_written,
            error_message,
            error_message_capacity);
    } catch (const std::exception& exception) {
        if (session && session->magic == BENCHMARK_SESSION_MAGIC) {
            mark_session_failed(*session);
        }
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_RUNTIME_ERROR,
            error_message,
            error_message_capacity,
            exception.what());
    } catch (...) {
        if (session && session->magic == BENCHMARK_SESSION_MAGIC) {
            mark_session_failed(*session);
        }
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_RUNTIME_ERROR,
            error_message,
            error_message_capacity,
            "unknown exception at benchmark batch C ABI boundary");
    }
}

static int benchmark_session_close_impl(
    SYCLBenchmarkSession* session,
    char* error_message,
    size_t error_message_capacity)
{
    set_error_text(error_message, error_message_capacity, "");
    if (!session || session->magic != BENCHMARK_SESSION_MAGIC) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT, error_message, error_message_capacity, "benchmark session is invalid");
    }

    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->closed || session->failed) {
        return benchmark_error(SYCL_BENCHMARK_STATUS_LIFECYCLE_ERROR, error_message, error_message_capacity, "benchmark session is already terminal");
    }
    ProcessLifecycle expected = ProcessLifecycle::SESSION_ACTIVE;
    if (!process_lifecycle.compare_exchange_strong(
            expected,
            ProcessLifecycle::CLOSED,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        session->failed = true;
        return benchmark_error(SYCL_BENCHMARK_STATUS_LIFECYCLE_ERROR, error_message, error_message_capacity, "benchmark process lifecycle is not session-active");
    }
    session->closed = true;

    // Deliberate process-lifetime ownership: do not delete the handle/runtime.
    // Their queue owns nonterminating IFFT/NTT commands, and destroying it here
    // could wait on those services. The caller must exit the process after close.
    return SYCL_BENCHMARK_STATUS_SUCCESS;
}

extern "C" int SYCL_benchmark_session_close(
    SYCLBenchmarkSession* session,
    char* error_message,
    size_t error_message_capacity)
{
    try {
        return benchmark_session_close_impl(
            session,
            error_message,
            error_message_capacity);
    } catch (const std::exception& exception) {
        if (session && session->magic == BENCHMARK_SESSION_MAGIC) {
            mark_session_failed(*session);
        }
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_RUNTIME_ERROR,
            error_message,
            error_message_capacity,
            exception.what());
    } catch (...) {
        if (session && session->magic == BENCHMARK_SESSION_MAGIC) {
            mark_session_failed(*session);
        }
        return benchmark_error(
            SYCL_BENCHMARK_STATUS_RUNTIME_ERROR,
            error_message,
            error_message_capacity,
            "unknown exception at benchmark close C ABI boundary");
    }
}

#ifdef SYCL_CKKS_BENCHMARK_PRIVATE_TEST
extern "C" void SYCL_benchmark_private_reset_lifecycle(void)
{
    process_lifecycle.store(ProcessLifecycle::UNUSED, std::memory_order_release);
}

extern "C" uint32_t SYCL_benchmark_private_lifecycle_state(void)
{
    return static_cast<uint32_t>(process_lifecycle.load(std::memory_order_acquire));
}

extern "C" void SYCL_benchmark_private_set_post_guard_hook(
    SYCLBenchmarkPrivatePostGuardHook hook,
    void* context)
{
    private_post_guard_context.store(context, std::memory_order_release);
    private_post_guard_hook.store(hook, std::memory_order_release);
}

extern "C" int SYCL_benchmark_private_try_claim_legacy(void)
{
    return try_claim_process_lifecycle(ProcessLifecycle::LEGACY_ACTIVE) ? 1 : 0;
}

extern "C" int SYCL_benchmark_private_try_claim_session(void)
{
    return try_claim_process_lifecycle(ProcessLifecycle::SESSION_ACTIVE) ? 1 : 0;
}

extern "C" void SYCL_benchmark_private_mark_legacy_used(void)
{
    ProcessLifecycle expected = ProcessLifecycle::LEGACY_ACTIVE;
    process_lifecycle.compare_exchange_strong(expected, ProcessLifecycle::LEGACY_USED);
}

extern "C" void SYCL_benchmark_private_mark_session_closed(void)
{
    ProcessLifecycle expected = ProcessLifecycle::SESSION_ACTIVE;
    process_lifecycle.compare_exchange_strong(expected, ProcessLifecycle::CLOSED);
}

extern "C" void SYCL_benchmark_private_mark_session_failed(void)
{
    ProcessLifecycle expected = ProcessLifecycle::SESSION_ACTIVE;
    process_lifecycle.compare_exchange_strong(expected, ProcessLifecycle::FAILED);
}
#endif

extern "C" void SYCL_encrypt(
    size_t n,
    const double* scales,
    const uint32_t* mod_values,
    const uint32_t* const_ratios,
    const complex_double* encoding_buffer,
    const int8_t* error_samples,
    const uint32_t* const* secret_keys,
    const uint32_t* const* uniform_polys,
    uint32_t** c0_outputs,
    uint32_t** c1_outputs,
    uint32_t** s_save,
    uint32_t** c1_save,
    uint32_t** ntt_pte_outputs)
{
    if (n != POLY_N || n != SYCL_POLY_N) {
        std::cerr << "[SYCL_encrypt] this accelerator image requires polynomial degree "
                  << SYCL_POLY_N << "; received " << n << "\n";
        std::exit(1);
    }
    if (n % LANES != 0) {
        std::cerr << "[SYCL_encrypt] polynomial degree must be divisible by " << LANES << "\n";
        std::exit(1);
    }
    if (!scales || !mod_values || !const_ratios || !encoding_buffer || !error_samples ||
        !secret_keys || !uniform_polys || !c0_outputs) {
        std::cerr << "[SYCL_encrypt] required 8K accelerator argument is null\n";
        std::exit(1);
    }

    std::array<ModulusParams, NUM_MODULI> mod_params;
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        if (!secret_keys[p] || !uniform_polys[p] || !c0_outputs[p]) {
            std::cerr << "[SYCL_encrypt] required modulus buffer is null at index " << p << "\n";
            std::exit(1);
        }

        uint8_t selector = get_modulus_selector(mod_values[p]);
        if (selector == 0xff) {
            std::cerr << "[SYCL_encrypt] modulus " << mod_values[p]
                      << " is not supported by the six-channel 8K NTT RTL\n";
            std::exit(1);
        }

        uint32_t expected_cr0 = 0;
        uint32_t expected_cr1 = 0;
        if (!get_barrett_constants(mod_values[p], expected_cr0, expected_cr1) ||
            expected_cr0 != const_ratios[p * 2] || expected_cr1 != const_ratios[p * 2 + 1]) {
            std::cerr << "[SYCL_encrypt] Barrett constants do not match modulus index " << p << "\n";
            std::exit(1);
        }

        mod_params[p].scale = scales[p];
        mod_params[p].mod_value = mod_values[p];
        mod_params[p].const_ratio[0] = const_ratios[p * 2];
        mod_params[p].const_ratio[1] = const_ratios[p * 2 + 1];
        mod_params[p].modulus_selector = selector;
        mod_params[p].save_ntt_s = (s_save && s_save[p]);
        mod_params[p].save_ntt_pte = (ntt_pte_outputs && ntt_pte_outputs[p]);

        if (c1_save && c1_save[p]) {
            std::memcpy(c1_save[p], uniform_polys[p], n * sizeof(uint32_t));
        }
    }

    size_t num_blocks = n / LANES;
    std::vector<PipelineInputBlock> input_blocks;
    pack_input(n, encoding_buffer, error_samples, secret_keys, uniform_polys, input_blocks);

    std::array<std::vector<u32x4>, NUM_MODULI> c0_blocks;
    std::array<std::vector<u32x4>, NUM_MODULI> ntt_s_blocks;
    std::array<std::vector<u32x4>, NUM_MODULI> ntt_pte_blocks;
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        c0_blocks[p].resize(num_blocks);
        ntt_s_blocks[p].resize(num_blocks);
        ntt_pte_blocks[p].resize(num_blocks);
    }

    if (!try_claim_process_lifecycle(ProcessLifecycle::LEGACY_ACTIVE)) {
        ProcessLifecycle state = process_lifecycle.load(std::memory_order_acquire);
        std::cerr << "[SYCL_encrypt] legacy call is prohibited in process lifecycle state "
                  << lifecycle_name(state) << "\n";
        std::exit(1);
    }

    {
        buffer<PipelineInputBlock, 1> input_buf(input_blocks.data(), range(num_blocks));
        std::array<buffer<u32x4, 1>, NUM_MODULI> c0_bufs = {
            buffer<u32x4, 1>(c0_blocks[0].data(), range(num_blocks)),
            buffer<u32x4, 1>(c0_blocks[1].data(), range(num_blocks)),
            buffer<u32x4, 1>(c0_blocks[2].data(), range(num_blocks)),
            buffer<u32x4, 1>(c0_blocks[3].data(), range(num_blocks)),
            buffer<u32x4, 1>(c0_blocks[4].data(), range(num_blocks)),
            buffer<u32x4, 1>(c0_blocks[5].data(), range(num_blocks))
        };
        std::array<buffer<u32x4, 1>, NUM_MODULI> ntt_s_bufs = {
            buffer<u32x4, 1>(ntt_s_blocks[0].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_s_blocks[1].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_s_blocks[2].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_s_blocks[3].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_s_blocks[4].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_s_blocks[5].data(), range(num_blocks))
        };
        std::array<buffer<u32x4, 1>, NUM_MODULI> ntt_pte_bufs = {
            buffer<u32x4, 1>(ntt_pte_blocks[0].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_pte_blocks[1].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_pte_blocks[2].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_pte_blocks[3].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_pte_blocks[4].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_pte_blocks[5].data(), range(num_blocks))
        };

#if FPGA_SIMULATOR
        auto selector = ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
        auto selector = ext::intel::fpga_selector_v;
#else
        auto selector = ext::intel::fpga_emulator_selector_v;
#endif
        queue q{selector, property::queue::enable_profiling()};
        auto events = run_pipeline(q, input_buf, c0_bufs, ntt_s_bufs, ntt_pte_bufs, mod_params);
        for (auto& ev : events) ev.wait_and_throw();
    }

    for (size_t p = 0; p < NUM_MODULI; ++p) {
        unpack_u32_blocks(n, c0_blocks[p], c0_outputs[p]);
        if (mod_params[p].save_ntt_s) {
            unpack_u32_blocks(n, ntt_s_blocks[p], s_save[p]);
        }
        if (mod_params[p].save_ntt_pte) {
            unpack_u32_blocks(n, ntt_pte_blocks[p], ntt_pte_outputs[p]);
        }
        if (c1_outputs && c1_outputs[p]) {
            std::memcpy(c1_outputs[p], uniform_polys[p], n * sizeof(uint32_t));
        }
    }

    process_lifecycle.store(ProcessLifecycle::LEGACY_USED, std::memory_order_release);
}
