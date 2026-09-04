#include "fpt2026_benchmark/accelerator_runner.hpp"

#include "fpt2026_benchmark/seal_oracle.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <string>

namespace fpt2026 {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end)
{
    const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    if (value < 0) {
        throw std::runtime_error("steady_clock produced a negative interval");
    }
    return static_cast<std::uint64_t>(value);
}

void check_accelerator_status(int status, const char* message, const char* operation)
{
    if (status != static_cast<int>(SYCL_BENCHMARK_STATUS_SUCCESS)) {
        throw std::runtime_error(
            std::string(operation) + " failed: " +
            (message && message[0] ? message : "unknown accelerator error"));
    }
}

std::size_t extract_pipeline_input_block_size(
    const std::vector<SYCLBenchmarkEventRecord>& events)
{
    const std::size_t blocks_per_frame = kPolyModulusDegree / 4;
    std::size_t block_size = 0;
    std::size_t h2d_records = 0;
    for (const auto& event : events) {
        if (event.stage != SYCL_BENCHMARK_STAGE_H2D ||
            event.transfer_kind != SYCL_BENCHMARK_TRANSFER_PACKED_INPUT) {
            continue;
        }
        ++h2d_records;
        if (event.byte_count == 0 || event.byte_count % blocks_per_frame != 0) {
            throw std::runtime_error("H2D event byte count cannot identify PipelineInputBlock size");
        }
        const std::size_t observed = static_cast<std::size_t>(event.byte_count / blocks_per_frame);
        if (block_size != 0 && observed != block_size) {
            throw std::runtime_error("inconsistent PipelineInputBlock size across H2D records");
        }
        block_size = observed;
    }
    if (h2d_records == 0 || block_size == 0) {
        throw std::runtime_error("accelerator returned no packed-input H2D event");
    }
    return block_size;
}

} // namespace

AcceleratorRunner::AcceleratorRunner(
    SealEmbeddedRunner& preparation,
    StockSealOracle& oracle,
    std::size_t max_frames,
    bool save_ntt_s,
    bool save_ntt_pte)
    : preparation_(&preparation),
      oracle_(&oracle),
      max_frames_(max_frames),
      save_ntt_s_(save_ntt_s),
      save_ntt_pte_(save_ntt_pte)
{
    if (max_frames == 0) {
        throw std::invalid_argument("accelerator max_frames must be nonzero");
    }
    SYCLBenchmarkConfig config{};
    config.abi_version = SYCL_BENCHMARK_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.n = kPolyModulusDegree;
    config.num_moduli = kDataModulusCount;
    config.max_frames = max_frames;
    config.save_ntt_s = save_ntt_s ? 1u : 0u;
    config.save_ntt_pte = save_ntt_pte ? 1u : 0u;
    config.enable_profiling = 1u;
    std::array<double, kDataModulusCount> scales{};
    scales.fill(kScale);
    char error[512]{};
    check_accelerator_status(
        SYCL_benchmark_session_create(
            &config, scales.data(), preparation.moduli().data(),
            preparation.const_ratios().data(), &session_, &event_capacity_,
            error, sizeof(error)),
        error, "SYCL_benchmark_session_create");
    const std::size_t expected = max_frames *
        accelerator_events_per_frame(save_ntt_s, save_ntt_pte);
    if (event_capacity_ != expected) {
        throw std::runtime_error("accelerator returned an incorrect exact event capacity");
    }
}

AcceleratorRunner::~AcceleratorRunner()
{
    if (session_ != nullptr && !closed_) {
        char error[512]{};
        (void)SYCL_benchmark_session_close(session_, error, sizeof(error));
        session_ = nullptr;
        closed_ = true;
    }
}

AcceleratorBatchResult AcceleratorRunner::encrypt(
    const std::vector<BenchmarkVector>& frames,
    const std::vector<TrialSeeds>& seeds)
{
    if (closed_ || session_ == nullptr) {
        throw std::logic_error("accelerator session is closed");
    }
    if (frames.empty() || frames.size() != seeds.size() ||
        frames.size() > max_frames_) {
        throw std::invalid_argument("accelerator batch has an invalid frame or seed count");
    }

    AcceleratorBatchResult result;
    result.residues.frame_count = frames.size();
    for (const auto& seed : seeds) {
        result.trial_seed_digests.push_back(seed.digest);
    }

    const auto application_begin = Clock::now();
    const auto preparation_begin = application_begin;
    auto inputs = preparation_->prepare_accelerator_inputs(frames, seeds);
    const auto preparation_end = Clock::now();
    result.timing.preparation_wall_ns = elapsed_ns(preparation_begin, preparation_end);

    const std::size_t frame_coefficients = frames.size() * kPolyModulusDegree;
    for (std::size_t p = 0; p < kDataModulusCount; ++p) {
        result.residues.c0[p].resize(frame_coefficients);
        result.residues.c1[p].resize(frame_coefficients);
    }
    std::array<const std::uint32_t*, kDataModulusCount> secret_ptrs{};
    std::array<const std::uint32_t*, kDataModulusCount> uniform_ptrs{};
    std::array<std::uint32_t*, kDataModulusCount> c0_ptrs{};
    std::array<std::uint32_t*, kDataModulusCount> c1_ptrs{};
    std::array<std::vector<std::uint32_t>, kDataModulusCount> ntt_s;
    std::array<std::vector<std::uint32_t>, kDataModulusCount> ntt_pte;
    std::array<std::uint32_t*, kDataModulusCount> ntt_s_ptrs{};
    std::array<std::uint32_t*, kDataModulusCount> ntt_pte_ptrs{};
    for (std::size_t p = 0; p < kDataModulusCount; ++p) {
        secret_ptrs[p] = inputs.secret_keys[p].data();
        uniform_ptrs[p] = inputs.uniform_polys[p].data();
        c0_ptrs[p] = result.residues.c0[p].data();
        c1_ptrs[p] = result.residues.c1[p].data();
        if (save_ntt_s_) {
            ntt_s[p].resize(frame_coefficients);
            ntt_s_ptrs[p] = ntt_s[p].data();
        }
        if (save_ntt_pte_) {
            ntt_pte[p].resize(frame_coefficients);
            ntt_pte_ptrs[p] = ntt_pte[p].data();
        }
    }

    const std::size_t exact_event_count = frames.size() *
        accelerator_events_per_frame(save_ntt_s_, save_ntt_pte_);
    result.events.resize(exact_event_count);
    for (auto& event : result.events) {
        event.abi_version = SYCL_BENCHMARK_ABI_VERSION;
        event.struct_size = sizeof(event);
    }
    SYCLBenchmarkTiming api_timing{};
    api_timing.abi_version = SYCL_BENCHMARK_ABI_VERSION;
    api_timing.struct_size = sizeof(api_timing);
    std::size_t records_written = 0;
    char error[512]{};
    check_accelerator_status(
        SYCL_benchmark_encrypt_batch(
            session_, frames.size(), inputs.encoding_buffers.data(),
            inputs.error_samples.data(), secret_ptrs.data(), uniform_ptrs.data(),
            c0_ptrs.data(), c1_ptrs.data(),
            save_ntt_s_ ? ntt_s_ptrs.data() : nullptr,
            save_ntt_pte_ ? ntt_pte_ptrs.data() : nullptr,
            &api_timing, result.events.data(), result.events.size(),
            &records_written, error, sizeof(error)),
        error, "SYCL_benchmark_encrypt_batch");
    if (records_written != exact_event_count) {
        throw std::runtime_error("accelerator returned the wrong event-record count");
    }
    if (api_timing.accelerator_api_wall_ns == 0) {
        throw std::runtime_error("accelerator returned an impossible zero API wall interval");
    }

    result.timing.pack_wall_ns = api_timing.pack_wall_ns;
    result.timing.h2d_wall_ns = api_timing.h2d_wall_ns;
    result.timing.h2d_device_ns = api_timing.h2d_device_ns;
    result.timing.graph_device_ns = api_timing.graph_device_ns;
    result.timing.d2h_device_ns = api_timing.d2h_device_ns;
    result.timing.d2h_wall_ns = api_timing.d2h_wall_ns;
    result.timing.graph_submit_wait_wall_ns = api_timing.graph_submit_wait_wall_ns;
    result.timing.h2d_device_available = api_timing.h2d_profiling_available != 0;
    result.timing.graph_device_available = api_timing.graph_profiling_available != 0;
    result.timing.d2h_device_available = api_timing.d2h_profiling_available != 0;
    result.timing.additive_wall_breakdown_available =
        api_timing.additive_wall_breakdown_available != 0;
    if (result.timing.additive_wall_breakdown_available != (frames.size() == 1)) {
        throw std::runtime_error("accelerator additive wall-breakdown availability is inconsistent");
    }
    result.pipeline_input_block_size = extract_pipeline_input_block_size(result.events);

    const auto assembly_begin = Clock::now();
    result.ciphertexts.reserve(frames.size());
    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        std::array<std::vector<std::uint32_t>, kDataModulusCount> c0;
        std::array<std::vector<std::uint32_t>, kDataModulusCount> c1;
        for (std::size_t p = 0; p < kDataModulusCount; ++p) {
            const auto first0 = result.residues.c0[p].begin() +
                static_cast<std::ptrdiff_t>(frame * kPolyModulusDegree);
            const auto first1 = result.residues.c1[p].begin() +
                static_cast<std::ptrdiff_t>(frame * kPolyModulusDegree);
            c0[p].assign(first0, first0 + static_cast<std::ptrdiff_t>(kPolyModulusDegree));
            c1[p].assign(first1, first1 + static_cast<std::ptrdiff_t>(kPolyModulusDegree));
        }
        result.ciphertexts.push_back(oracle_->assemble_ciphertext(c0, c1));
    }
    const auto application_end = Clock::now();
    result.timing.unpack_assembly_wall_ns =
        api_timing.unpack_wall_ns + elapsed_ns(assembly_begin, application_end);
    result.timing.application_e2e_ns = elapsed_ns(application_begin, application_end);
    std::uint64_t attributed = result.timing.preparation_wall_ns +
        result.timing.pack_wall_ns + result.timing.unpack_assembly_wall_ns;
    if (result.timing.additive_wall_breakdown_available) {
        attributed += result.timing.h2d_wall_ns +
            result.timing.graph_submit_wait_wall_ns + result.timing.d2h_wall_ns;
    }
    if (attributed > result.timing.application_e2e_ns) {
        throw std::runtime_error("host wall regions exceed application_e2e_ns");
    }
    result.timing.unattributed_wall_ns = result.timing.application_e2e_ns - attributed;

    // All transport, retained-c1, and semantic checks happen after copyback and
    // after the measured application boundary.
    result.passed = true;
    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        for (std::size_t p = 0; p < kDataModulusCount; ++p) {
            const std::size_t base = frame * kPolyModulusDegree;
            for (std::size_t i = 0; i < kPolyModulusDegree; ++i) {
                const std::uint32_t c0 = result.residues.c0[p][base + i];
                const std::uint32_t c1 = result.residues.c1[p][base + i];
                if (c0 >= kDataModuli[p] || c1 >= kDataModuli[p] ||
                    c1 != inputs.uniform_polys[p][base + i]) {
                    result.passed = false;
                }
            }
        }
        auto decoded = oracle_->decrypt_decode(result.ciphertexts[frame]);
        auto metrics = evaluate_slots(
            frames[frame].slots, decoded, frames[frame].active_slots,
            kCorrectnessThreshold);
        result.passed = result.passed && metrics.passed;
        result.decoded_slots.push_back(std::move(decoded));
        result.correctness.push_back(std::move(metrics));
    }
    return result;
}

void AcceleratorRunner::close()
{
    if (closed_) {
        return;
    }
    if (session_ == nullptr) {
        throw std::logic_error("accelerator session handle is null");
    }
    char error[512]{};
    check_accelerator_status(
        SYCL_benchmark_session_close(session_, error, sizeof(error)),
        error, "SYCL_benchmark_session_close");
    session_ = nullptr;
    closed_ = true;
}

std::size_t AcceleratorRunner::required_event_capacity() const noexcept
{
    return event_capacity_;
}

} // namespace fpt2026
