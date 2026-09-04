#pragma once

#include <array>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "fpt2026_benchmark/benchmark_config.hpp"
#include "fpt2026_benchmark/benchmark_vectors.hpp"
#include "fpt2026_benchmark/correctness_metrics.hpp"
#include "fpt2026_benchmark/provenance.hpp"
#include "fpt2026_benchmark/seal_embedded_c_runner.h"
#include "seal/seal.h"

namespace fpt2026 {

class StockSealOracle;

struct TrialSeeds {
    std::array<std::uint8_t, 64> shareable{};
    std::array<std::uint8_t, 64> error{};
    std::string digest;
};

TrialSeeds make_trial_seeds(
    const std::string& run_id,
    const std::string& case_id,
    std::size_t repetition,
    std::size_t frame_index);

struct TimingBreakdown {
    std::uint64_t application_e2e_ns{};
    std::uint64_t preparation_wall_ns{};
    std::uint64_t pack_wall_ns{};
    std::uint64_t h2d_wall_ns{};
    std::uint64_t h2d_device_ns{};
    std::uint64_t graph_device_ns{};
    std::uint64_t d2h_device_ns{};
    std::uint64_t d2h_wall_ns{};
    std::uint64_t graph_submit_wait_wall_ns{};
    std::uint64_t unpack_assembly_wall_ns{};
    std::uint64_t unattributed_wall_ns{};
    bool h2d_device_available{};
    bool graph_device_available{};
    bool d2h_device_available{};
    bool additive_wall_breakdown_available{};
};

struct ResidueBatch {
    std::size_t frame_count{};
    std::array<std::vector<std::uint32_t>, kDataModulusCount> c0;
    std::array<std::vector<std::uint32_t>, kDataModulusCount> c1;
};

struct TransportMetrics {
    std::size_t noncanonical_count{};
    std::size_t retained_c1_mismatch_count{};
    std::size_t mismatch_count{};
    bool passed{true};
};

struct BackendBatchResult {
    TimingBreakdown timing;
    ResidueBatch residues;
    std::vector<seal::Ciphertext> ciphertexts;
    std::vector<std::vector<std::complex<double>>> decoded_slots;
    std::vector<NumericalMetrics> correctness;
    std::vector<TransportMetrics> transport_correctness;
    std::vector<std::string> trial_seed_digests;
    bool passed{};
};

struct PreparedAcceleratorInputs {
    std::size_t frame_count{};
    std::vector<std::complex<double>> encoding_buffers;
    std::vector<std::int8_t> error_samples;
    std::array<std::vector<std::uint32_t>, kDataModulusCount> secret_keys;
    std::array<std::vector<std::uint32_t>, kDataModulusCount> uniform_polys;
};

class SealEmbeddedRunner {
public:
    SealEmbeddedRunner(
        const std::filesystem::path& compact_secret_key_path,
        StockSealOracle& oracle);
    ~SealEmbeddedRunner();

    SealEmbeddedRunner(const SealEmbeddedRunner&) = delete;
    SealEmbeddedRunner& operator=(const SealEmbeddedRunner&) = delete;

    BackendBatchResult encrypt(
        const std::vector<BenchmarkVector>& frames,
        const std::vector<TrialSeeds>& seeds);

    PreparedAcceleratorInputs prepare_accelerator_inputs(
        const std::vector<BenchmarkVector>& frames,
        const std::vector<TrialSeeds>& seeds);

    const std::vector<std::uint16_t>& index_map() const noexcept;
    const std::array<std::uint32_t, kDataModulusCount>& moduli() const noexcept;
    const std::array<std::uint32_t, 2 * kDataModulusCount>& const_ratios() const noexcept;

private:
    FPT2026SealEmbeddedState* state_{};
    StockSealOracle* oracle_{};
    std::array<std::uint32_t, kDataModulusCount> moduli_{};
    std::array<std::uint32_t, 2 * kDataModulusCount> const_ratios_{};
    std::vector<std::uint16_t> index_map_;
};

} // namespace fpt2026
