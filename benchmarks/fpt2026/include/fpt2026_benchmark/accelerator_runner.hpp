#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "fpt2026_benchmark/seal_embedded_runner.hpp"
#include "sycl_ckks_accelerator/SYCL_ckks_benchmark.h"

namespace fpt2026 {

class StockSealOracle;

struct AcceleratorBatchResult : BackendBatchResult {
    std::vector<SYCLBenchmarkEventRecord> events;
    std::size_t pipeline_input_block_size{};
};

class AcceleratorRunner {
public:
    AcceleratorRunner(
        SealEmbeddedRunner& preparation,
        StockSealOracle& oracle,
        std::size_t max_frames,
        bool save_ntt_s,
        bool save_ntt_pte);
    ~AcceleratorRunner();

    AcceleratorRunner(const AcceleratorRunner&) = delete;
    AcceleratorRunner& operator=(const AcceleratorRunner&) = delete;

    AcceleratorBatchResult encrypt(
        const std::vector<BenchmarkVector>& frames,
        const std::vector<TrialSeeds>& seeds);

    void close();
    std::size_t required_event_capacity() const noexcept;

private:
    SealEmbeddedRunner* preparation_{};
    StockSealOracle* oracle_{};
    SYCLBenchmarkSession* session_{};
    std::size_t max_frames_{};
    std::size_t event_capacity_{};
    bool save_ntt_s_{};
    bool save_ntt_pte_{};
    bool closed_{};
};

} // namespace fpt2026
