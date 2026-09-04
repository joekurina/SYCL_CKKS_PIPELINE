#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "fpt2026_benchmark/seal_embedded_runner.hpp"
#include "fpt2026_benchmark/seal_oracle.hpp"
#include "sycl_ckks_accelerator/SYCL_ckks_benchmark.h"

namespace fpt2026 {

struct RecordIdentity {
    std::string run_id;
    std::string attempt_id;
    std::string planned_unit_id;
    std::string experiment_id;
    std::string backend;
    std::string phase;
    std::string case_id;
    std::size_t repetition{};
    std::size_t trial_seed_index{};
};

struct EventFrontierRecord {
    std::size_t frame_index{};
    bool profiling_available{};
    std::optional<std::uint64_t> entry_end_ns;
    std::optional<std::uint64_t> fanout_end_ns;
    std::optional<std::uint64_t> scale_end_ns;
    std::optional<std::uint64_t> poly_end_ns;
    std::optional<std::uint64_t> exit_end_ns;
    std::optional<std::string> unavailable_reason;
};

struct TimingRecordInput {
    RecordIdentity identity;
    std::string sample_id;
    std::string vector_descriptor_sha256;
    std::string trial_seed_digest;
    std::string benchmark_key_pair_id;
    std::string control_before_id;
    std::string control_after_id;
    std::size_t batch_size{};
    std::size_t frame_count_submitted{};
    std::size_t frame_count_completed{};
    bool save_ntt_s{};
    bool save_ntt_pte{};
    TimingBreakdown timing;
    std::optional<std::uint64_t> cold_first_result_ns;
    std::optional<std::uint64_t> program_time_ns;
    std::uint64_t h2d_bytes{};
    std::uint64_t d2h_bytes{};
    std::vector<std::string> event_record_ids;
    std::vector<EventFrontierRecord> event_frontiers;
    std::vector<std::string> correctness_record_ids;
    double max_error{};
    double rms_error{};
    std::size_t mismatch_count{};
    bool verified_after_timing{};
    std::string status;
    std::optional<std::string> error;
};

struct CorrectnessRecordInput {
    RecordIdentity identity;
    std::string correctness_record_id;
    std::string check_id;
    std::optional<std::string> sample_id;
    std::string vector_descriptor_sha256;
    std::string trial_seed_digest;
    std::string benchmark_key_pair_id;
    std::string oracle_id;
    std::string verified_utc;
    std::size_t frame_index{};
    NumericalMetrics metrics;
    TransportMetrics transport_metrics;
    std::optional<std::string> paired_reference_correctness_record_id;
    std::optional<double> pairwise_max_abs_error;
    std::optional<double> pairwise_rms_error;
};

std::string escape_json(std::string_view value);
std::string timing_record_json(const TimingRecordInput& record);
std::string correctness_record_json(const CorrectnessRecordInput& record);
std::string event_record_json(
    const RecordIdentity& identity,
    const std::string& event_record_id,
    const std::optional<std::string>& sample_id,
    const SYCLBenchmarkEventRecord& event);
std::string context_provenance_json(const ContextProvenance& provenance);

// Appends one already-serialized object plus a newline and fsyncs before return.
void append_jsonl_durable(
    const std::filesystem::path& absolute_path,
    const std::string& json_object);

} // namespace fpt2026
