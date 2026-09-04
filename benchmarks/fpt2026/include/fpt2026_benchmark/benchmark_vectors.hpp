#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "fpt2026_benchmark/benchmark_config.hpp"

namespace fpt2026 {

enum class VectorCaseId {
    real_impulse,
    real_short_mixed,
    real_partial_64,
    real_partial_1024,
    real_full_4096,
    complex_short_mixed,
    complex_partial_64,
    complex_full_4096,
};

struct BenchmarkVector {
    VectorCaseId case_id{};
    std::string id;
    std::size_t active_slots{};
    std::vector<std::complex<double>> slots;
};

VectorCaseId parse_vector_case(std::string_view id);
std::string_view vector_case_name(VectorCaseId id) noexcept;
bool is_complex_case(VectorCaseId id) noexcept;
std::size_t active_slot_count(VectorCaseId id) noexcept;
std::vector<VectorCaseId> all_vector_cases();
BenchmarkVector generate_benchmark_vector(VectorCaseId id);

// Creates the exact N-element conjugate embedding expected by the accelerator.
// index_map must contain N entries produced by SEAL-Embedded ckks_setup().
std::vector<std::complex<double>> make_conjugate_embedding(
    const BenchmarkVector& input,
    const std::vector<std::uint16_t>& index_map);

// Canonical little-endian artifact: magic/version, case name, active count, then
// 4096 pairs of IEEE-754 binary64 real/imaginary values.
std::vector<std::uint8_t> serialize_vector_artifact(const BenchmarkVector& input);

// SHAKE256 with the FIPS 202 domain suffix. Exposed so trial seed derivation and
// vector generation share one audited implementation.
std::vector<std::uint8_t> shake256(
    std::string_view domain,
    std::size_t output_size);

} // namespace fpt2026
