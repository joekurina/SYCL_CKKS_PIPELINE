#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fpt2026 {

struct NumericalMetrics {
    std::size_t requested_slot_count{};
    std::size_t inactive_slot_count{};
    std::size_t compared_value_count{};
    std::size_t finite_value_count{};
    std::size_t nonfinite_value_count{};
    std::size_t mismatch_count{};
    std::size_t worst_index{};
    double max_abs_error{};
    double rms_error{};
    double max_real_error{};
    double max_imag_error{};
    double component_max_abs_error{};
    double threshold{};
    std::complex<double> worst_expected{};
    std::complex<double> worst_actual{};
    bool passed{};
    std::string failure_reason;
};

NumericalMetrics evaluate_slots(
    const std::vector<std::complex<double>>& expected,
    const std::vector<std::complex<double>>& actual,
    std::size_t requested_slot_count,
    double threshold);

struct ExactMetrics {
    std::size_t expected_count{};
    std::size_t actual_count{};
    std::size_t compared_count{};
    std::size_t mismatch_count{};
    std::size_t noncanonical_count{};
    std::size_t first_mismatch_index{};
    std::uint64_t first_expected{};
    std::uint64_t first_actual{};
    bool passed{};
};

ExactMetrics compare_residues(
    const std::vector<std::uint32_t>& expected,
    const std::vector<std::uint32_t>& actual,
    std::uint32_t modulus);

} // namespace fpt2026
