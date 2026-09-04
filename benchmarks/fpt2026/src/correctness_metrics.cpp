#include "fpt2026_benchmark/correctness_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fpt2026 {

NumericalMetrics evaluate_slots(
    const std::vector<std::complex<double>>& expected,
    const std::vector<std::complex<double>>& actual,
    std::size_t requested_slot_count,
    double threshold)
{
    if (expected.empty() || expected.size() != actual.size()) {
        throw std::invalid_argument("expected and actual slot vectors must have equal nonzero length");
    }
    if (requested_slot_count > expected.size()) {
        throw std::invalid_argument("requested slot count exceeds vector length");
    }
    if (!std::isfinite(threshold) || threshold < 0.0) {
        throw std::invalid_argument("correctness threshold must be finite and nonnegative");
    }

    NumericalMetrics result;
    result.requested_slot_count = requested_slot_count;
    result.inactive_slot_count = expected.size() - requested_slot_count;
    result.compared_value_count = expected.size();
    result.threshold = threshold;

    long double squared_error_sum = 0.0L;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& wanted = expected[i];
        const auto& observed = actual[i];
        const bool finite = std::isfinite(wanted.real()) && std::isfinite(wanted.imag()) &&
                            std::isfinite(observed.real()) && std::isfinite(observed.imag());
        if (!finite) {
            ++result.nonfinite_value_count;
            ++result.mismatch_count;
            result.max_abs_error = std::numeric_limits<double>::infinity();
            if (result.failure_reason.empty()) {
                result.failure_reason = "nonfinite slot component";
                result.worst_index = i;
                result.worst_expected = wanted;
                result.worst_actual = observed;
            }
            continue;
        }
        ++result.finite_value_count;
        const double real_error = std::abs(observed.real() - wanted.real());
        const double imag_error = std::abs(observed.imag() - wanted.imag());
        const double magnitude_error = std::abs(observed - wanted);
        result.max_real_error = std::max(result.max_real_error, real_error);
        result.max_imag_error = std::max(result.max_imag_error, imag_error);
        result.component_max_abs_error = std::max(
            result.component_max_abs_error, std::max(real_error, imag_error));
        squared_error_sum += static_cast<long double>(magnitude_error) * magnitude_error;
        if (magnitude_error > result.max_abs_error) {
            result.max_abs_error = magnitude_error;
            result.worst_index = i;
            result.worst_expected = wanted;
            result.worst_actual = observed;
        }
        if (magnitude_error > threshold) {
            ++result.mismatch_count;
        }
    }

    if (result.nonfinite_value_count == 0) {
        result.rms_error = std::sqrt(
            static_cast<double>(squared_error_sum / result.compared_value_count));
    } else {
        result.rms_error = std::numeric_limits<double>::infinity();
    }
    result.passed = result.nonfinite_value_count == 0 && result.mismatch_count == 0 &&
                    result.max_abs_error <= threshold;
    if (!result.passed && result.failure_reason.empty()) {
        result.failure_reason = "slot error exceeds threshold";
    }
    return result;
}

ExactMetrics compare_residues(
    const std::vector<std::uint32_t>& expected,
    const std::vector<std::uint32_t>& actual,
    std::uint32_t modulus)
{
    if (modulus < 2) {
        throw std::invalid_argument("residue modulus must be at least two");
    }
    ExactMetrics result;
    result.expected_count = expected.size();
    result.actual_count = actual.size();
    result.compared_count = std::min(expected.size(), actual.size());
    result.first_mismatch_index = std::numeric_limits<std::size_t>::max();

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] >= modulus) {
            ++result.noncanonical_count;
        }
    }
    for (std::size_t i = 0; i < result.compared_count; ++i) {
        if (expected[i] != actual[i]) {
            if (result.first_mismatch_index == std::numeric_limits<std::size_t>::max()) {
                result.first_mismatch_index = i;
                result.first_expected = expected[i];
                result.first_actual = actual[i];
            }
            ++result.mismatch_count;
        }
    }
    if (expected.size() != actual.size()) {
        result.mismatch_count += expected.size() > actual.size()
            ? expected.size() - actual.size()
            : actual.size() - expected.size();
        if (result.first_mismatch_index == std::numeric_limits<std::size_t>::max()) {
            result.first_mismatch_index = result.compared_count;
        }
    }
    result.passed = result.mismatch_count == 0 && result.noncanonical_count == 0;
    return result;
}

} // namespace fpt2026
