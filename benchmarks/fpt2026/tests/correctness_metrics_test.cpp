#include "fpt2026_benchmark/correctness_metrics.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    using namespace fpt2026;

    std::vector<std::complex<double>> expected(4);
    expected[0] = {1.0, -1.0};
    std::vector<std::complex<double>> actual = expected;
    auto exact = evaluate_slots(expected, actual, 1, 0.1);
    require(exact.passed && exact.mismatch_count == 0 && exact.inactive_slot_count == 3,
            "exact slot comparison failed");

    actual[0] += std::complex<double>(0.06, 0.08);
    auto boundary = evaluate_slots(expected, actual, 1, 0.1);
    require(boundary.passed && std::abs(boundary.max_abs_error - 0.1) < 1e-12,
            "magnitude threshold boundary failed");

    actual[3] = {0.11, 0.0};
    auto inactive_failure = evaluate_slots(expected, actual, 1, 0.1);
    require(!inactive_failure.passed && inactive_failure.mismatch_count == 1 &&
                inactive_failure.worst_index == 3,
            "inactive-slot failure was not counted");

    actual = expected;
    actual[2] = {std::numeric_limits<double>::quiet_NaN(), 0.0};
    auto nonfinite = evaluate_slots(expected, actual, 1, 0.1);
    require(!nonfinite.passed && nonfinite.nonfinite_value_count == 1,
            "nonfinite slot was not rejected");

    std::vector<std::uint32_t> residues{0, 1, 16};
    auto residue_pass = compare_residues(residues, residues, 17);
    require(residue_pass.passed, "canonical exact residues failed");
    std::vector<std::uint32_t> bad{0, 2, 17};
    auto residue_fail = compare_residues(residues, bad, 17);
    require(!residue_fail.passed && residue_fail.mismatch_count == 2 &&
                residue_fail.noncanonical_count == 1 &&
                residue_fail.first_mismatch_index == 1,
            "residue mismatches were not fully counted");

    std::cout << "correctness_metrics_test: PASS\n";
    return 0;
}
