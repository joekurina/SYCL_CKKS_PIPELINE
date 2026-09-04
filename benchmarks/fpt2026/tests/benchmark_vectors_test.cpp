#include "fpt2026_benchmark/benchmark_vectors.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string hex(const std::vector<std::uint8_t>& bytes)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto byte : bytes) {
        output << std::setw(2) << static_cast<unsigned>(byte);
    }
    return output.str();
}

} // namespace

int main()
{
    using namespace fpt2026;

    require(hex(shake256("", 64)) ==
        "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f"
        "d75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be",
        "SHAKE256 empty-string vector mismatch");

    const auto short_real = generate_benchmark_vector(VectorCaseId::real_short_mixed);
    const double expected_real[] = {
        1.25, -0.5, 0.125, -1.75, 2.5, -0.03125, 0.875, -2.125};
    require(short_real.active_slots == 8 && short_real.slots.size() == kSlotCount,
            "real_short_mixed shape mismatch");
    for (std::size_t i = 0; i < 8; ++i) {
        require(short_real.slots[i] == std::complex<double>(expected_real[i], 0.0),
                "real_short_mixed value mismatch");
    }
    require(std::all_of(short_real.slots.begin() + 8, short_real.slots.end(),
                        [](auto value) { return value == std::complex<double>{}; }),
            "inactive real slots are not zero");

    const auto short_complex = generate_benchmark_vector(VectorCaseId::complex_short_mixed);
    require(short_complex.slots.front() == std::complex<double>(1.25, 0.5),
            "complex first value mismatch");
    require(short_complex.slots[7] == std::complex<double>(-2.125, 0.625),
            "complex last active value mismatch");

    for (const auto id : all_vector_cases()) {
        const auto first = generate_benchmark_vector(id);
        const auto second = generate_benchmark_vector(id);
        require(first.id == vector_case_name(id) && first.slots == second.slots,
                "case generation is not deterministic");
        require(first.active_slots == active_slot_count(id),
                "active-slot count mismatch");
        for (std::size_t i = 0; i < first.active_slots; ++i) {
            require(first.slots[i] != std::complex<double>{},
                    "required active slot is zero");
        }
        const auto artifact1 = serialize_vector_artifact(first);
        const auto artifact2 = serialize_vector_artifact(second);
        require(artifact1 == artifact2 && artifact1.size() > 16 * kSlotCount,
                "vector artifact is not canonical and deterministic");
    }

    std::vector<std::uint16_t> identity(kPolyModulusDegree);
    for (std::size_t i = 0; i < identity.size(); ++i) {
        identity[i] = static_cast<std::uint16_t>(i);
    }
    const auto embedded = make_conjugate_embedding(short_complex, identity);
    for (std::size_t i = 0; i < kSlotCount; ++i) {
        require(embedded[i] == short_complex.slots[i], "direct embedding mismatch");
        require(embedded[i + kSlotCount] == std::conj(short_complex.slots[i]),
                "conjugate embedding mismatch");
    }

    std::cout << "benchmark_vectors_test: PASS\n";
    return 0;
}
