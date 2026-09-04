#include "fpt2026_benchmark/benchmark_vectors.hpp"
#include "fpt2026_benchmark/correctness_metrics.hpp"
#include "fpt2026_benchmark/seal_oracle.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "seal/seal.h"

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main(int argc, char** argv)
{
    using namespace fpt2026;
    if (argc != 2) {
        std::cerr << "usage: seal_ciphertext_bridge_test /absolute/path/sk_8192_seal.dat\n";
        return 2;
    }

    StockSealOracle oracle(std::filesystem::path(argv[1]));
    for (const auto case_id : all_vector_cases()) {
        const auto input = generate_benchmark_vector(case_id);
        const auto source = oracle.encrypt_reference(input.slots);
        require(source.size() == 2 && source.coeff_modulus_size() == kDataModulusCount &&
                    source.is_ntt_form(),
                "source ciphertext metadata mismatch");

        std::array<std::vector<std::uint32_t>, kDataModulusCount> c0;
        std::array<std::vector<std::uint32_t>, kDataModulusCount> c1;
        for (std::size_t p = 0; p < kDataModulusCount; ++p) {
            c0[p].resize(kPolyModulusDegree);
            c1[p].resize(kPolyModulusDegree);
            for (std::size_t i = 0; i < kPolyModulusDegree; ++i) {
                const auto value0 = source.data(0)[p * kPolyModulusDegree + i];
                const auto value1 = source.data(1)[p * kPolyModulusDegree + i];
                require(value0 < kDataModuli[p] && value1 < kDataModuli[p],
                        "source ciphertext residue is not canonical");
                c0[p][i] = static_cast<std::uint32_t>(value0);
                c1[p][i] = static_cast<std::uint32_t>(value1);
            }
        }
        const auto rebuilt = oracle.assemble_ciphertext(c0, c1);
        const auto decoded = oracle.decrypt_decode(rebuilt);
        const auto metrics = evaluate_slots(
            input.slots, decoded, input.active_slots, kCorrectnessThreshold);
        require(metrics.passed, "rebuilt ciphertext failed decrypt/decode");

        if (case_id == VectorCaseId::real_impulse) {
            c0[0][0] = static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(c0[0][0]) + (UINT64_C(1) << 26)) %
                kDataModuli[0]);
            const auto corrupted = oracle.assemble_ciphertext(c0, c1);
            const auto corrupted_decoded = oracle.decrypt_decode(corrupted);
            const auto corrupted_metrics = evaluate_slots(
                input.slots, corrupted_decoded, input.active_slots,
                kCorrectnessThreshold);
            require(!corrupted_metrics.passed,
                    "deliberately corrupted ciphertext did not fail correctness");
        }
    }

    std::cout << "seal_ciphertext_bridge_test: PASS\n";
    return 0;
}
