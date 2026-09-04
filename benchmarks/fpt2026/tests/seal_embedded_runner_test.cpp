#include "fpt2026_benchmark/benchmark_vectors.hpp"
#include "fpt2026_benchmark/seal_embedded_runner.hpp"
#include "fpt2026_benchmark/seal_oracle.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv)
{
    using namespace fpt2026;
    if (argc != 3) {
        std::cerr << "usage: seal_embedded_runner_test /absolute/path/sk_8192.dat "
                     "/absolute/path/sk_8192_seal.dat\n";
        return 2;
    }

    StockSealOracle oracle(std::filesystem::path(argv[2]));
    SealEmbeddedRunner runner(std::filesystem::path(argv[1]), oracle);
    for (std::size_t message = 0; message < 2; ++message) {
        auto input = generate_benchmark_vector(VectorCaseId::real_short_mixed);
        auto seed = make_trial_seeds("cpu-runner-test", input.id, message, 0);
        auto result = runner.encrypt({input}, {seed});
        if (!result.passed || result.correctness.size() != 1 ||
            !result.correctness.front().passed) {
            throw std::runtime_error("SEAL-Embedded runner failed a verified message");
        }
    }
    std::cout << "seal_embedded_runner_test: PASS\n";
    return 0;
}
