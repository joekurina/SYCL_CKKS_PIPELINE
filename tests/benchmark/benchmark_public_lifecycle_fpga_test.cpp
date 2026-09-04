#define SYCL_CKKS_BENCHMARK_PRIVATE_TEST 1
#include "sycl_ckks_accelerator/SYCL_ckks_benchmark.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::array<uint32_t, SYCL_NUM_MODULI> MODULI = {
    1053818881u, 1054015489u, 1054212097u,
    1055260673u, 1056178177u, 1056440321u,
};
constexpr std::array<uint32_t, SYCL_NUM_MODULI * 2> RATIOS = {
    0x135bf4bau, 0x00000004u,
    0x132a2218u, 0x00000004u,
    0x12f85437u, 0x00000004u,
    0x11ef051eu, 0x00000004u,
    0x11074e88u, 0x00000004u,
    0x10c52d4au, 0x00000004u,
};
constexpr std::array<double, SYCL_NUM_MODULI> SCALES = {
    33554432.0, 33554432.0, 33554432.0,
    33554432.0, 33554432.0, 33554432.0,
};

enum class Scenario : uint32_t {
    SESSION_AFTER_LEGACY,
    SECOND_SESSION_ACTIVE,
    SESSION_AFTER_CLOSE,
    SESSION_AFTER_FAILURE,
    SECOND_LEGACY,
    LEGACY_WHILE_SESSION_ACTIVE,
    LEGACY_AFTER_CLOSE,
    LEGACY_AFTER_FAILURE,
};

void seed_state(Scenario scenario)
{
    SYCL_benchmark_private_reset_lifecycle();
    switch (scenario) {
    case Scenario::SESSION_AFTER_LEGACY:
    case Scenario::SECOND_LEGACY:
        if (!SYCL_benchmark_private_try_claim_legacy()) {
            std::_Exit(90);
        }
        SYCL_benchmark_private_mark_legacy_used();
        break;
    case Scenario::SECOND_SESSION_ACTIVE:
    case Scenario::LEGACY_WHILE_SESSION_ACTIVE:
        if (!SYCL_benchmark_private_try_claim_session()) {
            std::_Exit(91);
        }
        break;
    case Scenario::SESSION_AFTER_CLOSE:
    case Scenario::LEGACY_AFTER_CLOSE:
        if (!SYCL_benchmark_private_try_claim_session()) {
            std::_Exit(92);
        }
        SYCL_benchmark_private_mark_session_closed();
        break;
    case Scenario::SESSION_AFTER_FAILURE:
    case Scenario::LEGACY_AFTER_FAILURE:
        if (!SYCL_benchmark_private_try_claim_session()) {
            std::_Exit(93);
        }
        SYCL_benchmark_private_mark_session_failed();
        break;
    }
}

[[noreturn]] void invoke_prohibited_legacy(Scenario scenario)
{
    seed_state(scenario);
    std::vector<complex_double> encoding(SYCL_POLY_N);
    std::vector<int8_t> errors(SYCL_POLY_N);
    std::array<std::vector<uint32_t>, SYCL_NUM_MODULI> secrets;
    std::array<std::vector<uint32_t>, SYCL_NUM_MODULI> uniforms;
    std::array<std::vector<uint32_t>, SYCL_NUM_MODULI> c0;
    std::array<const uint32_t *, SYCL_NUM_MODULI> secret_ptrs{};
    std::array<const uint32_t *, SYCL_NUM_MODULI> uniform_ptrs{};
    std::array<uint32_t *, SYCL_NUM_MODULI> c0_ptrs{};
    for (size_t p = 0; p < SYCL_NUM_MODULI; ++p) {
        secrets[p].resize(SYCL_POLY_N);
        uniforms[p].resize(SYCL_POLY_N);
        c0[p].resize(SYCL_POLY_N);
        secret_ptrs[p] = secrets[p].data();
        uniform_ptrs[p] = uniforms[p].data();
        c0_ptrs[p] = c0[p].data();
    }
    SYCL_encrypt(
        SYCL_POLY_N,
        SCALES.data(),
        MODULI.data(),
        RATIOS.data(),
        encoding.data(),
        errors.data(),
        secret_ptrs.data(),
        uniform_ptrs.data(),
        c0_ptrs.data(),
        nullptr,
        nullptr,
        nullptr,
        nullptr);
    std::_Exit(94); // A prohibited legacy call must terminate before returning.
}

int invoke_prohibited_session(Scenario scenario)
{
    seed_state(scenario);
    const SYCLBenchmarkConfig config{
        SYCL_BENCHMARK_ABI_VERSION,
        sizeof(SYCLBenchmarkConfig),
        SYCL_POLY_N,
        SYCL_NUM_MODULI,
        1,
        0,
        0,
        1,
    };
    SYCLBenchmarkSession *session = nullptr;
    size_t capacity = 0;
    char error[256]{};
    const int status = SYCL_benchmark_session_create(
        &config,
        SCALES.data(),
        MODULI.data(),
        RATIOS.data(),
        &session,
        &capacity,
        error,
        sizeof(error));
    return status == SYCL_BENCHMARK_STATUS_LIFECYCLE_ERROR && session == nullptr ? 0 : 95;
}

bool run_child(Scenario scenario, bool legacy)
{
    const pid_t pid = fork();
    if (pid < 0) {
        return false;
    }
    if (pid == 0) {
        alarm(5);
        if (legacy) {
            invoke_prohibited_legacy(scenario);
        }
        std::_Exit(invoke_prohibited_session(scenario));
    }
    int wait_status = 0;
    if (waitpid(pid, &wait_status, 0) != pid || !WIFEXITED(wait_status)) {
        return false;
    }
    const int exit_status = WEXITSTATUS(wait_status);
    return legacy ? exit_status == static_cast<int>(SYCL_BENCHMARK_STATUS_INVALID_ARGUMENT)
                  : exit_status == 0;
}

} // namespace

/*
 * Source-only public guard FPGA Test. It never permits a winner to create a
 * queue. Task 15 later exercises successful first calls and the same misuse
 * cases in fresh watchdog-bounded emulator children.
 */
int main()
{
    if (!run_child(Scenario::SESSION_AFTER_LEGACY, false)) return 1;
    if (!run_child(Scenario::SECOND_SESSION_ACTIVE, false)) return 2;
    if (!run_child(Scenario::SESSION_AFTER_CLOSE, false)) return 3;
    if (!run_child(Scenario::SESSION_AFTER_FAILURE, false)) return 4;
    if (!run_child(Scenario::SECOND_LEGACY, true)) return 5;
    if (!run_child(Scenario::LEGACY_WHILE_SESSION_ACTIVE, true)) return 6;
    if (!run_child(Scenario::LEGACY_AFTER_CLOSE, true)) return 7;
    if (!run_child(Scenario::LEGACY_AFTER_FAILURE, true)) return 8;
    return 0;
}
