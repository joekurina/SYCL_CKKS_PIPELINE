#define SYCL_CKKS_BENCHMARK_PRIVATE_TEST 1
#include "sycl_ckks_accelerator/SYCL_ckks_benchmark.h"

#include <atomic>
#include <cstdint>
#include <thread>

namespace {

using ClaimFunction = int (*)();

void count_post_guard_entry(void *context)
{
    static_cast<std::atomic<unsigned> *>(context)->fetch_add(1, std::memory_order_relaxed);
}

bool run_claim_race(ClaimFunction first, ClaimFunction second)
{
    for (unsigned trial = 0; trial < 100; ++trial) {
        SYCL_benchmark_private_reset_lifecycle();
        std::atomic<unsigned> post_guard_entries{0};
        std::atomic<unsigned> ready{0};
        std::atomic<bool> start{false};
        int first_won = 0;
        int second_won = 0;

        SYCL_benchmark_private_set_post_guard_hook(
            &count_post_guard_entry,
            &post_guard_entries);
        std::thread first_thread([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
            }
            first_won = first();
        });
        std::thread second_thread([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
            }
            second_won = second();
        });

        while (ready.load(std::memory_order_acquire) != 2) {
        }
        start.store(true, std::memory_order_release);
        first_thread.join();
        second_thread.join();
        SYCL_benchmark_private_set_post_guard_hook(nullptr, nullptr);

        if (first_won + second_won != 1 ||
            post_guard_entries.load(std::memory_order_relaxed) != 1) {
            return false;
        }
    }
    return true;
}

bool sequential_transitions_are_terminal()
{
    SYCL_benchmark_private_reset_lifecycle();
    if (!SYCL_benchmark_private_try_claim_legacy() ||
        SYCL_benchmark_private_try_claim_legacy() ||
        SYCL_benchmark_private_try_claim_session()) {
        return false;
    }
    SYCL_benchmark_private_mark_legacy_used();
    if (SYCL_benchmark_private_lifecycle_state() != SYCL_BENCHMARK_PRIVATE_LEGACY_USED ||
        SYCL_benchmark_private_try_claim_legacy() ||
        SYCL_benchmark_private_try_claim_session()) {
        return false;
    }

    SYCL_benchmark_private_reset_lifecycle();
    if (!SYCL_benchmark_private_try_claim_session() ||
        SYCL_benchmark_private_try_claim_session() ||
        SYCL_benchmark_private_try_claim_legacy()) {
        return false;
    }
    SYCL_benchmark_private_mark_session_closed();
    if (SYCL_benchmark_private_lifecycle_state() != SYCL_BENCHMARK_PRIVATE_CLOSED ||
        SYCL_benchmark_private_try_claim_session() ||
        SYCL_benchmark_private_try_claim_legacy()) {
        return false;
    }

    SYCL_benchmark_private_reset_lifecycle();
    if (!SYCL_benchmark_private_try_claim_session()) {
        return false;
    }
    SYCL_benchmark_private_mark_session_failed();
    return SYCL_benchmark_private_lifecycle_state() == SYCL_BENCHMARK_PRIVATE_FAILED &&
           !SYCL_benchmark_private_try_claim_session() &&
           !SYCL_benchmark_private_try_claim_legacy();
}

} // namespace

/* Host-only lifecycle FPGA Test: the private seams never create a SYCL queue. */
int main()
{
    if (!sequential_transitions_are_terminal()) {
        return 1;
    }
    if (!run_claim_race(
            &SYCL_benchmark_private_try_claim_legacy,
            &SYCL_benchmark_private_try_claim_legacy)) {
        return 2;
    }
    if (!run_claim_race(
            &SYCL_benchmark_private_try_claim_session,
            &SYCL_benchmark_private_try_claim_session)) {
        return 3;
    }
    if (!run_claim_race(
            &SYCL_benchmark_private_try_claim_legacy,
            &SYCL_benchmark_private_try_claim_session)) {
        return 4;
    }
    return 0;
}
