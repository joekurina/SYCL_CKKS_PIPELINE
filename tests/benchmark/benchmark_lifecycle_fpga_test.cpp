#define SYCL_CKKS_BENCHMARK_PRIVATE_TEST 1
#include "sycl_ckks_accelerator/SYCL_ckks_benchmark.h"

#include <atomic>
#include <cstdint>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

using ClaimFunction = int (*)();

void count_post_guard_entry(void *context)
{
    static_cast<std::atomic<unsigned> *>(context)->fetch_add(1, std::memory_order_relaxed);
}

struct RaceObservation {
    int first_won;
    int second_won;
    unsigned post_guard_entries;
};

RaceObservation run_claim_race_once(ClaimFunction first, ClaimFunction second)
{
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
    return RaceObservation{
        first_won,
        second_won,
        post_guard_entries.load(std::memory_order_relaxed)};
}

bool run_claim_race(ClaimFunction first, ClaimFunction second)
{
    for (unsigned trial = 0; trial < 100; ++trial) {
        int channel[2];
        if (pipe(channel) != 0) {
            return false;
        }
        const pid_t child = fork();
        if (child < 0) {
            close(channel[0]);
            close(channel[1]);
            return false;
        }
        if (child == 0) {
            close(channel[0]);
            alarm(5);
            const RaceObservation observation = run_claim_race_once(first, second);
            const ssize_t written = write(channel[1], &observation, sizeof(observation));
            close(channel[1]);
            _exit(written == static_cast<ssize_t>(sizeof(observation)) ? 0 : 10);
        }
        close(channel[1]);
        RaceObservation observation{};
        const ssize_t received = read(channel[0], &observation, sizeof(observation));
        close(channel[0]);
        int child_status = 0;
        if (waitpid(child, &child_status, 0) != child ||
            !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0 ||
            received != static_cast<ssize_t>(sizeof(observation)) ||
            observation.first_won + observation.second_won != 1 ||
            observation.post_guard_entries != 1) {
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
