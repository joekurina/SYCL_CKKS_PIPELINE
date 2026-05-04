#include "SYCL_ntt.h"
#include "kernel_test_common.h"
#include "kernel_test_queue.h"
#include "kernel_test_reference.h"
#include <iostream>

#ifndef CKKS_TEST_P
#define CKKS_TEST_P 0
#endif

using namespace sycl;
using namespace sycl_ckks;

class FeedNTTBTask;
class DrainNTTBDownstreamTask;
class DrainNTTBExitTask;

static bool run_case(bool save_output)
{
    constexpr int P = CKKS_TEST_P;
    const uint32_t mod = sycl_ckks::test::test_modulus(P);
    const uint8_t mod_sel = get_modulus_selector(mod);

    std::vector<u32x4> input = sycl_ckks::test::make_u32x4_sequence(17000u + 1000u * P);
#ifdef FPGA_EMULATOR
    std::vector<u32x4> expected = sycl_ckks::test::reference_ntt(input, mod_sel);
#else
    std::vector<u32x4> expected(NUM_BLOCKS);
#endif
    std::vector<u32x4> downstream(NUM_BLOCKS);
    std::vector<u32x4> saved(NUM_BLOCKS);

    buffer<u32x4, 1> in_buf(input.data(), range<1>(NUM_BLOCKS));
    buffer<u32x4, 1> down_buf(downstream.data(), range<1>(NUM_BLOCKS));
    buffer<u32x4, 1> save_buf(saved.data(), range<1>(NUM_BLOCKS));

    auto q = sycl_ckks::test::make_queue();
    std::vector<event> events;

    events.push_back(q.submit([&](handler& h) {
        auto out = down_buf.template get_access<access::mode::write>(h);
        h.single_task<DrainNTTBDownstreamTask>([=]() {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) out[blk] = PipeSet<P>::NTTBToPolyAddPipe::read();
        });
    }));
    if (save_output) {
        events.push_back(q.submit([&](handler& h) {
            auto out = save_buf.template get_access<access::mode::write>(h);
            h.single_task<DrainNTTBExitTask>([=]() {
                for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) out[blk] = PipeSet<P>::NTTBToExitPipe::read();
            });
        }));
    }
    events.push_back(q.submit([&](handler& h) {
        auto in = in_buf.template get_access<access::mode::read>(h);
        h.single_task<FeedNTTBTask>([=]() {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) PipeSet<P>::ScaleReduceToNTTBPipe::write(in[blk]);
        });
    }));
    events.push_back(q.submit([&](handler& h) {
        NTTKernelB<P> kernel(mod_sel, save_output);
        kernel(h);
    }));

    for (auto& e : events) e.wait();

    bool ok = true;
#ifdef FPGA_EMULATOR
    ok &= sycl_ckks::test::expect_u32x4_vector("ntt_b_reference", expected, downstream);
#endif
    if (save_output) ok &= sycl_ckks::test::expect_u32x4_vector("ntt_b_save", downstream, saved);
    return ok;
}

int main()
{
    constexpr int P = CKKS_TEST_P;
    if (!run_case(false)) return 1;
    if (!run_case(true)) return 1;
    std::cout << "PASSED ckks_test_ntt_b_p" << P << "\n";
    return 0;
}
