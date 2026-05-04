#include "SYCL_pipeline_exit.h"
#include "kernel_test_common.h"
#include "kernel_test_queue.h"
#include <iostream>

#ifndef CKKS_TEST_P
#define CKKS_TEST_P 0
#endif

using namespace sycl;
using namespace sycl_ckks;

class FeedExitNTTBTask;

int main()
{
    constexpr int P = CKKS_TEST_P;
    auto input = sycl_ckks::test::make_u32x4_sequence(500000u + 10000u * P);
    std::vector<u32x4> actual(NUM_BLOCKS);

    buffer<u32x4, 1> in_buf(input.data(), range<1>(NUM_BLOCKS));
    buffer<u32x4, 1> out_buf(actual.data(), range<1>(NUM_BLOCKS));

    auto q = sycl_ckks::test::make_queue();
    std::vector<event> events;

    events.push_back(q.submit([&](handler& h) {
        ExitNTTBKernel<P> kernel(out_buf);
        kernel(h);
    }));
    events.push_back(q.submit([&](handler& h) {
        auto in = in_buf.template get_access<access::mode::read>(h);
        h.single_task<FeedExitNTTBTask>([=]() {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) PipeSet<P>::NTTBToExitPipe::write(in[blk]);
        });
    }));

    for (auto& e : events) e.wait();
    if (!sycl_ckks::test::expect_u32x4_vector("exit_ntt_b", input, actual)) return 1;
    std::cout << "PASSED ckks_test_exit_ntt_b_p" << P << "\n";
    return 0;
}
