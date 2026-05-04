#include "SYCL_scale_and_reduce.h"
#include "kernel_test_common.h"
#include "kernel_test_queue.h"
#include <cmath>
#include <iostream>

#ifndef CKKS_TEST_P
#define CKKS_TEST_P 0
#endif

using namespace sycl;
using namespace sycl_ckks;

class FeedScaleEncTask;
class FeedScaleErrTask;
class DrainScaleTask;

int main()
{
    constexpr int P = CKKS_TEST_P;
    constexpr double scale = 4096.0;
    const uint32_t mod = sycl_ckks::test::test_modulus(P);
    uint32_t cr[2];
    sycl_ckks::test::test_const_ratio(P, cr);

    std::vector<encoding_block> enc(NUM_BLOCKS);
    std::vector<i8x4> err(NUM_BLOCKS);
    std::vector<u32x4> expected(NUM_BLOCKS);
    std::vector<u32x4> actual(NUM_BLOCKS);

    for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        enc[blk] = sycl_ckks::test::make_encoding_block(static_cast<double>(blk * LANES));
        err[blk] = sycl_ckks::test::make_i8x4(static_cast<int>((blk % 13) - 6));
        for (size_t lane = 0; lane < LANES; ++lane) {
            double scaled = std::round(sycl_ckks::test::lane_at(enc[blk], lane).real() * scale / static_cast<double>(POLY_N));
            int64_t int_val = static_cast<int64_t>(scaled) + sycl_ckks::test::lane_at(err[blk], lane);
            sycl_ckks::test::set_lane(expected[blk], lane, barrett_reduce_64_core(int_val, mod, cr[0], cr[1], false));
        }
    }

    buffer<encoding_block, 1> enc_buf(enc.data(), range<1>(NUM_BLOCKS));
    buffer<i8x4, 1> err_buf(err.data(), range<1>(NUM_BLOCKS));
    buffer<u32x4, 1> out_buf(actual.data(), range<1>(NUM_BLOCKS));

    auto q = sycl_ckks::test::make_queue();
    std::vector<event> events;

    events.push_back(q.submit([&](handler& h) {
        auto out = out_buf.template get_access<access::mode::write>(h);
        h.single_task<DrainScaleTask>([=]() {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) out[blk] = PipeSet<P>::ScaleReduceToNTTBPipe::read();
        });
    }));
    events.push_back(q.submit([&](handler& h) {
        auto in = enc_buf.template get_access<access::mode::read>(h);
        h.single_task<FeedScaleEncTask>([=]() {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) IFFTToScaleReducePipes::PipeAt<P>::write(in[blk]);
        });
    }));
    events.push_back(q.submit([&](handler& h) {
        auto in = err_buf.template get_access<access::mode::read>(h);
        h.single_task<FeedScaleErrTask>([=]() {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) ErrorToScaleReducePipes::PipeAt<P>::write(in[blk]);
        });
    }));
    events.push_back(q.submit([&](handler& h) {
        ScaleAndReduceKernel<P> kernel(scale, mod, cr);
        kernel(h);
    }));

    for (auto& e : events) e.wait();
    if (!sycl_ckks::test::expect_u32x4_vector("scale_reduce", expected, actual)) return 1;
    std::cout << "PASSED ckks_test_scale_reduce_p" << P << "\n";
    return 0;
}
