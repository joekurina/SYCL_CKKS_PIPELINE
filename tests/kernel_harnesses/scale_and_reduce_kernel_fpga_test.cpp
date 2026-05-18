#include "harness_common.h"

#include "SYCL_pipes.h"
#include "SYCL_scale_and_reduce.h"

#include <array>
#include <cmath>
#include <vector>

using namespace sycl_ckks;

namespace sycl_ckks::harness {

template <int P>
class FeedScaleAndReduceKernelTask;

template <int P>
class FeedScaleAndReduceKernel {
public:
    void operator()(sycl::handler& h) const
    {
        h.single_task<FeedScaleAndReduceKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                IFFTToScaleReducePipes::PipeAt<P>::write(pattern_encoding(blk));
                ErrorToScaleReducePipes::PipeAt<P>::write(pattern_i8x4(blk));
            }
        });
    }
};

template <int P>
class DrainScaleAndReduceKernelTask;

template <int P>
class DrainScaleAndReduceKernel {
private:
    mutable sycl::buffer<u32x4, 1> output_buf;

public:
    explicit DrainScaleAndReduceKernel(sycl::buffer<u32x4, 1>& buf) : output_buf(buf) {}

    void operator()(sycl::handler& h) const
    {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);
        h.single_task<DrainScaleAndReduceKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                output[blk] = PipeSet<P>::ScaleReduceToNTTBPipe::read();
            }
        });
    }
};

inline uint32_t expected_scale_reduce_lane(size_t block, size_t lane, double scale, uint32_t mod, const uint32_t ratio[2])
{
    const auto enc = pattern_encoding(block);
    const auto err = pattern_i8x4(block);
    double real = 0.0;
    int8_t noise = 0;
    switch (lane) {
        case 0: real = enc.element0.real(); noise = err.element0; break;
        case 1: real = enc.element1.real(); noise = err.element1; break;
        case 2: real = enc.element2.real(); noise = err.element2; break;
        default: real = enc.element3.real(); noise = err.element3; break;
    }
    const double n_inv = scale / static_cast<double>(POLY_N);
    const auto rounded = static_cast<int64_t>(std::llround(real * n_inv));
    return barrett_reduce_64_core(rounded + noise, mod, ratio[0], ratio[1], false);
}

} // namespace sycl_ckks::harness

int main()
{
    constexpr int P = 0;
    constexpr double scale = 1048576.0;
    const uint32_t mod = sycl_ckks::harness::default_modulus(P);
    uint32_t ratio[2];
    sycl_ckks::harness::default_const_ratio(P, ratio);

    std::vector<u32x4> output(NUM_BLOCKS);
    sycl::buffer<u32x4, 1> output_buf(output.data(), sycl::range<1>(NUM_BLOCKS));

    auto q = sycl_ckks::harness::make_queue();
    auto drain_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::DrainScaleAndReduceKernel<P> kernel(output_buf);
        kernel(h);
    });
    auto scale_event = q.submit([&](sycl::handler& h) {
        ScaleAndReduceKernel<P> kernel(scale, mod, ratio);
        kernel(h);
    });
    auto feed_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::FeedScaleAndReduceKernel<P> kernel;
        kernel(h);
    });

    feed_event.wait();
    scale_event.wait();
    drain_event.wait();

    for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        u32x4 expected;
        expected.element0 = sycl_ckks::harness::expected_scale_reduce_lane(blk, 0, scale, mod, ratio);
        expected.element1 = sycl_ckks::harness::expected_scale_reduce_lane(blk, 1, scale, mod, ratio);
        expected.element2 = sycl_ckks::harness::expected_scale_reduce_lane(blk, 2, scale, mod, ratio);
        expected.element3 = sycl_ckks::harness::expected_scale_reduce_lane(blk, 3, scale, mod, ratio);
        sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(output[blk], expected), "ScaleAndReduceKernel output mismatch");
    }
    return 0;
}
