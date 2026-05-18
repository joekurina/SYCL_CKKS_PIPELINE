#include "harness_common.h"

#include "SYCL_ntt.h"
#include "SYCL_pipes.h"

#include <vector>

using namespace sycl_ckks;

namespace sycl_ckks::harness {

template <int P>
class FeedNTTBKernelTask;

template <int P>
class FeedNTTBKernel {
public:
    void operator()(sycl::handler& h) const
    {
        h.single_task<FeedNTTBKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                PipeSet<P>::ScaleReduceToNTTBPipe::write(pattern_u32x4(blk, 500u));
            }
        });
    }
};

template <int P>
class DrainNTTBKernelTask;

template <int P>
class DrainNTTBKernel {
private:
    mutable sycl::buffer<u32x4, 1> output_buf;

public:
    explicit DrainNTTBKernel(sycl::buffer<u32x4, 1>& buf) : output_buf(buf) {}

    void operator()(sycl::handler& h) const
    {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);
        h.single_task<DrainNTTBKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                output[blk] = PipeSet<P>::NTTBToPolyAddPipe::read();
            }
        });
    }
};

} // namespace sycl_ckks::harness

int main()
{
    constexpr int P = 0;
    const uint8_t modulus_selector = get_modulus_selector(sycl_ckks::harness::default_modulus(P));
    std::vector<u32x4> output(NUM_BLOCKS);
    sycl::buffer<u32x4, 1> output_buf(output.data(), sycl::range<1>(NUM_BLOCKS));

    auto q = sycl_ckks::harness::make_queue();
    auto drain_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::DrainNTTBKernel<P> kernel(output_buf);
        kernel(h);
    });
    auto ntt_event = q.submit([&](sycl::handler& h) {
        NTTKernelB<P> kernel(modulus_selector, false);
        kernel(h);
    });
    auto feed_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::FeedNTTBKernel<P> kernel;
        kernel(h);
    });

    feed_event.wait();
    ntt_event.wait();
    drain_event.wait();

    return 0;
}
