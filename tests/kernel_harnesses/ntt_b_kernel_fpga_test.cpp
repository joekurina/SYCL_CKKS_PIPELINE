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
    sycl_ckks::harness::host_debug("ntt-b: preparing host buffers");
    constexpr int P = 0;
    const uint8_t modulus_selector = get_modulus_selector(sycl_ckks::harness::default_modulus(P));
    std::vector<u32x4> output(NUM_BLOCKS);
    sycl::buffer<u32x4, 1> output_buf(output.data(), sycl::range<1>(NUM_BLOCKS));

    sycl_ckks::harness::host_debug("ntt-b: creating SYCL queue");
    auto q = sycl_ckks::harness::make_queue();
    sycl_ckks::harness::host_debug("ntt-b: submitting drain kernel");
    auto drain_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::DrainNTTBKernel<P> kernel(output_buf);
        kernel(h);
    });
    sycl_ckks::harness::host_debug("ntt-b: submitting NTT-B kernel");
    auto ntt_event = q.submit([&](sycl::handler& h) {
        NTTKernelB<P> kernel(modulus_selector, false);
        kernel(h);
    });
    sycl_ckks::harness::host_debug("ntt-b: submitting feeder kernel");
    auto feed_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::FeedNTTBKernel<P> kernel;
        kernel(h);
    });

    (void)drain_event;
    (void)ntt_event;
    (void)feed_event;
    sycl_ckks::harness::host_debug("ntt-b: waiting for submitted kernels");
    q.wait_and_throw();
    sycl_ckks::harness::host_debug("ntt-b: PASS");

    return 0;
}
