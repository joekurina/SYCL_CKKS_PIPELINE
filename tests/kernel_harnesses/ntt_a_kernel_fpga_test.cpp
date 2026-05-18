#include "harness_common.h"

#include "SYCL_ntt.h"
#include "SYCL_pipes.h"

#include <vector>

using namespace sycl_ckks;

namespace sycl_ckks::harness {

template <int P>
class FeedNTTAKernelTask;

template <int P>
class FeedNTTAKernel {
public:
    void operator()(sycl::handler& h) const
    {
        h.single_task<FeedNTTAKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                PipeSet<P>::EntryToNTTAPipe::write(pattern_u32x4(blk, 400u));
            }
        });
    }
};

template <int P>
class DrainNTTAKernelTask;

template <int P>
class DrainNTTAKernel {
private:
    mutable sycl::buffer<u32x4, 1> output_buf;

public:
    explicit DrainNTTAKernel(sycl::buffer<u32x4, 1>& buf) : output_buf(buf) {}

    void operator()(sycl::handler& h) const
    {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);
        h.single_task<DrainNTTAKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                output[blk] = PipeSet<P>::NTTAToPolyMultNegPipe::read();
            }
        });
    }
};

} // namespace sycl_ckks::harness

int main()
{
    sycl_ckks::harness::host_debug("ntt-a: preparing host buffers");
    constexpr int P = 0;
    const uint8_t modulus_selector = get_modulus_selector(sycl_ckks::harness::default_modulus(P));
    std::vector<u32x4> output(NUM_BLOCKS);
    sycl::buffer<u32x4, 1> output_buf(output.data(), sycl::range<1>(NUM_BLOCKS));

    sycl_ckks::harness::host_debug("ntt-a: creating SYCL queue");
    auto q = sycl_ckks::harness::make_queue();
    sycl_ckks::harness::host_debug("ntt-a: submitting drain kernel");
    auto drain_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::DrainNTTAKernel<P> kernel(output_buf);
        kernel(h);
    });
    sycl_ckks::harness::host_debug("ntt-a: submitting NTT-A kernel");
    auto ntt_event = q.submit([&](sycl::handler& h) {
        NTTKernelA<P> kernel(modulus_selector, false);
        kernel(h);
    });
    sycl_ckks::harness::host_debug("ntt-a: submitting feeder kernel");
    auto feed_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::FeedNTTAKernel<P> kernel;
        kernel(h);
    });

    (void)drain_event;
    (void)ntt_event;
    (void)feed_event;
    sycl_ckks::harness::host_debug("ntt-a: waiting for submitted kernels");
    q.wait_and_throw();
    sycl_ckks::harness::host_debug("ntt-a: PASS");

    return 0;
}
