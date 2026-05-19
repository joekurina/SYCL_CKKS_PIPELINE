#include "harness_common.h"

#include "SYCL_pipeline_exit.h"
#include "SYCL_pipes.h"

#include <vector>

using namespace sycl_ckks;

namespace sycl_ckks::harness {

template <int P>
class FeedExitNTTBKernelTask;

template <int P>
class FeedExitNTTBKernel {
public:
    void operator()(sycl::handler& h) const
    {
        h.single_task<FeedExitNTTBKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                PipeSet<P>::NTTBToExitPipe::write(pattern_u32x4(blk, 30u));
            }
        });
    }
};

} // namespace sycl_ckks::harness

int main()
{
    sycl_ckks::harness::host_debug("exit-ntt-b: preparing host buffers");
    constexpr int P = 0;
    std::vector<u32x4> output(NUM_BLOCKS);
    {
        sycl::buffer<u32x4, 1> output_buf(output.data(), sycl::range<1>(NUM_BLOCKS));

        sycl_ckks::harness::host_debug("exit-ntt-b: creating SYCL queue");
        auto q = sycl_ckks::harness::make_queue();
        sycl_ckks::harness::host_debug("exit-ntt-b: submitting exit/drain kernel");
        auto exit_event = q.submit([&](sycl::handler& h) {
            ExitNTTBKernel<P> kernel(output_buf);
            kernel(h);
        });
        sycl_ckks::harness::host_debug("exit-ntt-b: submitting feeder kernel");
        auto feed_event = q.submit([&](sycl::handler& h) {
            sycl_ckks::harness::FeedExitNTTBKernel<P> kernel;
            kernel(h);
        });

        (void)exit_event;
        (void)feed_event;
        sycl_ckks::harness::host_debug("exit-ntt-b: waiting for submitted kernels");
        q.wait_and_throw();
    }

    sycl_ckks::harness::host_debug("exit-ntt-b: kernels completed; verifying host outputs");

    for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(output[blk], sycl_ckks::harness::pattern_u32x4(blk, 30u)), "ExitNTTBKernel output mismatch");
    }
    sycl_ckks::harness::host_debug("exit-ntt-b: PASS");
    return 0;
}
