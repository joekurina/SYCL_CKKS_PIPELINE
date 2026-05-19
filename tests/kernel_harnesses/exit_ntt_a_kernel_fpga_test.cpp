#include "harness_common.h"

#include "SYCL_pipeline_exit.h"
#include "SYCL_pipes.h"

#include <vector>

using namespace sycl_ckks;

namespace sycl_ckks::harness {

template <int P>
class FeedExitNTTAKernelTask;

template <int P>
class FeedExitNTTAKernel {
public:
    void operator()(sycl::handler& h) const
    {
        h.single_task<FeedExitNTTAKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                PipeSet<P>::NTTAToExitPipe::write(pattern_u32x4(blk, 20u));
            }
        });
    }
};

} // namespace sycl_ckks::harness

int main()
{
    sycl_ckks::harness::host_debug("exit-ntt-a: preparing host buffers");
    constexpr int P = 0;
    std::vector<u32x4> output(NUM_BLOCKS);
    {
        sycl::buffer<u32x4, 1> output_buf(output.data(), sycl::range<1>(NUM_BLOCKS));

        sycl_ckks::harness::host_debug("exit-ntt-a: creating SYCL queue");
        auto q = sycl_ckks::harness::make_queue();
        sycl_ckks::harness::host_debug("exit-ntt-a: submitting exit/drain kernel");
        auto exit_event = q.submit([&](sycl::handler& h) {
            ExitNTTASKernel<P> kernel(output_buf);
            kernel(h);
        });
        sycl_ckks::harness::host_debug("exit-ntt-a: submitting feeder kernel");
        auto feed_event = q.submit([&](sycl::handler& h) {
            sycl_ckks::harness::FeedExitNTTAKernel<P> kernel;
            kernel(h);
        });

        (void)exit_event;
        (void)feed_event;
        sycl_ckks::harness::host_debug("exit-ntt-a: waiting for submitted kernels");
        q.wait_and_throw();
    }

    sycl_ckks::harness::host_debug("exit-ntt-a: kernels completed; verifying host outputs");

    for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(output[blk], sycl_ckks::harness::pattern_u32x4(blk, 20u)), "ExitNTTASKernel output mismatch");
    }
    sycl_ckks::harness::host_debug("exit-ntt-a: PASS");
    return 0;
}
