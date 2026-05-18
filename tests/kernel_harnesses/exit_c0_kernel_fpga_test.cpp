#include "harness_common.h"

#include "SYCL_pipeline_exit.h"
#include "SYCL_pipes.h"

#include <vector>

using namespace sycl_ckks;

namespace sycl_ckks::harness {

template <int P>
class FeedExitC0KernelTask;

template <int P>
class FeedExitC0Kernel {
public:
    void operator()(sycl::handler& h) const
    {
        h.single_task<FeedExitC0KernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                PipeSet<P>::PolyAddToExitPipe::write(pattern_u32x4(blk, 10u));
            }
        });
    }
};

} // namespace sycl_ckks::harness

int main()
{
    constexpr int P = 0;
    std::vector<u32x4> output(NUM_BLOCKS);
    sycl::buffer<u32x4, 1> output_buf(output.data(), sycl::range<1>(NUM_BLOCKS));

    auto q = sycl_ckks::harness::make_queue();
    auto exit_event = q.submit([&](sycl::handler& h) {
        ExitC0Kernel<P> kernel(output_buf);
        kernel(h);
    });
    auto feed_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::FeedExitC0Kernel<P> kernel;
        kernel(h);
    });

    feed_event.wait();
    exit_event.wait();

    for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(output[blk], sycl_ckks::harness::pattern_u32x4(blk, 10u)), "ExitC0Kernel output mismatch");
    }
    return 0;
}
