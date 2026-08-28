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
private:
    mutable sycl::buffer<u32x4, 1> input_buf;

public:
    explicit FeedExitC0Kernel(sycl::buffer<u32x4, 1>& input)
        : input_buf(input) {}

    void operator()(sycl::handler& h) const
    {
        auto input = input_buf.template get_access<sycl::access::mode::read>(h);

        h.single_task<FeedExitC0KernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t i = 0; i < NUM_BLOCKS; ++i) {
                PipeSet<P>::PolyAddToExitPipe::write(input[i]);
            }
        });
    }
};

} // namespace sycl_ckks::harness

int main()
{
    sycl_ckks::harness::host_debug("exit-c0: creating simple host test vectors");
    constexpr int P = 0;

    // The exit kernel should copy exactly one input pipe into the output buffer.
    // Keep the input obvious: one u32x4 block per index i, derived from i.
    std::vector<u32x4> input(NUM_BLOCKS);
    std::vector<u32x4> output(NUM_BLOCKS);
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        input[i] = sycl_ckks::harness::make_test_u32x4(i);
    }

    {
        sycl::buffer<u32x4, 1> input_buf(input.data(), sycl::range<1>(NUM_BLOCKS));
        sycl::buffer<u32x4, 1> output_buf(output.data(), sycl::range<1>(NUM_BLOCKS));

        auto q = sycl_ckks::harness::make_queue();

        // ExitC0Kernel is the kernel under test. It drains PipeSet<P>::PolyAddToExitPipe into
        // output_buf. The feeder writes the same pipe from input_buf.
        auto exit_event = q.submit([&](sycl::handler& h) {
            ExitC0Kernel<P, NUM_BLOCKS> kernel(output_buf);
            kernel(h);
        });
        auto feed_event = q.submit([&](sycl::handler& h) {
            sycl_ckks::harness::FeedExitC0Kernel<P> kernel(input_buf);
            kernel(h);
        });

        (void)exit_event;
        (void)feed_event;
        q.wait_and_throw();
    }

    sycl_ckks::harness::host_debug("exit-c0: comparing output with expected input copy");
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(output[i], input[i]), "ExitC0Kernel output mismatch");
    }

    sycl_ckks::harness::host_debug("exit-c0: PASS");
    return 0;
}
