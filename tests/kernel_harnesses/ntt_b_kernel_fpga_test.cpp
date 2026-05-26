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
private:
    mutable sycl::buffer<u32x4, 1> input_buf;

public:
    explicit FeedNTTBKernel(sycl::buffer<u32x4, 1>& input)
        : input_buf(input) {}

    void operator()(sycl::handler& h) const
    {
        auto input = input_buf.template get_access<sycl::access::mode::read>(h);

        h.single_task<FeedNTTBKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t i = 0; i < NUM_BLOCKS; ++i) {
                PipeSet<P>::ScaleReduceToNTTBPipe::write(input[i]);
            }
        });
    }
};

template <int P>
class DrainNTTBKernelTask;

template <int P>
class DrainNTTBKernel {
private:
    mutable sycl::buffer<u32x4, 1> downstream_buf;
    mutable sycl::buffer<u32x4, 1> exit_buf;

public:
    DrainNTTBKernel(sycl::buffer<u32x4, 1>& downstream,
                  sycl::buffer<u32x4, 1>& exit)
        : downstream_buf(downstream), exit_buf(exit) {}

    void operator()(sycl::handler& h) const
    {
        auto downstream = downstream_buf.template get_access<sycl::access::mode::write>(h);
        auto exit = exit_buf.template get_access<sycl::access::mode::write>(h);

        h.single_task<DrainNTTBKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t i = 0; i < NUM_BLOCKS; ++i) {
                downstream[i] = PipeSet<P>::NTTBToPolyAddPipe::read();
                exit[i] = PipeSet<P>::NTTBToExitPipe::read();
            }
        });
    }
};

} // namespace sycl_ckks::harness

int main()
{
    sycl_ckks::harness::host_debug("ntt-b: creating simple host test vectors");
    constexpr int P = 0;
    const uint8_t modulus_selector = get_modulus_selector(sycl_ckks::harness::default_modulus(P));

    // Input is the requested simple pattern: one u32x4 block per index i,
    // with each lane derived directly from i.
    std::vector<u32x4> input(NUM_BLOCKS);
    std::vector<u32x4> downstream_output(NUM_BLOCKS);
    std::vector<u32x4> exit_output(NUM_BLOCKS);
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        input[i] = sycl_ckks::harness::make_test_u32x4(i);
    }

    {
        sycl::buffer<u32x4, 1> input_buf(input.data(), sycl::range<1>(NUM_BLOCKS));
        sycl::buffer<u32x4, 1> downstream_buf(downstream_output.data(), sycl::range<1>(NUM_BLOCKS));
        sycl::buffer<u32x4, 1> exit_buf(exit_output.data(), sycl::range<1>(NUM_BLOCKS));

        auto q = sycl_ckks::harness::make_queue();

        // NTTKernelB<P> wraps the imported RTL NTT block. The feeder writes the
        // wrapper input pipe from input_buf. The production kernel is launched
        // with save_output=true so the same RTL output is written to both its
        // normal downstream pipe and its diagnostic exit pipe. The drain copies
        // both pipes to host buffers so the harness can verify that the wrapper
        // delivered a consistent output frame.
        auto drain_event = q.submit([&](sycl::handler& h) {
            sycl_ckks::harness::DrainNTTBKernel<P> kernel(downstream_buf, exit_buf);
            kernel(h);
        });
        auto ntt_event = q.submit([&](sycl::handler& h) {
            NTTKernelB<P> kernel(modulus_selector, true);
            kernel(h);
        });
        auto feed_event = q.submit([&](sycl::handler& h) {
            sycl_ckks::harness::FeedNTTBKernel<P> kernel(input_buf);
            kernel(h);
        });

        (void)drain_event;
        (void)ntt_event;
        (void)feed_event;
        q.wait_and_throw();
    }

    sycl_ckks::harness::host_debug("ntt-b: comparing downstream and exit copies of RTL output");
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(downstream_output[i], exit_output[i]), "NTTKernelB downstream/exit output mismatch");
    }

    sycl_ckks::harness::host_debug("ntt-b: PASS");
    return 0;
}
