#include "harness_common.h"

#include "SYCL_ifft.h"
#include "SYCL_pipes.h"

#include <array>
#include <vector>

using namespace sycl_ckks;

namespace sycl_ckks::harness {

class FeedIFFTKernelTask;

class FeedIFFTKernel {
private:
    mutable sycl::buffer<encoding_block, 1> input_buf;

public:
    explicit FeedIFFTKernel(sycl::buffer<encoding_block, 1>& input)
        : input_buf(input) {}

    void operator()(sycl::handler& h) const
    {
        auto input = input_buf.template get_access<sycl::access::mode::read>(h);

        h.single_task<FeedIFFTKernelTask>([=]() [[intel::kernel_args_restrict]] {
            for (size_t i = 0; i < NUM_BLOCKS; ++i) {
                SharedToIFFTPipe::write(input[i]);
            }
        });
    }
};

class DrainIFFTKernelTask;

class DrainIFFTKernel {
private:
    mutable std::array<sycl::buffer<encoding_block, 1>, NUM_PHYSICAL_PIPELINES> output_bufs;

public:
    explicit DrainIFFTKernel(std::array<sycl::buffer<encoding_block, 1>, NUM_PHYSICAL_PIPELINES>& output)
        : output_bufs(output) {}

    void operator()(sycl::handler& h) const
    {
        auto out0 = output_bufs[0].template get_access<sycl::access::mode::write>(h);
        auto out1 = output_bufs[1].template get_access<sycl::access::mode::write>(h);
        auto out2 = output_bufs[2].template get_access<sycl::access::mode::write>(h);
        auto out3 = output_bufs[3].template get_access<sycl::access::mode::write>(h);
        auto out4 = output_bufs[4].template get_access<sycl::access::mode::write>(h);
        auto out5 = output_bufs[5].template get_access<sycl::access::mode::write>(h);

        h.single_task<DrainIFFTKernelTask>([=]() [[intel::kernel_args_restrict]] {
            for (size_t i = 0; i < NUM_BLOCKS; ++i) {
                out0[i] = IFFTToScaleReducePipes::PipeAt<0>::read();
                out1[i] = IFFTToScaleReducePipes::PipeAt<1>::read();
                out2[i] = IFFTToScaleReducePipes::PipeAt<2>::read();
                out3[i] = IFFTToScaleReducePipes::PipeAt<3>::read();
                out4[i] = IFFTToScaleReducePipes::PipeAt<4>::read();
                out5[i] = IFFTToScaleReducePipes::PipeAt<5>::read();
            }
        });
    }
};

} // namespace sycl_ckks::harness

int main()
{
    sycl_ckks::harness::host_debug("ifft: creating simple host test vectors");

    std::vector<encoding_block> input(NUM_BLOCKS);
    std::array<std::vector<encoding_block>, NUM_PHYSICAL_PIPELINES> output;
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        input[i] = sycl_ckks::harness::make_test_encoding(i);
    }
    for (auto& per_modulus : output) {
        per_modulus.resize(NUM_BLOCKS);
    }

    {
        sycl::buffer<encoding_block, 1> input_buf(input.data(), sycl::range<1>(NUM_BLOCKS));
        std::array<sycl::buffer<encoding_block, 1>, NUM_PHYSICAL_PIPELINES> output_bufs = {
            sycl::buffer<encoding_block, 1>(output[0].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<encoding_block, 1>(output[1].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<encoding_block, 1>(output[2].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<encoding_block, 1>(output[3].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<encoding_block, 1>(output[4].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<encoding_block, 1>(output[5].data(), sycl::range<1>(NUM_BLOCKS)),
        };

        auto q = sycl_ckks::harness::make_queue();

        // IFFTKernel consumes SharedToIFFTPipe and fans the RTL output to one
        // ScaleReduce pipe per modulus. The harness verifies that all fanout
        // pipes receive the same RTL output frame.
        auto drain_event = q.submit([&](sycl::handler& h) {
            sycl_ckks::harness::DrainIFFTKernel kernel(output_bufs);
            kernel(h);
        });
        auto fanout_event = q.submit([&](sycl::handler& h) {
            IFFTFanoutKernel<NUM_BLOCKS> kernel;
            kernel(h);
        });
        auto ifft_event = q.submit([&](sycl::handler& h) {
            IFFTKernel kernel;
            kernel(h);
        });
        auto feed_event = q.submit([&](sycl::handler& h) {
            sycl_ckks::harness::FeedIFFTKernel kernel(input_buf);
            kernel(h);
        });

        (void)ifft_event;
        feed_event.wait_and_throw();
        fanout_event.wait_and_throw();
        drain_event.wait_and_throw();
    }

    sycl_ckks::harness::host_debug("ifft: comparing fanout pipes with expected matching frames");
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        sycl_ckks::harness::require(sycl_ckks::harness::equal_encoding(output[1][i], output[0][i]), "IFFTKernel fanout 0/1 mismatch");
        sycl_ckks::harness::require(sycl_ckks::harness::equal_encoding(output[2][i], output[0][i]), "IFFTKernel fanout 0/2 mismatch");
        sycl_ckks::harness::require(sycl_ckks::harness::equal_encoding(output[3][i], output[0][i]), "IFFTKernel fanout 0/3 mismatch");
        sycl_ckks::harness::require(sycl_ckks::harness::equal_encoding(output[4][i], output[0][i]), "IFFTKernel fanout 0/4 mismatch");
        sycl_ckks::harness::require(sycl_ckks::harness::equal_encoding(output[5][i], output[0][i]), "IFFTKernel fanout 0/5 mismatch");
    }

    sycl_ckks::harness::host_debug("ifft: PASS");
    return 0;
}
