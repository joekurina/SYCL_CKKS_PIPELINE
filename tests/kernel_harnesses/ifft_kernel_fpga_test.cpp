#include "harness_common.h"

#include "SYCL_ifft.h"
#include "SYCL_pipes.h"

#include <array>
#include <vector>

using namespace sycl_ckks;

namespace sycl_ckks::harness {

class FeedIFFTKernelTask;

class FeedIFFTKernel {
public:
    void operator()(sycl::handler& h) const
    {
        h.single_task<FeedIFFTKernelTask>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                SharedToIFFTPipe::write(pattern_encoding(blk));
            }
        });
    }
};

class DrainIFFTKernelTask;

class DrainIFFTKernel {
private:
    mutable std::array<sycl::buffer<encoding_block, 1>, NUM_MODULI> output_bufs;

public:
    explicit DrainIFFTKernel(std::array<sycl::buffer<encoding_block, 1>, NUM_MODULI>& bufs)
        : output_bufs(bufs) {}

    void operator()(sycl::handler& h) const
    {
        auto out0 = output_bufs[0].template get_access<sycl::access::mode::write>(h);
        auto out1 = output_bufs[1].template get_access<sycl::access::mode::write>(h);
        auto out2 = output_bufs[2].template get_access<sycl::access::mode::write>(h);
        h.single_task<DrainIFFTKernelTask>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                out0[blk] = IFFTToScaleReducePipes::PipeAt<0>::read();
                out1[blk] = IFFTToScaleReducePipes::PipeAt<1>::read();
                out2[blk] = IFFTToScaleReducePipes::PipeAt<2>::read();
            }
        });
    }
};

} // namespace sycl_ckks::harness

int main()
{
    sycl_ckks::harness::host_debug("ifft: preparing host buffers");
    std::array<std::vector<encoding_block>, NUM_MODULI> output;
    for (auto& per_modulus : output) {
        per_modulus.resize(NUM_BLOCKS);
    }

    std::array<sycl::buffer<encoding_block, 1>, NUM_MODULI> output_bufs = {
        sycl::buffer<encoding_block, 1>(output[0].data(), sycl::range<1>(NUM_BLOCKS)),
        sycl::buffer<encoding_block, 1>(output[1].data(), sycl::range<1>(NUM_BLOCKS)),
        sycl::buffer<encoding_block, 1>(output[2].data(), sycl::range<1>(NUM_BLOCKS)),
    };

    sycl_ckks::harness::host_debug("ifft: creating SYCL queue");
    auto q = sycl_ckks::harness::make_queue();
    sycl_ckks::harness::host_debug("ifft: submitting drain kernel");
    auto drain_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::DrainIFFTKernel kernel(output_bufs);
        kernel(h);
    });
    sycl_ckks::harness::host_debug("ifft: submitting IFFT kernel");
    auto ifft_event = q.submit([&](sycl::handler& h) {
        IFFTKernel kernel;
        kernel(h);
    });
    sycl_ckks::harness::host_debug("ifft: submitting feeder kernel");
    auto feed_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::FeedIFFTKernel kernel;
        kernel(h);
    });

    (void)drain_event;
    (void)ifft_event;
    (void)feed_event;
    sycl_ckks::harness::host_debug("ifft: waiting for submitted kernels");
    q.wait_and_throw();
    sycl_ckks::harness::host_debug("ifft: kernels completed; verifying outputs");

    for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        sycl_ckks::harness::require(sycl_ckks::harness::equal_encoding(output[0][blk], output[1][blk]), "IFFTKernel fanout 0/1 mismatch");
        sycl_ckks::harness::require(sycl_ckks::harness::equal_encoding(output[1][blk], output[2][blk]), "IFFTKernel fanout 1/2 mismatch");
    }
    sycl_ckks::harness::host_debug("ifft: PASS");
    return 0;
}
