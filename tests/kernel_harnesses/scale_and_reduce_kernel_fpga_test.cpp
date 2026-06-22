#include "harness_common.h"

#include "SYCL_pipes.h"
#include "SYCL_scale_and_reduce.h"

#include <cmath>
#include <vector>

using namespace sycl_ckks;

namespace sycl_ckks::harness {

template <int P>
class FeedScaleAndReduceKernelTask;

template <int P>
class FeedScaleAndReduceKernel {
private:
    mutable sycl::buffer<encoding_block, 1> encoding_buf;
    mutable sycl::buffer<i8x4, 1> error_buf;

public:
    FeedScaleAndReduceKernel(sycl::buffer<encoding_block, 1>& encoding,
                             sycl::buffer<i8x4, 1>& error)
        : encoding_buf(encoding), error_buf(error) {}

    void operator()(sycl::handler& h) const
    {
        auto encoding = encoding_buf.template get_access<sycl::access::mode::read>(h);
        auto error = error_buf.template get_access<sycl::access::mode::read>(h);

        h.single_task<FeedScaleAndReduceKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t i = 0; i < NUM_BLOCKS; ++i) {
                IFFTToScaleReducePipes::PipeAt<P>::write(encoding[i]);
                ErrorToScaleReducePipes::PipeAt<P>::write(error[i]);
            }
        });
    }
};

template <int P>
class DrainScaleAndReduceKernelTask;

template <int P>
class DrainScaleAndReduceKernel {
private:
    mutable sycl::buffer<u32x4, 1> output_buf;

public:
    explicit DrainScaleAndReduceKernel(sycl::buffer<u32x4, 1>& output)
        : output_buf(output) {}

    void operator()(sycl::handler& h) const
    {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);

        h.single_task<DrainScaleAndReduceKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t i = 0; i < NUM_BLOCKS; ++i) {
                output[i] = PipeSet<P>::ScaleReduceToNTTBPipe::read();
            }
        });
    }
};

inline uint32_t expected_scale_reduce_lane(double real, int8_t noise, double scale, uint32_t mod, const uint32_t ratio[2])
{
    const double n_inv = scale / static_cast<double>(POLY_N);
    const auto rounded = static_cast<int64_t>(std::llround(real * n_inv));
    return barrett_reduce_64_core(rounded + noise, mod, ratio[0], ratio[1], false);
}

inline u32x4 expected_scale_reduce_block(const encoding_block& encoding, const i8x4& error,
                                         double scale, uint32_t mod, const uint32_t ratio[2])
{
    u32x4 expected;
    expected.element0 = expected_scale_reduce_lane(encoding.element0.real(), error.element0, scale, mod, ratio);
    expected.element1 = expected_scale_reduce_lane(encoding.element1.real(), error.element1, scale, mod, ratio);
    expected.element2 = expected_scale_reduce_lane(encoding.element2.real(), error.element2, scale, mod, ratio);
    expected.element3 = expected_scale_reduce_lane(encoding.element3.real(), error.element3, scale, mod, ratio);
    return expected;
}

} // namespace sycl_ckks::harness

int main()
{
    sycl_ckks::harness::host_debug("scale-reduce: creating simple host test vectors");
    constexpr int P = 0;
    constexpr double scale = 1048576.0;
    const uint32_t mod = sycl_ckks::harness::default_modulus(P);
    uint32_t ratio[2];
    sycl_ckks::harness::default_const_ratio(P, ratio);

    // Conceptually: for (i = 0; i < NUM_BLOCKS; ++i) test_vector[i] = i.
    // The scalar index is adapted to the aggregate encoding/error lane types.
    std::vector<encoding_block> encoding_input(NUM_BLOCKS);
    std::vector<i8x4> error_input(NUM_BLOCKS);
    std::vector<u32x4> output(NUM_BLOCKS);
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        encoding_input[i] = sycl_ckks::harness::make_test_encoding(i);
        error_input[i] = sycl_ckks::harness::make_test_i8x4(i);
    }

    {
        sycl::buffer<encoding_block, 1> encoding_buf(encoding_input.data(), sycl::range<1>(NUM_BLOCKS));
        sycl::buffer<i8x4, 1> error_buf(error_input.data(), sycl::range<1>(NUM_BLOCKS));
        sycl::buffer<u32x4, 1> output_buf(output.data(), sycl::range<1>(NUM_BLOCKS));

        auto q = sycl_ckks::harness::make_queue();

        // Drain first, run the production kernel, then feed its two input pipes.
        // The feeder connects host buffers to the exact pipes consumed by
        // ScaleAndReduceKernel<P>.
        auto drain_event = q.submit([&](sycl::handler& h) {
            sycl_ckks::harness::DrainScaleAndReduceKernel<P> kernel(output_buf);
            kernel(h);
        });
        auto scale_event = q.submit([&](sycl::handler& h) {
            ScaleAndReduceKernel<P> kernel(scale, mod, ratio);
            kernel(h);
        });
        auto feed_event = q.submit([&](sycl::handler& h) {
            sycl_ckks::harness::FeedScaleAndReduceKernel<P> kernel(encoding_buf, error_buf);
            kernel(h);
        });

        (void)drain_event;
        (void)scale_event;
        (void)feed_event;
        q.wait_and_throw();
    }

    sycl_ckks::harness::host_debug("scale-reduce: comparing output with host arithmetic oracle");
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        const u32x4 expected = sycl_ckks::harness::expected_scale_reduce_block(encoding_input[i], error_input[i], scale, mod, ratio);
        sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(output[i], expected), "ScaleAndReduceKernel output mismatch");
    }

    sycl_ckks::harness::host_debug("scale-reduce: PASS");
    return 0;
}
