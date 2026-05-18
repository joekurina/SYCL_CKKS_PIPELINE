#include "harness_common.h"

#include "SYCL_pipes.h"
#include "SYCL_poly_mult_neg_add.h"

#include <vector>

using namespace sycl_ckks;

namespace sycl_ckks::harness {

template <int P>
class FeedPolyMultNegAddKernelTask;

template <int P>
class FeedPolyMultNegAddKernel {
public:
    void operator()(sycl::handler& h) const
    {
        h.single_task<FeedPolyMultNegAddKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                PipeSet<P>::NTTAToPolyMultNegPipe::write(pattern_u32x4(blk, 100u));
                PipeSet<P>::EntryToPolyMultNegPipe::write(pattern_u32x4(blk, 200u));
                PipeSet<P>::NTTBToPolyAddPipe::write(pattern_u32x4(blk, 300u));
            }
        });
    }
};

template <int P>
class DrainPolyMultNegAddKernelTask;

template <int P>
class DrainPolyMultNegAddKernel {
private:
    mutable sycl::buffer<u32x4, 1> output_buf;

public:
    explicit DrainPolyMultNegAddKernel(sycl::buffer<u32x4, 1>& buf) : output_buf(buf) {}

    void operator()(sycl::handler& h) const
    {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);
        h.single_task<DrainPolyMultNegAddKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                output[blk] = PipeSet<P>::PolyAddToExitPipe::read();
            }
        });
    }
};

inline uint32_t expected_poly_lane(size_t block, size_t lane, uint32_t mod, const uint32_t ratio[2])
{
    const uint32_t ntt_s = pattern_u32(block, lane, 100u);
    const uint32_t c1 = pattern_u32(block, lane, 200u);
    const uint32_t ntt_pte = pattern_u32(block, lane, 300u);
    const uint32_t prod = barrett_reduce_u64_core(static_cast<uint64_t>(ntt_s) * c1, mod, ratio[0], ratio[1]);
    return mod_add(mod_neg(prod, mod), ntt_pte, mod);
}

} // namespace sycl_ckks::harness

int main()
{
    sycl_ckks::harness::host_debug("poly-mult-neg-add: preparing host buffers");
    constexpr int P = 0;
    const uint32_t mod = sycl_ckks::harness::default_modulus(P);
    uint32_t ratio[2];
    sycl_ckks::harness::default_const_ratio(P, ratio);

    std::vector<u32x4> output(NUM_BLOCKS);
    sycl::buffer<u32x4, 1> output_buf(output.data(), sycl::range<1>(NUM_BLOCKS));

    sycl_ckks::harness::host_debug("poly-mult-neg-add: creating SYCL queue");
    auto q = sycl_ckks::harness::make_queue();
    sycl_ckks::harness::host_debug("poly-mult-neg-add: submitting drain kernel");
    auto drain_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::DrainPolyMultNegAddKernel<P> kernel(output_buf);
        kernel(h);
    });
    sycl_ckks::harness::host_debug("poly-mult-neg-add: submitting poly kernel");
    auto poly_event = q.submit([&](sycl::handler& h) {
        PolyMultNegAddKernel<P> kernel(mod, ratio);
        kernel(h);
    });
    sycl_ckks::harness::host_debug("poly-mult-neg-add: submitting feeder kernel");
    auto feed_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::FeedPolyMultNegAddKernel<P> kernel;
        kernel(h);
    });

    (void)drain_event;
    (void)poly_event;
    (void)feed_event;
    sycl_ckks::harness::host_debug("poly-mult-neg-add: waiting for submitted kernels");
    q.wait_and_throw();
    sycl_ckks::harness::host_debug("poly-mult-neg-add: kernels completed; verifying outputs");

    for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        u32x4 expected;
        expected.element0 = sycl_ckks::harness::expected_poly_lane(blk, 0, mod, ratio);
        expected.element1 = sycl_ckks::harness::expected_poly_lane(blk, 1, mod, ratio);
        expected.element2 = sycl_ckks::harness::expected_poly_lane(blk, 2, mod, ratio);
        expected.element3 = sycl_ckks::harness::expected_poly_lane(blk, 3, mod, ratio);
        sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(output[blk], expected), "PolyMultNegAddKernel output mismatch");
    }
    sycl_ckks::harness::host_debug("poly-mult-neg-add: PASS");
    return 0;
}
