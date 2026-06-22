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
private:
    mutable sycl::buffer<u32x4, 1> ntt_a_buf;
    mutable sycl::buffer<u32x4, 1> c1_buf;
    mutable sycl::buffer<u32x4, 1> ntt_b_buf;

public:
    FeedPolyMultNegAddKernel(sycl::buffer<u32x4, 1>& ntt_a,
                             sycl::buffer<u32x4, 1>& c1,
                             sycl::buffer<u32x4, 1>& ntt_b)
        : ntt_a_buf(ntt_a), c1_buf(c1), ntt_b_buf(ntt_b) {}

    void operator()(sycl::handler& h) const
    {
        auto ntt_a = ntt_a_buf.template get_access<sycl::access::mode::read>(h);
        auto c1 = c1_buf.template get_access<sycl::access::mode::read>(h);
        auto ntt_b = ntt_b_buf.template get_access<sycl::access::mode::read>(h);

        h.single_task<FeedPolyMultNegAddKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t i = 0; i < NUM_BLOCKS; ++i) {
                PipeSet<P>::NTTAToPolyMultNegPipe::write(ntt_a[i]);
                PipeSet<P>::EntryToPolyMultNegPipe::write(c1[i]);
                PipeSet<P>::NTTBToPolyAddPipe::write(ntt_b[i]);
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
    explicit DrainPolyMultNegAddKernel(sycl::buffer<u32x4, 1>& output)
        : output_buf(output) {}

    void operator()(sycl::handler& h) const
    {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);

        h.single_task<DrainPolyMultNegAddKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t i = 0; i < NUM_BLOCKS; ++i) {
                output[i] = PipeSet<P>::PolyAddToExitPipe::read();
            }
        });
    }
};

inline uint32_t expected_poly_lane(uint32_t ntt_s, uint32_t c1, uint32_t ntt_pte,
                                   uint32_t mod, const uint32_t ratio[2])
{
    const uint32_t prod = barrett_reduce_u64_core(static_cast<uint64_t>(ntt_s) * c1, mod, ratio[0], ratio[1]);
    return mod_add(mod_neg(prod, mod), ntt_pte, mod);
}

inline u32x4 expected_poly_block(const u32x4& ntt_a, const u32x4& c1, const u32x4& ntt_b,
                                 uint32_t mod, const uint32_t ratio[2])
{
    u32x4 expected;
    expected.element0 = expected_poly_lane(ntt_a.element0, c1.element0, ntt_b.element0, mod, ratio);
    expected.element1 = expected_poly_lane(ntt_a.element1, c1.element1, ntt_b.element1, mod, ratio);
    expected.element2 = expected_poly_lane(ntt_a.element2, c1.element2, ntt_b.element2, mod, ratio);
    expected.element3 = expected_poly_lane(ntt_a.element3, c1.element3, ntt_b.element3, mod, ratio);
    return expected;
}

} // namespace sycl_ckks::harness

int main()
{
    sycl_ckks::harness::host_debug("poly-mult-neg-add: creating simple host test vectors");
    constexpr int P = 0;
    const uint32_t mod = sycl_ckks::harness::default_modulus(P);
    uint32_t ratio[2];
    sycl_ckks::harness::default_const_ratio(P, ratio);

    // Three input vectors feed the three production input pipes. Each vector is
    // derived from index i; offsets just keep the operands distinct.
    std::vector<u32x4> ntt_a_input(NUM_BLOCKS);
    std::vector<u32x4> c1_input(NUM_BLOCKS);
    std::vector<u32x4> ntt_b_input(NUM_BLOCKS);
    std::vector<u32x4> output(NUM_BLOCKS);
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        ntt_a_input[i] = sycl_ckks::harness::make_test_u32x4(i, 10u);
        c1_input[i] = sycl_ckks::harness::make_test_u32x4(i, 20u);
        ntt_b_input[i] = sycl_ckks::harness::make_test_u32x4(i, 30u);
    }

    {
        sycl::buffer<u32x4, 1> ntt_a_buf(ntt_a_input.data(), sycl::range<1>(NUM_BLOCKS));
        sycl::buffer<u32x4, 1> c1_buf(c1_input.data(), sycl::range<1>(NUM_BLOCKS));
        sycl::buffer<u32x4, 1> ntt_b_buf(ntt_b_input.data(), sycl::range<1>(NUM_BLOCKS));
        sycl::buffer<u32x4, 1> output_buf(output.data(), sycl::range<1>(NUM_BLOCKS));

        auto q = sycl_ckks::harness::make_queue();

        auto drain_event = q.submit([&](sycl::handler& h) {
            sycl_ckks::harness::DrainPolyMultNegAddKernel<P> kernel(output_buf);
            kernel(h);
        });
        auto poly_event = q.submit([&](sycl::handler& h) {
            PolyMultNegAddKernel<P> kernel(mod, ratio);
            kernel(h);
        });
        auto feed_event = q.submit([&](sycl::handler& h) {
            sycl_ckks::harness::FeedPolyMultNegAddKernel<P> kernel(ntt_a_buf, c1_buf, ntt_b_buf);
            kernel(h);
        });

        (void)drain_event;
        (void)poly_event;
        (void)feed_event;
        q.wait_and_throw();
    }

    sycl_ckks::harness::host_debug("poly-mult-neg-add: comparing output with host arithmetic oracle");
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        const u32x4 expected = sycl_ckks::harness::expected_poly_block(ntt_a_input[i], c1_input[i], ntt_b_input[i], mod, ratio);
        sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(output[i], expected), "PolyMultNegAddKernel output mismatch");
    }

    sycl_ckks::harness::host_debug("poly-mult-neg-add: PASS");
    return 0;
}
