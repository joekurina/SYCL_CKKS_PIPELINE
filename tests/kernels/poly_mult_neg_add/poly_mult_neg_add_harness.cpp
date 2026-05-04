#include "SYCL_poly_mult_neg_add.h"
#include "kernel_test_common.h"
#include "kernel_test_queue.h"
#include <iostream>

#ifndef CKKS_TEST_P
#define CKKS_TEST_P 0
#endif

using namespace sycl;
using namespace sycl_ckks;

class FeedPolyNTTATask;
class FeedPolyC1Task;
class FeedPolyNTTBTask;
class DrainPolyTask;

int main()
{
    constexpr int P = CKKS_TEST_P;
    const uint32_t mod = sycl_ckks::test::test_modulus(P);
    uint32_t cr[2];
    sycl_ckks::test::test_const_ratio(P, cr);

    std::vector<u32x4> ntt_s(NUM_BLOCKS);
    std::vector<u32x4> c1(NUM_BLOCKS);
    std::vector<u32x4> ntt_pte(NUM_BLOCKS);
    std::vector<u32x4> expected(NUM_BLOCKS);
    std::vector<u32x4> actual(NUM_BLOCKS);

    for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        ntt_s[blk] = sycl_ckks::test::make_u32x4(1000u + static_cast<uint32_t>(blk * LANES));
        c1[blk] = sycl_ckks::test::make_u32x4(2000u + static_cast<uint32_t>(blk * LANES));
        ntt_pte[blk] = sycl_ckks::test::make_u32x4(3000u + static_cast<uint32_t>(blk * LANES));
        for (size_t lane = 0; lane < LANES; ++lane) {
            uint64_t prod = static_cast<uint64_t>(sycl_ckks::test::lane_at(ntt_s[blk], lane)) *
                            static_cast<uint64_t>(sycl_ckks::test::lane_at(c1[blk], lane));
            uint32_t red = barrett_reduce_u64_core(prod, mod, cr[0], cr[1]);
            uint32_t neg = mod_neg(red, mod);
            uint32_t out = mod_add(neg, sycl_ckks::test::lane_at(ntt_pte[blk], lane), mod);
            sycl_ckks::test::set_lane(expected[blk], lane, out);
        }
    }

    buffer<u32x4, 1> ntt_s_buf(ntt_s.data(), range<1>(NUM_BLOCKS));
    buffer<u32x4, 1> c1_buf(c1.data(), range<1>(NUM_BLOCKS));
    buffer<u32x4, 1> ntt_pte_buf(ntt_pte.data(), range<1>(NUM_BLOCKS));
    buffer<u32x4, 1> out_buf(actual.data(), range<1>(NUM_BLOCKS));

    auto q = sycl_ckks::test::make_queue();
    std::vector<event> events;

    events.push_back(q.submit([&](handler& h) {
        auto out = out_buf.template get_access<access::mode::write>(h);
        h.single_task<DrainPolyTask>([=]() {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) out[blk] = PipeSet<P>::PolyAddToExitPipe::read();
        });
    }));
    events.push_back(q.submit([&](handler& h) {
        auto in = ntt_s_buf.template get_access<access::mode::read>(h);
        h.single_task<FeedPolyNTTATask>([=]() {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) PipeSet<P>::NTTAToPolyMultNegPipe::write(in[blk]);
        });
    }));
    events.push_back(q.submit([&](handler& h) {
        auto in = c1_buf.template get_access<access::mode::read>(h);
        h.single_task<FeedPolyC1Task>([=]() {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) PipeSet<P>::EntryToPolyMultNegPipe::write(in[blk]);
        });
    }));
    events.push_back(q.submit([&](handler& h) {
        auto in = ntt_pte_buf.template get_access<access::mode::read>(h);
        h.single_task<FeedPolyNTTBTask>([=]() {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) PipeSet<P>::NTTBToPolyAddPipe::write(in[blk]);
        });
    }));
    events.push_back(q.submit([&](handler& h) {
        PolyMultNegAddKernel<P> kernel(mod, cr);
        kernel(h);
    }));

    for (auto& e : events) e.wait();
    if (!sycl_ckks::test::expect_u32x4_vector("poly_mult_neg_add", expected, actual)) return 1;
    std::cout << "PASSED ckks_test_poly_mult_neg_add_p" << P << "\n";
    return 0;
}
