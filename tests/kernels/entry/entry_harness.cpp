#include "SYCL_entry.h"
#include "kernel_test_common.h"
#include "kernel_test_queue.h"
#include <array>
#include <iostream>

using namespace sycl;
using namespace sycl_ckks;

class DrainSharedToIFFTTask;
template <int P> class DrainErrorTask;
template <int P> class DrainEntryToNTTATask;
template <int P> class DrainEntryToPolyTask;

int main()
{
    std::vector<PipelineInputBlock> input(NUM_BLOCKS);
    std::vector<encoding_block> expected_encoding(NUM_BLOCKS);
    std::vector<i8x4> expected_error(NUM_BLOCKS);
    std::array<std::vector<u32x4>, NUM_MODULI> expected_secret;
    std::array<std::vector<u32x4>, NUM_MODULI> expected_c1;

    for (size_t p = 0; p < NUM_MODULI; ++p) {
        expected_secret[p].resize(NUM_BLOCKS);
        expected_c1[p].resize(NUM_BLOCKS);
    }

    for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        input[blk].encoding = sycl_ckks::test::make_encoding_block(static_cast<double>(blk * LANES));
        input[blk].error = sycl_ckks::test::make_i8x4(static_cast<int>((blk % 31) - 15));
        expected_encoding[blk] = input[blk].encoding;
        expected_error[blk] = input[blk].error;
        for (size_t p = 0; p < NUM_MODULI; ++p) {
            input[blk].secret_key[p] = sycl_ckks::test::make_u32x4(100000u * static_cast<uint32_t>(p + 1) + static_cast<uint32_t>(blk * LANES));
            input[blk].c1[p] = sycl_ckks::test::make_u32x4(200000u * static_cast<uint32_t>(p + 1) + static_cast<uint32_t>(blk * LANES));
            expected_secret[p][blk] = input[blk].secret_key[p];
            expected_c1[p][blk] = input[blk].c1[p];
        }
    }

    std::vector<encoding_block> actual_encoding(NUM_BLOCKS);
    std::array<std::vector<i8x4>, NUM_MODULI> actual_error;
    std::array<std::vector<u32x4>, NUM_MODULI> actual_secret;
    std::array<std::vector<u32x4>, NUM_MODULI> actual_c1;
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        actual_error[p].resize(NUM_BLOCKS);
        actual_secret[p].resize(NUM_BLOCKS);
        actual_c1[p].resize(NUM_BLOCKS);
    }

    buffer<PipelineInputBlock, 1> input_buf(input.data(), range<1>(NUM_BLOCKS));
    buffer<encoding_block, 1> enc_buf(actual_encoding.data(), range<1>(NUM_BLOCKS));
    std::array<buffer<i8x4, 1>, NUM_MODULI> err_bufs = {
        buffer<i8x4, 1>(actual_error[0].data(), range<1>(NUM_BLOCKS)),
        buffer<i8x4, 1>(actual_error[1].data(), range<1>(NUM_BLOCKS)),
        buffer<i8x4, 1>(actual_error[2].data(), range<1>(NUM_BLOCKS))
    };
    std::array<buffer<u32x4, 1>, NUM_MODULI> secret_bufs = {
        buffer<u32x4, 1>(actual_secret[0].data(), range<1>(NUM_BLOCKS)),
        buffer<u32x4, 1>(actual_secret[1].data(), range<1>(NUM_BLOCKS)),
        buffer<u32x4, 1>(actual_secret[2].data(), range<1>(NUM_BLOCKS))
    };
    std::array<buffer<u32x4, 1>, NUM_MODULI> c1_bufs = {
        buffer<u32x4, 1>(actual_c1[0].data(), range<1>(NUM_BLOCKS)),
        buffer<u32x4, 1>(actual_c1[1].data(), range<1>(NUM_BLOCKS)),
        buffer<u32x4, 1>(actual_c1[2].data(), range<1>(NUM_BLOCKS))
    };

    auto q = sycl_ckks::test::make_queue();
    std::vector<event> events;

    events.push_back(q.submit([&](handler& h) {
        auto out = enc_buf.template get_access<access::mode::write>(h);
        h.single_task<DrainSharedToIFFTTask>([=]() {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) out[blk] = SharedToIFFTPipe::read();
        });
    }));

    auto submit_err = [&](auto p_c) {
        constexpr int P = decltype(p_c)::value;
        events.push_back(q.submit([&](handler& h) {
            auto out = err_bufs[P].template get_access<access::mode::write>(h);
            h.single_task<DrainErrorTask<P>>([=]() {
                for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) out[blk] = ErrorToScaleReducePipes::PipeAt<P>::read();
            });
        }));
    };

    auto submit_secret = [&](auto p_c) {
        constexpr int P = decltype(p_c)::value;
        events.push_back(q.submit([&](handler& h) {
            auto out = secret_bufs[P].template get_access<access::mode::write>(h);
            h.single_task<DrainEntryToNTTATask<P>>([=]() {
                for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) out[blk] = PipeSet<P>::EntryToNTTAPipe::read();
            });
        }));
    };

    auto submit_c1 = [&](auto p_c) {
        constexpr int P = decltype(p_c)::value;
        events.push_back(q.submit([&](handler& h) {
            auto out = c1_bufs[P].template get_access<access::mode::write>(h);
            h.single_task<DrainEntryToPolyTask<P>>([=]() {
                for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) out[blk] = PipeSet<P>::EntryToPolyMultNegPipe::read();
            });
        }));
    };

    submit_err(std::integral_constant<int, 0>{});
    submit_err(std::integral_constant<int, 1>{});
    submit_err(std::integral_constant<int, 2>{});
    submit_secret(std::integral_constant<int, 0>{});
    submit_secret(std::integral_constant<int, 1>{});
    submit_secret(std::integral_constant<int, 2>{});
    submit_c1(std::integral_constant<int, 0>{});
    submit_c1(std::integral_constant<int, 1>{});
    submit_c1(std::integral_constant<int, 2>{});

    events.push_back(q.submit([&](handler& h) {
        EntryKernel kernel(input_buf);
        kernel(h);
    }));

    for (auto& e : events) e.wait();

    bool ok = sycl_ckks::test::expect_encoding_vector("entry.encoding", expected_encoding, actual_encoding);
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        ok &= sycl_ckks::test::expect_i8x4_vector("entry.error", expected_error, actual_error[p]);
        ok &= sycl_ckks::test::expect_u32x4_vector("entry.secret", expected_secret[p], actual_secret[p]);
        ok &= sycl_ckks::test::expect_u32x4_vector("entry.c1", expected_c1[p], actual_c1[p]);
    }

    if (!ok) return 1;
    std::cout << "PASSED ckks_test_entry\n";
    return 0;
}
