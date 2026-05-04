#include "SYCL_ifft.h"
#include "kernel_test_common.h"
#include "kernel_test_queue.h"
#include "kernel_test_reference.h"
#include <array>
#include <iostream>

using namespace sycl;
using namespace sycl_ckks;

class FeedIFFTTask;
template <int P> class DrainIFFTTask;

int main()
{
    std::vector<encoding_block> input(NUM_BLOCKS);
    for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        input[blk] = sycl_ckks::test::make_encoding_block(static_cast<double>(blk * LANES));
    }

#ifdef FPGA_EMULATOR
    std::vector<encoding_block> expected = sycl_ckks::test::reference_ifft(input);
#else
    std::vector<encoding_block> expected(NUM_BLOCKS);
#endif

    std::array<std::vector<encoding_block>, NUM_MODULI> actual;
    for (size_t p = 0; p < NUM_MODULI; ++p) actual[p].resize(NUM_BLOCKS);

    buffer<encoding_block, 1> in_buf(input.data(), range<1>(NUM_BLOCKS));
    std::array<buffer<encoding_block, 1>, NUM_MODULI> out_bufs = {
        buffer<encoding_block, 1>(actual[0].data(), range<1>(NUM_BLOCKS)),
        buffer<encoding_block, 1>(actual[1].data(), range<1>(NUM_BLOCKS)),
        buffer<encoding_block, 1>(actual[2].data(), range<1>(NUM_BLOCKS))
    };

    auto q = sycl_ckks::test::make_queue();
    std::vector<event> events;

    auto submit_drain = [&](auto p_c) {
        constexpr int P = decltype(p_c)::value;
        events.push_back(q.submit([&](handler& h) {
            auto out = out_bufs[P].template get_access<access::mode::write>(h);
            h.single_task<DrainIFFTTask<P>>([=]() {
                for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) out[blk] = IFFTToScaleReducePipes::PipeAt<P>::read();
            });
        }));
    };
    submit_drain(std::integral_constant<int, 0>{});
    submit_drain(std::integral_constant<int, 1>{});
    submit_drain(std::integral_constant<int, 2>{});

    events.push_back(q.submit([&](handler& h) {
        auto in = in_buf.template get_access<access::mode::read>(h);
        h.single_task<FeedIFFTTask>([=]() {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) SharedToIFFTPipe::write(in[blk]);
        });
    }));
    events.push_back(q.submit([&](handler& h) {
        IFFTKernel kernel;
        kernel(h);
    }));

    for (auto& e : events) e.wait();

    bool ok = true;
    ok &= sycl_ckks::test::expect_encoding_vector("ifft_fanout_p1", actual[0], actual[1]);
    ok &= sycl_ckks::test::expect_encoding_vector("ifft_fanout_p2", actual[0], actual[2]);
#ifdef FPGA_EMULATOR
    ok &= sycl_ckks::test::expect_encoding_vector("ifft_reference", expected, actual[0], 0.0);
#endif

    if (!ok) return 1;
    std::cout << "PASSED ckks_test_ifft\n";
    return 0;
}
