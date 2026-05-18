#include "harness_common.h"

#include "SYCL_entry.h"
#include "SYCL_pipes.h"

#include <array>
#include <vector>

using namespace sycl_ckks;

namespace sycl_ckks::harness {

class DrainEntryKernelTask;

class DrainEntryKernel {
private:
    mutable sycl::buffer<encoding_block, 1> encoding_out;
    mutable std::array<sycl::buffer<i8x4, 1>, NUM_MODULI> error_out;
    mutable std::array<sycl::buffer<u32x4, 1>, NUM_MODULI> secret_out;
    mutable std::array<sycl::buffer<u32x4, 1>, NUM_MODULI> c1_out;

public:
    DrainEntryKernel(
        sycl::buffer<encoding_block, 1>& enc,
        std::array<sycl::buffer<i8x4, 1>, NUM_MODULI>& err,
        std::array<sycl::buffer<u32x4, 1>, NUM_MODULI>& sk,
        std::array<sycl::buffer<u32x4, 1>, NUM_MODULI>& c1)
        : encoding_out(enc), error_out(err), secret_out(sk), c1_out(c1) {}

    void operator()(sycl::handler& h) const
    {
        auto enc = encoding_out.template get_access<sycl::access::mode::write>(h);
        auto err0 = error_out[0].template get_access<sycl::access::mode::write>(h);
        auto err1 = error_out[1].template get_access<sycl::access::mode::write>(h);
        auto err2 = error_out[2].template get_access<sycl::access::mode::write>(h);
        auto sk0 = secret_out[0].template get_access<sycl::access::mode::write>(h);
        auto sk1 = secret_out[1].template get_access<sycl::access::mode::write>(h);
        auto sk2 = secret_out[2].template get_access<sycl::access::mode::write>(h);
        auto c10 = c1_out[0].template get_access<sycl::access::mode::write>(h);
        auto c11 = c1_out[1].template get_access<sycl::access::mode::write>(h);
        auto c12 = c1_out[2].template get_access<sycl::access::mode::write>(h);

        h.single_task<DrainEntryKernelTask>([=]() [[intel::kernel_args_restrict]] {
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                enc[blk] = SharedToIFFTPipe::read();
                err0[blk] = ErrorToScaleReducePipes::PipeAt<0>::read();
                err1[blk] = ErrorToScaleReducePipes::PipeAt<1>::read();
                err2[blk] = ErrorToScaleReducePipes::PipeAt<2>::read();
                sk0[blk] = PipeSet<0>::EntryToNTTAPipe::read();
                sk1[blk] = PipeSet<1>::EntryToNTTAPipe::read();
                sk2[blk] = PipeSet<2>::EntryToNTTAPipe::read();
                c10[blk] = PipeSet<0>::EntryToPolyMultNegPipe::read();
                c11[blk] = PipeSet<1>::EntryToPolyMultNegPipe::read();
                c12[blk] = PipeSet<2>::EntryToPolyMultNegPipe::read();
            }
        });
    }
};

} // namespace sycl_ckks::harness

int main()
{
    sycl_ckks::harness::host_debug("entry: preparing host buffers");
    std::vector<PipelineInputBlock> input(NUM_BLOCKS);
    std::vector<encoding_block> encoding_out(NUM_BLOCKS);
    std::array<std::vector<i8x4>, NUM_MODULI> error_out;
    std::array<std::vector<u32x4>, NUM_MODULI> secret_out;
    std::array<std::vector<u32x4>, NUM_MODULI> c1_out;

    for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        input[blk] = sycl_ckks::harness::pattern_input_block(blk);
    }
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        error_out[p].resize(NUM_BLOCKS);
        secret_out[p].resize(NUM_BLOCKS);
        c1_out[p].resize(NUM_BLOCKS);
    }

    sycl::buffer<PipelineInputBlock, 1> input_buf(input.data(), sycl::range<1>(NUM_BLOCKS));
    sycl::buffer<encoding_block, 1> encoding_buf(encoding_out.data(), sycl::range<1>(NUM_BLOCKS));
    std::array<sycl::buffer<i8x4, 1>, NUM_MODULI> error_bufs = {
        sycl::buffer<i8x4, 1>(error_out[0].data(), sycl::range<1>(NUM_BLOCKS)),
        sycl::buffer<i8x4, 1>(error_out[1].data(), sycl::range<1>(NUM_BLOCKS)),
        sycl::buffer<i8x4, 1>(error_out[2].data(), sycl::range<1>(NUM_BLOCKS)),
    };
    std::array<sycl::buffer<u32x4, 1>, NUM_MODULI> secret_bufs = {
        sycl::buffer<u32x4, 1>(secret_out[0].data(), sycl::range<1>(NUM_BLOCKS)),
        sycl::buffer<u32x4, 1>(secret_out[1].data(), sycl::range<1>(NUM_BLOCKS)),
        sycl::buffer<u32x4, 1>(secret_out[2].data(), sycl::range<1>(NUM_BLOCKS)),
    };
    std::array<sycl::buffer<u32x4, 1>, NUM_MODULI> c1_bufs = {
        sycl::buffer<u32x4, 1>(c1_out[0].data(), sycl::range<1>(NUM_BLOCKS)),
        sycl::buffer<u32x4, 1>(c1_out[1].data(), sycl::range<1>(NUM_BLOCKS)),
        sycl::buffer<u32x4, 1>(c1_out[2].data(), sycl::range<1>(NUM_BLOCKS)),
    };

    sycl_ckks::harness::host_debug("entry: creating SYCL queue");
    auto q = sycl_ckks::harness::make_queue();
    sycl_ckks::harness::host_debug("entry: submitting drain kernel");
    auto drain_event = q.submit([&](sycl::handler& h) {
        sycl_ckks::harness::DrainEntryKernel kernel(encoding_buf, error_bufs, secret_bufs, c1_bufs);
        kernel(h);
    });
    sycl_ckks::harness::host_debug("entry: submitting entry kernel");
    auto entry_event = q.submit([&](sycl::handler& h) {
        EntryKernel kernel(input_buf);
        kernel(h);
    });
    (void)drain_event;
    (void)entry_event;
    sycl_ckks::harness::host_debug("entry: waiting for submitted kernels");
    q.wait_and_throw();
    sycl_ckks::harness::host_debug("entry: kernels completed; verifying outputs");

    for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        sycl_ckks::harness::require(sycl_ckks::harness::equal_encoding(encoding_out[blk], input[blk].encoding), "EntryKernel encoding fanout mismatch");
        for (size_t p = 0; p < NUM_MODULI; ++p) {
            sycl_ckks::harness::require(sycl_ckks::harness::equal_i8x4(error_out[p][blk], input[blk].error), "EntryKernel error fanout mismatch");
            sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(secret_out[p][blk], input[blk].secret_key[p]), "EntryKernel secret-key fanout mismatch");
            sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(c1_out[p][blk], input[blk].c1[p]), "EntryKernel c1 fanout mismatch");
        }
    }
    sycl_ckks::harness::host_debug("entry: PASS");
    return 0;
}
