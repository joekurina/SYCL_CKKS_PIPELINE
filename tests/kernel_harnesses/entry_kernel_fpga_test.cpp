#include "harness_common.h"

#include "SYCL_entry.h"
#include "SYCL_pipes.h"

#include <array>
#include <vector>

using namespace sycl_ckks;

namespace sycl_ckks::harness {

class DrainEntryKernelTask;

// EntryKernel is itself the feeder from an input SYCL buffer into the pipeline.
// This drain kernel reads every pipe that EntryKernel should populate and copies
// those pipe values into ordinary output buffers for host-side comparison.
class DrainEntryKernel {
private:
    mutable sycl::buffer<encoding_block, 1> encoding_out;
    mutable std::array<sycl::buffer<i8x4, 1>, NUM_PHYSICAL_PIPELINES> error_out;
    mutable std::array<sycl::buffer<u32x4, 1>, NUM_PHYSICAL_PIPELINES> secret_out;
    mutable std::array<sycl::buffer<u32x4, 1>, NUM_PHYSICAL_PIPELINES> c1_out;

public:
    DrainEntryKernel(
        sycl::buffer<encoding_block, 1>& enc,
        std::array<sycl::buffer<i8x4, 1>, NUM_PHYSICAL_PIPELINES>& err,
        std::array<sycl::buffer<u32x4, 1>, NUM_PHYSICAL_PIPELINES>& sk,
        std::array<sycl::buffer<u32x4, 1>, NUM_PHYSICAL_PIPELINES>& c1)
        : encoding_out(enc), error_out(err), secret_out(sk), c1_out(c1) {}

    void operator()(sycl::handler& h) const
    {
        auto enc = encoding_out.template get_access<sycl::access::mode::write>(h);
        auto err0 = error_out[0].template get_access<sycl::access::mode::write>(h);
        auto err1 = error_out[1].template get_access<sycl::access::mode::write>(h);
        auto err2 = error_out[2].template get_access<sycl::access::mode::write>(h);
        auto err3 = error_out[3].template get_access<sycl::access::mode::write>(h);
        auto err4 = error_out[4].template get_access<sycl::access::mode::write>(h);
        auto err5 = error_out[5].template get_access<sycl::access::mode::write>(h);
        auto sk0 = secret_out[0].template get_access<sycl::access::mode::write>(h);
        auto sk1 = secret_out[1].template get_access<sycl::access::mode::write>(h);
        auto sk2 = secret_out[2].template get_access<sycl::access::mode::write>(h);
        auto sk3 = secret_out[3].template get_access<sycl::access::mode::write>(h);
        auto sk4 = secret_out[4].template get_access<sycl::access::mode::write>(h);
        auto sk5 = secret_out[5].template get_access<sycl::access::mode::write>(h);
        auto c10 = c1_out[0].template get_access<sycl::access::mode::write>(h);
        auto c11 = c1_out[1].template get_access<sycl::access::mode::write>(h);
        auto c12 = c1_out[2].template get_access<sycl::access::mode::write>(h);
        auto c13 = c1_out[3].template get_access<sycl::access::mode::write>(h);
        auto c14 = c1_out[4].template get_access<sycl::access::mode::write>(h);
        auto c15 = c1_out[5].template get_access<sycl::access::mode::write>(h);

        h.single_task<DrainEntryKernelTask>([=]() [[intel::kernel_args_restrict]] {
            for (size_t i = 0; i < NUM_BLOCKS; ++i) {
                enc[i] = SharedToIFFTPipe::read();
                err0[i] = ErrorToScaleReducePipes::PipeAt<0>::read();
                err1[i] = ErrorToScaleReducePipes::PipeAt<1>::read();
                err2[i] = ErrorToScaleReducePipes::PipeAt<2>::read();
                err3[i] = ErrorToScaleReducePipes::PipeAt<3>::read();
                err4[i] = ErrorToScaleReducePipes::PipeAt<4>::read();
                err5[i] = ErrorToScaleReducePipes::PipeAt<5>::read();
                sk0[i] = PipeSet<0>::EntryToNTTAPipe::read();
                sk1[i] = PipeSet<1>::EntryToNTTAPipe::read();
                sk2[i] = PipeSet<2>::EntryToNTTAPipe::read();
                sk3[i] = PipeSet<3>::EntryToNTTAPipe::read();
                sk4[i] = PipeSet<4>::EntryToNTTAPipe::read();
                sk5[i] = PipeSet<5>::EntryToNTTAPipe::read();
                c10[i] = PipeSet<0>::EntryToPolyMultNegPipe::read();
                c11[i] = PipeSet<1>::EntryToPolyMultNegPipe::read();
                c12[i] = PipeSet<2>::EntryToPolyMultNegPipe::read();
                c13[i] = PipeSet<3>::EntryToPolyMultNegPipe::read();
                c14[i] = PipeSet<4>::EntryToPolyMultNegPipe::read();
                c15[i] = PipeSet<5>::EntryToPolyMultNegPipe::read();
            }
        });
    }
};

} // namespace sycl_ckks::harness

int main()
{
    sycl_ckks::harness::host_debug("entry: creating simple host test vectors");

    // Each input block is derived from its index. Conceptually this is the
    // requested test_vector[i] = i pattern, adapted to PipelineInputBlock.
    std::vector<PipelineInputBlock> input(NUM_BLOCKS);
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        input[i] = sycl_ckks::harness::make_test_input_block(i);
    }

    std::vector<encoding_block> encoding_out(NUM_BLOCKS);
    std::array<std::vector<i8x4>, NUM_PHYSICAL_PIPELINES> error_out;
    std::array<std::vector<u32x4>, NUM_PHYSICAL_PIPELINES> secret_out;
    std::array<std::vector<u32x4>, NUM_PHYSICAL_PIPELINES> c1_out;
    for (size_t p = 0; p < NUM_PHYSICAL_PIPELINES; ++p) {
        error_out[p].resize(NUM_BLOCKS);
        secret_out[p].resize(NUM_BLOCKS);
        c1_out[p].resize(NUM_BLOCKS);
    }

    {
        // Buffers are scoped so their destructors copy device results back into
        // the host vectors before the comparisons below run.
        sycl::buffer<PipelineInputBlock, 1> input_buf(input.data(), sycl::range<1>(NUM_BLOCKS));
        sycl::buffer<encoding_block, 1> encoding_buf(encoding_out.data(), sycl::range<1>(NUM_BLOCKS));
        std::array<sycl::buffer<i8x4, 1>, NUM_PHYSICAL_PIPELINES> error_bufs = {
            sycl::buffer<i8x4, 1>(error_out[0].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<i8x4, 1>(error_out[1].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<i8x4, 1>(error_out[2].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<i8x4, 1>(error_out[3].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<i8x4, 1>(error_out[4].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<i8x4, 1>(error_out[5].data(), sycl::range<1>(NUM_BLOCKS)),
        };
        std::array<sycl::buffer<u32x4, 1>, NUM_PHYSICAL_PIPELINES> secret_bufs = {
            sycl::buffer<u32x4, 1>(secret_out[0].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<u32x4, 1>(secret_out[1].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<u32x4, 1>(secret_out[2].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<u32x4, 1>(secret_out[3].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<u32x4, 1>(secret_out[4].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<u32x4, 1>(secret_out[5].data(), sycl::range<1>(NUM_BLOCKS)),
        };
        std::array<sycl::buffer<u32x4, 1>, NUM_PHYSICAL_PIPELINES> c1_bufs = {
            sycl::buffer<u32x4, 1>(c1_out[0].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<u32x4, 1>(c1_out[1].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<u32x4, 1>(c1_out[2].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<u32x4, 1>(c1_out[3].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<u32x4, 1>(c1_out[4].data(), sycl::range<1>(NUM_BLOCKS)),
            sycl::buffer<u32x4, 1>(c1_out[5].data(), sycl::range<1>(NUM_BLOCKS)),
        };

        auto q = sycl_ckks::harness::make_queue();

        // Start the drain first so EntryKernel has consumers for every pipe it
        // writes. The blocking pipe reads then pair with EntryKernel's writes.
        auto drain_event = q.submit([&](sycl::handler& h) {
            sycl_ckks::harness::DrainEntryKernel kernel(encoding_buf, error_bufs, secret_bufs, c1_bufs);
            kernel(h);
        });
        auto entry_event = q.submit([&](sycl::handler& h) {
            EntryKernel<NUM_BLOCKS> kernel(input_buf);
            kernel(h);
        });

        (void)drain_event;
        (void)entry_event;
        q.wait_and_throw();
    }

    sycl_ckks::harness::host_debug("entry: comparing drained pipe data with expected fanout");
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        sycl_ckks::harness::require(sycl_ckks::harness::equal_encoding(encoding_out[i], input[i].encoding), "EntryKernel encoding fanout mismatch");
        for (size_t p = 0; p < NUM_PHYSICAL_PIPELINES; ++p) {
            sycl_ckks::harness::require(sycl_ckks::harness::equal_i8x4(error_out[p][i], input[i].error), "EntryKernel error fanout mismatch");
            sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(secret_out[p][i], input[i].secret_key[p]), "EntryKernel secret-key fanout mismatch");
            sycl_ckks::harness::require(sycl_ckks::harness::equal_u32x4(c1_out[p][i], input[i].c1[p]), "EntryKernel c1 fanout mismatch");
        }
    }

    sycl_ckks::harness::host_debug("entry: PASS");
    return 0;
}
