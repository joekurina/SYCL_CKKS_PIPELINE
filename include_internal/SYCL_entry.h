#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

template <size_t TotalBlocks>
class EntryKernelTask;

template <size_t TotalBlocks = NUM_BLOCKS>
class EntryKernel {
private:
    mutable sycl::buffer<PipelineInputBlock, 1> input_buf;

public:
    EntryKernel(sycl::buffer<PipelineInputBlock, 1>& buf) : input_buf(buf) {}

    void operator()(sycl::handler& h) const {
        auto input = input_buf.template get_access<sycl::access::mode::read>(h);

        h.single_task<EntryKernelTask<TotalBlocks>>([=]() [[intel::kernel_args_restrict]] {
            // One input frame feeds all six physical modulus pipelines in
            // parallel. Encoding and error values are broadcast; secret-key
            // and c1 values remain per-modulus.
            for (size_t blk = 0; blk < TotalBlocks; ++blk) {
                PipelineInputBlock block = input[blk];

                SharedToIFFTPipe::write(block.encoding);
                ErrorToScaleReducePipes::write(block.error);

                PipeSet<0>::EntryToNTTAPipe::write(block.secret_key[0]);
                PipeSet<1>::EntryToNTTAPipe::write(block.secret_key[1]);
                PipeSet<2>::EntryToNTTAPipe::write(block.secret_key[2]);
                PipeSet<3>::EntryToNTTAPipe::write(block.secret_key[3]);
                PipeSet<4>::EntryToNTTAPipe::write(block.secret_key[4]);
                PipeSet<5>::EntryToNTTAPipe::write(block.secret_key[5]);

                PipeSet<0>::EntryToPolyMultNegPipe::write(block.c1[0]);
                PipeSet<1>::EntryToPolyMultNegPipe::write(block.c1[1]);
                PipeSet<2>::EntryToPolyMultNegPipe::write(block.c1[2]);
                PipeSet<3>::EntryToPolyMultNegPipe::write(block.c1[3]);
                PipeSet<4>::EntryToPolyMultNegPipe::write(block.c1[4]);
                PipeSet<5>::EntryToPolyMultNegPipe::write(block.c1[5]);
            }
        });
    }
};

}
