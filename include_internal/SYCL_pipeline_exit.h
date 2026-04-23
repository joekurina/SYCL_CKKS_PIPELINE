#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

template <int P>
class ExitKernelTask;

template <int P>
class ExitKernel {
private:
    mutable sycl::buffer<PerModulusOutputBlock, 1> output_buf;
    bool save_ntt_s;
    bool save_ntt_pte;

public:
    ExitKernel(sycl::buffer<PerModulusOutputBlock, 1>& buf, bool save_s = false, bool save_pte = false)
        : output_buf(buf), save_ntt_s(save_s), save_ntt_pte(save_pte) {}

    void operator()(sycl::handler& h) const {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);
        bool kernel_save_s = save_ntt_s;
        bool kernel_save_pte = save_ntt_pte;

        h.single_task<ExitKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;

            [[intel::initiation_interval(1)]]
            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                PerModulusOutputBlock out_block{};

                out_block.c0 = Pipes::PolyAddToExitPipe::read();

                if (kernel_save_s) {
                    out_block.ntt_s = Pipes::NTTAToExitPipe::read();
                }

                if (kernel_save_pte) {
                    out_block.ntt_pte = Pipes::NTTBToExitPipe::read();
                }

                output[blk] = out_block;
            }
        });
    }
};

}
