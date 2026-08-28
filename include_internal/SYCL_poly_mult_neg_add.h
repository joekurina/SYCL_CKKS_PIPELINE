#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

template <int P>
class PolyMultNegAddKernelTask;

template <int P>
class PolyMultNegAddKernel {
private:
    uint32_t mod_value;
    uint32_t const_ratio[2];

public:
    PolyMultNegAddKernel(uint32_t mod, const uint32_t* cr)
        : mod_value(mod)
    {
        const_ratio[0] = cr[0];
        const_ratio[1] = cr[1];
    }

    void operator()(sycl::handler& h) const {
        uint32_t kernel_mod = mod_value;
        uint32_t kernel_cr0 = const_ratio[0];
        uint32_t kernel_cr1 = const_ratio[1];

        h.single_task<PolyMultNegAddKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;

            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                u32x4 ntt_s = Pipes::NTTAToPolyMultNegPipe::read();
                u32x4 c1 = Pipes::EntryToPolyMultNegPipe::read();
                u32x4 ntt_pte = Pipes::NTTBToPolyAddPipe::read();

                uint64_t prod0 = static_cast<uint64_t>(ntt_s.element0) * static_cast<uint64_t>(c1.element0);
                uint64_t prod1 = static_cast<uint64_t>(ntt_s.element1) * static_cast<uint64_t>(c1.element1);
                uint64_t prod2 = static_cast<uint64_t>(ntt_s.element2) * static_cast<uint64_t>(c1.element2);
                uint64_t prod3 = static_cast<uint64_t>(ntt_s.element3) * static_cast<uint64_t>(c1.element3);

                uint32_t red0 = barrett_reduce_u64_core(prod0, kernel_mod, kernel_cr0, kernel_cr1);
                uint32_t red1 = barrett_reduce_u64_core(prod1, kernel_mod, kernel_cr0, kernel_cr1);
                uint32_t red2 = barrett_reduce_u64_core(prod2, kernel_mod, kernel_cr0, kernel_cr1);
                uint32_t red3 = barrett_reduce_u64_core(prod3, kernel_mod, kernel_cr0, kernel_cr1);

                uint32_t neg0 = mod_neg(red0, kernel_mod);
                uint32_t neg1 = mod_neg(red1, kernel_mod);
                uint32_t neg2 = mod_neg(red2, kernel_mod);
                uint32_t neg3 = mod_neg(red3, kernel_mod);

                u32x4 out;
                out.element0 = mod_add(neg0, ntt_pte.element0, kernel_mod);
                out.element1 = mod_add(neg1, ntt_pte.element1, kernel_mod);
                out.element2 = mod_add(neg2, ntt_pte.element2, kernel_mod);
                out.element3 = mod_add(neg3, ntt_pte.element3, kernel_mod);

                Pipes::PolyAddToExitPipe::write(out);
            }
        });
    }
};

}
