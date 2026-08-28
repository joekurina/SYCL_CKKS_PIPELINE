#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include "rtl/the_nwc_8k_ntt_sycl.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

// Set to 1 only for source-level diagnostics that intentionally bypass the
// imported 8K RTL. Production and numerical validation must keep this at 0.
#define NTT_DIAGNOSTIC_BYPASS_RTL 0

static_assert(sizeof(the_nwc_8k_ntt_input_t) == 34,
              "unexpected the_nwc_8k_ntt input ABI size");
static_assert(sizeof(the_nwc_8k_ntt_output_t) == 34,
              "unexpected the_nwc_8k_ntt output ABI size");

inline void pack_ntt_coefficient(uint32_t port[2], uint32_t value)
{
    port[0] = value;
    port[1] = 0;
}

inline uint32_t unpack_ntt_coefficient(const uint32_t port[2])
{
    // All supported 8K moduli are below 2^30, so a valid reduced coefficient
    // occupies the low word of the RTL's 64-bit lane.
    return port[0];
}

struct NTT_A_Tag {};
struct NTT_B_Tag {};

template <int P, typename Tag>
struct NTTPipeTraits;

template <int P>
struct NTTPipeTraits<P, NTT_A_Tag> {
    using InputPipe = typename PipeSet<P>::EntryToNTTAPipe;
    using DownstreamPipe = typename PipeSet<P>::NTTAToPolyMultNegPipe;
    using ExitPipe = typename PipeSet<P>::NTTAToExitPipe;
};

template <int P>
struct NTTPipeTraits<P, NTT_B_Tag> {
    using InputPipe = typename PipeSet<P>::ScaleReduceToNTTBPipe;
    using DownstreamPipe = typename PipeSet<P>::NTTBToPolyAddPipe;
    using ExitPipe = typename PipeSet<P>::NTTBToExitPipe;
};

template <int P, typename Tag>
class NTTKernelTask;

template <int P, typename Tag>
class NTTKernel {
private:
    uint8_t modulus_selector;
    bool write_to_exit;

public:
    NTTKernel(uint8_t mod_sel, bool save_output = false)
        : modulus_selector(mod_sel), write_to_exit(save_output) {}

    void operator()(sycl::handler& h) const {
        uint8_t kernel_mod_sel = modulus_selector;
        bool kernel_write_exit = write_to_exit;

        h.single_task<NTTKernelTask<P, Tag>>([=]() [[intel::kernel_args_restrict]] {
            using Traits = NTTPipeTraits<P, Tag>;

#if !NTT_DIAGNOSTIC_BYPASS_RTL
#ifdef FPGA_EMULATOR
            reg_test_verifyNTT_multi_DUT* rtl_instance = the_nwc_8k_ntt_new_instance();
#endif
#endif

            // The 8K RTL NTT is frame-oriented: it consumes 2048 four-lane
            // beats before bursting an output frame. This service kernel is
            // deliberately persistent so issuing later input beats never
            // depends on observing output from an earlier call. A blocking
            // input read prevents idle iterations from spending imported-RTL
            // call credits. Each of the six physical service kernels owns one
            // fixed modulus selector for its entire frame.
            while (1) {
                u32x4 input_block = Traits::InputPipe::read();

                the_nwc_8k_ntt_input_t hw_in{};
                hw_in.port_in_v_s = 1;
                hw_in.port_in_c_s = static_cast<int8_t>(kernel_mod_sel);
                pack_ntt_coefficient(hw_in.port_x_in_0, input_block.element0);
                pack_ntt_coefficient(hw_in.port_x_in_1, input_block.element1);
                pack_ntt_coefficient(hw_in.port_x_in_2, input_block.element2);
                pack_ntt_coefficient(hw_in.port_x_in_3, input_block.element3);

#if NTT_DIAGNOSTIC_BYPASS_RTL
                the_nwc_8k_ntt_output_t hw_out{};
                hw_out.port_out_v_s = hw_in.port_in_v_s;
                hw_out.port_out_c_s = hw_in.port_in_c_s;
                hw_out.port_out_q_0[0] = hw_in.port_x_in_0[0];
                hw_out.port_out_q_1[0] = hw_in.port_x_in_1[0];
                hw_out.port_out_q_2[0] = hw_in.port_x_in_2[0];
                hw_out.port_out_q_3[0] = hw_in.port_x_in_3[0];
#else
#ifdef FPGA_EMULATOR
                the_nwc_8k_ntt_output_t hw_out = the_nwc_8k_ntt(rtl_instance, hw_in);
#else
                the_nwc_8k_ntt_output_t hw_out = the_nwc_8k_ntt(hw_in);
#endif
#endif

                if (hw_out.port_out_v_s != 0) {
                    u32x4 output_block;
                    output_block.element0 = unpack_ntt_coefficient(hw_out.port_out_q_0);
                    output_block.element1 = unpack_ntt_coefficient(hw_out.port_out_q_1);
                    output_block.element2 = unpack_ntt_coefficient(hw_out.port_out_q_2);
                    output_block.element3 = unpack_ntt_coefficient(hw_out.port_out_q_3);

                    Traits::DownstreamPipe::write(output_block);
                    if (kernel_write_exit) {
                        Traits::ExitPipe::write(output_block);
                    }
                }
            }
        });
    }
};

template <int P>
using NTTKernelA = NTTKernel<P, NTT_A_Tag>;

template <int P>
using NTTKernelB = NTTKernel<P, NTT_B_Tag>;

}
