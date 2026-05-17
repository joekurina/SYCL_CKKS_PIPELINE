#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include "rtl/the_nwc_4k_ntt_sycl.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

struct NTT_A_Tag {};
struct NTT_B_Tag {};

template <int P, typename Tag>
struct NTTPipeTraits;

template <int P>
struct NTTPipeTraits<P, NTT_A_Tag> {
    using InputPipe = typename PipeSet<P>::EntryToNTTAPipe;
    using RTLInputPipe = typename PipeSet<P>::NTTAInputPipe;
    using RTLModSelectorPipe = typename PipeSet<P>::NTTAModSelectorPipe;
    using RTLOutputPipe = typename PipeSet<P>::NTTAOutputPipe;
    using DownstreamPipe = typename PipeSet<P>::NTTAToPolyMultNegPipe;
    using ExitPipe = typename PipeSet<P>::NTTAToExitPipe;
};

template <int P>
struct NTTPipeTraits<P, NTT_B_Tag> {
    using InputPipe = typename PipeSet<P>::ScaleReduceToNTTBPipe;
    using RTLInputPipe = typename PipeSet<P>::NTTBInputPipe;
    using RTLModSelectorPipe = typename PipeSet<P>::NTTBModSelectorPipe;
    using RTLOutputPipe = typename PipeSet<P>::NTTBOutputPipe;
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
            using RTLInput = typename PipeSet<P>::NTTRTLInputData;
            using RTLOutput = typename PipeSet<P>::NTTRTLOutputData;

#ifdef FPGA_EMULATOR
            reg_test_verifyNTT_multi_DUT* rtl_instance = the_nwc_4k_ntt_new_instance();
#endif

            // hls-samples/Tutorials/DesignPatterns/restartable_streaming_kernel
            // uses non-blocking pipe operations plus explicit state advancement so
            // one stalled interface cannot prevent the kernel from servicing other
            // interfaces. Apply the same pattern around the frame-oriented RTL NTT:
            // feed exactly one contiguous input frame, then hold each RTL output as
            // pending until every required destination accepts it.
            size_t input_count = 0;
            size_t output_count = 0;
            bool have_pending_output = false;
            bool need_downstream_write = false;
            bool need_exit_write = false;
            u32x4 pending_output{};

            while (output_count < NUM_BLOCKS) {
                bool input_valid = false;
                u32x4 input_block{};

                if (!have_pending_output && input_count < NUM_BLOCKS) {
                    input_block = Traits::InputPipe::read();
                    input_valid = true;
                    input_count++;
                }

                if (!have_pending_output) {
                    RTLInput rtl_in;
                    rtl_in.x0 = static_cast<int32_t>(input_block.element0);
                    rtl_in.x1 = static_cast<int32_t>(input_block.element1);
                    rtl_in.x2 = static_cast<int32_t>(input_block.element2);
                    rtl_in.x3 = static_cast<int32_t>(input_block.element3);

                    the_nwc_4k_ntt_input_t hw_in;
                    hw_in.port_in_v_s = input_valid;
                    hw_in.port_in_c_s = kernel_mod_sel;
                    hw_in.port_x_in_0 = rtl_in.x0;
                    hw_in.port_x_in_1 = rtl_in.x1;
                    hw_in.port_x_in_2 = rtl_in.x2;
                    hw_in.port_x_in_3 = rtl_in.x3;

#ifdef FPGA_EMULATOR
                    the_nwc_4k_ntt_output_t hw_out = the_nwc_4k_ntt(rtl_instance, hw_in);
#else
                    the_nwc_4k_ntt_output_t hw_out = the_nwc_4k_ntt(hw_in);
#endif

                    if (hw_out.port_out_v_s == 1) {
                        pending_output.element0 = static_cast<uint32_t>(hw_out.port_out_q_0);
                        pending_output.element1 = static_cast<uint32_t>(hw_out.port_out_q_1);
                        pending_output.element2 = static_cast<uint32_t>(hw_out.port_out_q_2);
                        pending_output.element3 = static_cast<uint32_t>(hw_out.port_out_q_3);
                        have_pending_output = true;
                        need_downstream_write = true;
                        need_exit_write = kernel_write_exit;
                    }
                }

                if (have_pending_output) {
                    if (need_downstream_write) {
                        bool wrote_downstream = false;
                        Traits::DownstreamPipe::write(pending_output, wrote_downstream);
                        if (wrote_downstream) {
                            need_downstream_write = false;
                        }
                    }

                    if (need_exit_write) {
                        bool wrote_exit = false;
                        Traits::ExitPipe::write(pending_output, wrote_exit);
                        if (wrote_exit) {
                            need_exit_write = false;
                        }
                    }

                    if (!need_downstream_write && !need_exit_write) {
                        have_pending_output = false;
                        output_count++;
                    }
                }
            }

#ifdef FPGA_EMULATOR
            the_nwc_4k_ntt_delete_instance(rtl_instance);
#endif
        });
    }
};

template <int P>
using NTTKernelA = NTTKernel<P, NTT_A_Tag>;

template <int P>
using NTTKernelB = NTTKernel<P, NTT_B_Tag>;

}
