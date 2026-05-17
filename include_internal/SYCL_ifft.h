#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include "rtl/fhe_ifft_4k_4lanes_double_253_sycl.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

class IFFTKernelTask;

class IFFTKernel {
public:
    IFFTKernel() {}

    void operator()(sycl::handler& h) const {
        h.single_task<IFFTKernelTask>([=]() [[intel::kernel_args_restrict]] {

#ifdef FPGA_EMULATOR
            ifft4k_base_DUT* rtl_instance = fhe_ifft_4k_4lanes_double_253_new_instance();
#endif

            // The RTL IFFT consumes a full 4096-point frame. Use blocking reads
            // for the input frame so startup skew between EntryKernel and IFFTKernel
            // cannot turn missing pipe data into dropped samples. Then keep draining
            // the RTL until exactly NUM_BLOCKS valid output blocks are forwarded.
            size_t input_count = 0;
            size_t output_count = 0;

            // hls-samples/Tutorials/DesignPatterns/restartable_streaming_kernel
            // and include/pipe_utils.hpp show the useful liveness pattern here:
            // keep explicit state, use non-blocking writes on multi-interface
            // fanout, and advance the frame counter only after every destination
            // has accepted the pending beat. Do not keep advancing the RTL while
            // a previous output beat is still pending, or a later valid output
            // could be dropped.
            bool have_pending_output = false;
            bool need_scale_reduce_0 = false;
            bool need_scale_reduce_1 = false;
            bool need_scale_reduce_2 = false;
            encoding_block pending_output{};

            while (output_count < NUM_BLOCKS) {
                bool input_valid = false;
                encoding_block block{};

                if (!have_pending_output) {
                    if (input_count < NUM_BLOCKS) {
                        block = SharedToIFFTPipe::read();
                        input_valid = true;
                        input_count++;
                    }

                    fhe_ifft_4k_4lanes_double_253_input_t hw_in;
                    hw_in.port_v_in_s = input_valid;
                    hw_in.port_data_in_0re = block.element0.real();
                    hw_in.port_data_in_0im = block.element0.imag();
                    hw_in.port_data_in_1re = block.element1.real();
                    hw_in.port_data_in_1im = block.element1.imag();
                    hw_in.port_data_in_2re = block.element2.real();
                    hw_in.port_data_in_2im = block.element2.imag();
                    hw_in.port_data_in_3re = block.element3.real();
                    hw_in.port_data_in_3im = block.element3.imag();

#ifdef FPGA_EMULATOR
                    fhe_ifft_4k_4lanes_double_253_output_t hw_out = fhe_ifft_4k_4lanes_double_253(rtl_instance, hw_in);
#else
                    fhe_ifft_4k_4lanes_double_253_output_t hw_out = fhe_ifft_4k_4lanes_double_253(hw_in);
#endif

                    if (hw_out.port_v_out_s == 1) {
                        pending_output.element0 = complex_double(hw_out.port_data_out_0re, hw_out.port_data_out_0im);
                        pending_output.element1 = complex_double(hw_out.port_data_out_1re, hw_out.port_data_out_1im);
                        pending_output.element2 = complex_double(hw_out.port_data_out_2re, hw_out.port_data_out_2im);
                        pending_output.element3 = complex_double(hw_out.port_data_out_3re, hw_out.port_data_out_3im);
                        have_pending_output = true;
                        need_scale_reduce_0 = true;
                        need_scale_reduce_1 = true;
                        need_scale_reduce_2 = true;
                    }
                }

                if (have_pending_output) {
                    if (need_scale_reduce_0) {
                        bool wrote_0 = false;
                        IFFTToScaleReducePipes::PipeAt<0>::write(pending_output, wrote_0);
                        if (wrote_0) {
                            need_scale_reduce_0 = false;
                        }
                    }

                    if (need_scale_reduce_1) {
                        bool wrote_1 = false;
                        IFFTToScaleReducePipes::PipeAt<1>::write(pending_output, wrote_1);
                        if (wrote_1) {
                            need_scale_reduce_1 = false;
                        }
                    }

                    if (need_scale_reduce_2) {
                        bool wrote_2 = false;
                        IFFTToScaleReducePipes::PipeAt<2>::write(pending_output, wrote_2);
                        if (wrote_2) {
                            need_scale_reduce_2 = false;
                        }
                    }

                    if (!need_scale_reduce_0 && !need_scale_reduce_1 && !need_scale_reduce_2) {
                        have_pending_output = false;
                        output_count++;
                    }
                }
            }

#ifdef FPGA_EMULATOR
            fhe_ifft_4k_4lanes_double_253_delete_instance(rtl_instance);
#endif
        });
    }
};

}
