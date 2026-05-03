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

            while (output_count < NUM_BLOCKS) {
                bool input_valid = false;
                encoding_block block{};

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
                    encoding_block out;
                    out.element0 = complex_double(hw_out.port_data_out_0re, hw_out.port_data_out_0im);
                    out.element1 = complex_double(hw_out.port_data_out_1re, hw_out.port_data_out_1im);
                    out.element2 = complex_double(hw_out.port_data_out_2re, hw_out.port_data_out_2im);
                    out.element3 = complex_double(hw_out.port_data_out_3re, hw_out.port_data_out_3im);

                    IFFTToScaleReducePipes::write(out);
                    output_count++;
                }
            }

#ifdef FPGA_EMULATOR
            fhe_ifft_4k_4lanes_double_253_delete_instance(rtl_instance);
#endif
        });
    }
};

}
