#pragma once

#include "kernel_test_common.h"
#include "rtl/fhe_ifft_4k_4lanes_double_253_sycl.hpp"
#include "rtl/the_nwc_4k_ntt_sycl.hpp"
#include <vector>

namespace sycl_ckks::test {

#ifdef FPGA_EMULATOR
inline std::vector<encoding_block> reference_ifft(const std::vector<encoding_block>& input)
{
    std::vector<encoding_block> output(NUM_BLOCKS);
    ifft4k_base_DUT* instance = fhe_ifft_4k_4lanes_double_253_new_instance();
    size_t input_count = 0;
    size_t output_count = 0;
    while (output_count < NUM_BLOCKS) {
        encoding_block block{};
        bool input_valid = false;
        if (input_count < NUM_BLOCKS) {
            block = input[input_count++];
            input_valid = true;
        }
        fhe_ifft_4k_4lanes_double_253_input_t hw_in{};
        hw_in.port_v_in_s = input_valid;
        hw_in.port_data_in_0re = block.element0.real();
        hw_in.port_data_in_0im = block.element0.imag();
        hw_in.port_data_in_1re = block.element1.real();
        hw_in.port_data_in_1im = block.element1.imag();
        hw_in.port_data_in_2re = block.element2.real();
        hw_in.port_data_in_2im = block.element2.imag();
        hw_in.port_data_in_3re = block.element3.real();
        hw_in.port_data_in_3im = block.element3.imag();
        auto hw_out = fhe_ifft_4k_4lanes_double_253(instance, hw_in);
        if (hw_out.port_v_out_s == 1) {
            encoding_block out{};
            out.element0 = complex_double(hw_out.port_data_out_0re, hw_out.port_data_out_0im);
            out.element1 = complex_double(hw_out.port_data_out_1re, hw_out.port_data_out_1im);
            out.element2 = complex_double(hw_out.port_data_out_2re, hw_out.port_data_out_2im);
            out.element3 = complex_double(hw_out.port_data_out_3re, hw_out.port_data_out_3im);
            output[output_count++] = out;
        }
    }
    fhe_ifft_4k_4lanes_double_253_delete_instance(instance);
    return output;
}

inline std::vector<u32x4> reference_ntt(const std::vector<u32x4>& input, uint8_t modulus_selector)
{
    std::vector<u32x4> output(NUM_BLOCKS);
    reg_test_verifyNTT_multi_DUT* instance = the_nwc_4k_ntt_new_instance();
    size_t input_count = 0;
    size_t output_count = 0;
    while (output_count < NUM_BLOCKS) {
        u32x4 block{};
        bool input_valid = false;
        if (input_count < NUM_BLOCKS) {
            block = input[input_count++];
            input_valid = true;
        }
        the_nwc_4k_ntt_input_t hw_in{};
        hw_in.port_in_v_s = input_valid;
        hw_in.port_in_c_s = modulus_selector;
        hw_in.port_x_in_0 = static_cast<int32_t>(block.element0);
        hw_in.port_x_in_1 = static_cast<int32_t>(block.element1);
        hw_in.port_x_in_2 = static_cast<int32_t>(block.element2);
        hw_in.port_x_in_3 = static_cast<int32_t>(block.element3);
        auto hw_out = the_nwc_4k_ntt(instance, hw_in);
        if (hw_out.port_out_v_s == 1) {
            u32x4 out{};
            out.element0 = static_cast<uint32_t>(hw_out.port_out_q_0);
            out.element1 = static_cast<uint32_t>(hw_out.port_out_q_1);
            out.element2 = static_cast<uint32_t>(hw_out.port_out_q_2);
            out.element3 = static_cast<uint32_t>(hw_out.port_out_q_3);
            output[output_count++] = out;
        }
    }
    the_nwc_4k_ntt_delete_instance(instance);
    return output;
}
#endif

}  // namespace sycl_ckks::test
