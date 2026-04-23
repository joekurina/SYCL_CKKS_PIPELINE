// ------------------------------------------------------------------------- 
// High Level Design Compiler for Altera(R) FPGAs Version 25.3 (Release Build #17e6417164)
// Software model created on 2025-10-01 16:11:47
// Generation mode: Bit Accurate
// ------------------------------------------------------------------------- 
#ifndef SOFTWARE_MODEL_WRAPPER_THE_NWC_4K_NTT_H_
#define SOFTWARE_MODEL_WRAPPER_THE_NWC_4K_NTT_H_
class reg_test_verifyNTT_multi_DUT;

#ifndef NO_SYCL
#include <sycl/sycl.hpp>
#endif
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #define CSL_PACKED( struct_def ) __pragma( pack(push, 1) ) struct_def __pragma( pack(pop) )
#else
    #define CSL_PACKED( struct_def ) struct_def __attribute__((__packed__))
#endif

CSL_PACKED(typedef struct
{
    int8_t port_in_v_s;
    int8_t port_in_c_s;
    int32_t port_x_in_0;
    int32_t port_x_in_1;
    int32_t port_x_in_2;
    int32_t port_x_in_3;

}) the_nwc_4k_ntt_input_t;

CSL_PACKED(typedef struct
{
    int8_t port_out_v_s;
    int8_t port_out_c_s;
    int32_t port_out_q_0;
    int32_t port_out_q_1;
    int32_t port_out_q_2;
    int32_t port_out_q_3;

}) the_nwc_4k_ntt_output_t;

#ifdef FPGA_EMULATOR
#ifdef NO_SYCL
the_nwc_4k_ntt_output_t the_nwc_4k_ntt(reg_test_verifyNTT_multi_DUT* instance, the_nwc_4k_ntt_input_t input);
#else
SYCL_EXTERNAL the_nwc_4k_ntt_output_t the_nwc_4k_ntt(reg_test_verifyNTT_multi_DUT* instance, the_nwc_4k_ntt_input_t input);
SYCL_EXTERNAL reg_test_verifyNTT_multi_DUT* the_nwc_4k_ntt_new_instance();
SYCL_EXTERNAL void the_nwc_4k_ntt_delete_instance(reg_test_verifyNTT_multi_DUT* instance);
#endif
#else
SYCL_EXTERNAL the_nwc_4k_ntt_output_t the_nwc_4k_ntt(the_nwc_4k_ntt_input_t input);
#endif

#ifdef __cplusplus
}
#endif 

#endif // SOFTWARE_MODEL_WRAPPER_THE_NWC_4K_NTT_H_
