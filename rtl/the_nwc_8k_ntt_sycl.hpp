/**
 * High Level Design Compiler for Altera(R) FPGAs Version 2026.3 (Release Build #847757300c)
 * Software model created on 2026-08-26 10:03:58
 * Generation mode: Bit Accurate
 */
#ifndef SOFTWARE_MODEL_WRAPPER_THE_NWC_8K_NTT_H_
#define SOFTWARE_MODEL_WRAPPER_THE_NWC_8K_NTT_H_
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
    uint32_t port_x_in_0[2];
    uint32_t port_x_in_1[2];
    uint32_t port_x_in_2[2];
    uint32_t port_x_in_3[2];

}) the_nwc_8k_ntt_input_t;

CSL_PACKED(typedef struct
{
    int8_t port_out_v_s;
    int8_t port_out_c_s;
    uint32_t port_out_q_0[2];
    uint32_t port_out_q_1[2];
    uint32_t port_out_q_2[2];
    uint32_t port_out_q_3[2];

}) the_nwc_8k_ntt_output_t;

#ifdef FPGA_EMULATOR
#ifdef NO_SYCL
the_nwc_8k_ntt_output_t the_nwc_8k_ntt(reg_test_verifyNTT_multi_DUT* instance, the_nwc_8k_ntt_input_t input);
#else
SYCL_EXTERNAL the_nwc_8k_ntt_output_t the_nwc_8k_ntt(reg_test_verifyNTT_multi_DUT* instance, the_nwc_8k_ntt_input_t input);
SYCL_EXTERNAL reg_test_verifyNTT_multi_DUT* the_nwc_8k_ntt_new_instance();
SYCL_EXTERNAL void the_nwc_8k_ntt_delete_instance(reg_test_verifyNTT_multi_DUT* instance);
#endif
#else
SYCL_EXTERNAL the_nwc_8k_ntt_output_t the_nwc_8k_ntt(the_nwc_8k_ntt_input_t input);
#endif

#ifdef __cplusplus
}
#endif 

#endif // SOFTWARE_MODEL_WRAPPER_THE_NWC_8K_NTT_H_
