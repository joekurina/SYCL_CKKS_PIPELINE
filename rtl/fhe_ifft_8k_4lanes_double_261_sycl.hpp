/**
 * High Level Design Compiler for Altera(R) FPGAs Version 26.1 (Release Build #805d06383c)
 * Software model created on 2026-08-25 13:21:30
 * Generation mode: Bit Accurate
 */
#ifndef SOFTWARE_MODEL_WRAPPER_FHE_IFFT_8K_4LANES_DOUBLE_261_H_
#define SOFTWARE_MODEL_WRAPPER_FHE_IFFT_8K_4LANES_DOUBLE_261_H_
class ifft8k_base_DUT;

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
    int8_t port_v_in_s;
    double port_data_in_0re;
    double port_data_in_0im;
    double port_data_in_1re;
    double port_data_in_1im;
    double port_data_in_2re;
    double port_data_in_2im;
    double port_data_in_3re;
    double port_data_in_3im;

}) fhe_ifft_8k_4lanes_double_261_input_t;

CSL_PACKED(typedef struct
{
    int8_t port_v_out_s;
    double port_data_out_0re;
    double port_data_out_0im;
    double port_data_out_1re;
    double port_data_out_1im;
    double port_data_out_2re;
    double port_data_out_2im;
    double port_data_out_3re;
    double port_data_out_3im;

}) fhe_ifft_8k_4lanes_double_261_output_t;

#ifdef FPGA_EMULATOR
#ifdef NO_SYCL
fhe_ifft_8k_4lanes_double_261_output_t fhe_ifft_8k_4lanes_double_261(ifft8k_base_DUT* instance, fhe_ifft_8k_4lanes_double_261_input_t input);
#else
SYCL_EXTERNAL fhe_ifft_8k_4lanes_double_261_output_t fhe_ifft_8k_4lanes_double_261(ifft8k_base_DUT* instance, fhe_ifft_8k_4lanes_double_261_input_t input);
SYCL_EXTERNAL ifft8k_base_DUT* fhe_ifft_8k_4lanes_double_261_new_instance();
SYCL_EXTERNAL void fhe_ifft_8k_4lanes_double_261_delete_instance(ifft8k_base_DUT* instance);
#endif
#else
SYCL_EXTERNAL fhe_ifft_8k_4lanes_double_261_output_t fhe_ifft_8k_4lanes_double_261(fhe_ifft_8k_4lanes_double_261_input_t input);
#endif

#ifdef __cplusplus
}
#endif 

#endif // SOFTWARE_MODEL_WRAPPER_FHE_IFFT_8K_4LANES_DOUBLE_261_H_
