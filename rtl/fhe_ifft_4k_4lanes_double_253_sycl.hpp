// ------------------------------------------------------------------------- 
// High Level Design Compiler for Altera(R) FPGAs Version 25.3 (Release Build #17e6417164)
// Software model created on 2026-04-02 08:17:35
// Generation mode: Bit Accurate
// ------------------------------------------------------------------------- 
#ifndef SOFTWARE_MODEL_WRAPPER_FHE_IFFT_4K_4LANES_DOUBLE_253_H_
#define SOFTWARE_MODEL_WRAPPER_FHE_IFFT_4K_4LANES_DOUBLE_253_H_
class ifft4k_base_DUT;

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

}) fhe_ifft_4k_4lanes_double_253_input_t;

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

}) fhe_ifft_4k_4lanes_double_253_output_t;

#ifdef FPGA_EMULATOR
#ifdef NO_SYCL
fhe_ifft_4k_4lanes_double_253_output_t fhe_ifft_4k_4lanes_double_253(ifft4k_base_DUT* instance, fhe_ifft_4k_4lanes_double_253_input_t input);
#else
SYCL_EXTERNAL fhe_ifft_4k_4lanes_double_253_output_t fhe_ifft_4k_4lanes_double_253(ifft4k_base_DUT* instance, fhe_ifft_4k_4lanes_double_253_input_t input);
SYCL_EXTERNAL ifft4k_base_DUT* fhe_ifft_4k_4lanes_double_253_new_instance();
SYCL_EXTERNAL void fhe_ifft_4k_4lanes_double_253_delete_instance(ifft4k_base_DUT* instance);
#endif
#else
SYCL_EXTERNAL fhe_ifft_4k_4lanes_double_253_output_t fhe_ifft_4k_4lanes_double_253(fhe_ifft_4k_4lanes_double_253_input_t input);
#endif

#ifdef __cplusplus
}
#endif 

#endif // SOFTWARE_MODEL_WRAPPER_FHE_IFFT_4K_4LANES_DOUBLE_253_H_
