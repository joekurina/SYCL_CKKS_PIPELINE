#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <complex>
typedef std::complex<double> complex_double;
#else
#include <complex.h>
typedef double complex complex_double;
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SYCL_POLY_N 8192
#define SYCL_NUM_MODULI 6
#define SYCL_NUM_PHYSICAL_PIPELINES 6

void SYCL_encrypt(
    size_t n,
    const double* scales,
    const uint32_t* mod_values,
    const uint32_t* const_ratios,  // [SYCL_NUM_MODULI * 2]: {cr0_p0, cr1_p0, ...}
    const complex_double* encoding_buffer,
    const int8_t* error_samples,
    const uint32_t* const* secret_keys,
    const uint32_t* const* uniform_polys,
    uint32_t** c0_outputs,
    uint32_t** c1_outputs,
    uint32_t** s_save,
    uint32_t** c1_save,
    uint32_t** ntt_pte_outputs);

#ifdef __cplusplus
}
#endif
