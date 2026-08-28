#pragma once

#include "SYCL_common.h"
#include <cstdint>
#include <complex>

namespace sycl_ckks {

#ifndef CSL_PACKED
#ifdef _WIN32
#define CSL_PACKED(struct_def) __pragma(pack(push, 1)) struct_def __pragma(pack(pop))
#else
#define CSL_PACKED(struct_def) struct_def __attribute__((__packed__))
#endif
#endif

using complex_double = std::complex<double>;

CSL_PACKED(struct encoding_block {
    complex_double element0;
    complex_double element1;
    complex_double element2;
    complex_double element3;
});

CSL_PACKED(struct u32x4 {
    uint32_t element0;
    uint32_t element1;
    uint32_t element2;
    uint32_t element3;
});

CSL_PACKED(struct i8x4 {
    int8_t element0;
    int8_t element1;
    int8_t element2;
    int8_t element3;
});

CSL_PACKED(struct i64x4 {
    int64_t element0;
    int64_t element1;
    int64_t element2;
    int64_t element3;
});

static_assert(sizeof(encoding_block) == sizeof(complex_double) * 4, "encoding_block size mismatch");
static_assert(sizeof(u32x4) == sizeof(uint32_t) * 4, "u32x4 size mismatch");
static_assert(sizeof(i8x4) == sizeof(int8_t) * 4, "i8x4 size mismatch");
static_assert(sizeof(i64x4) == sizeof(int64_t) * 4, "i64x4 size mismatch");

struct PipelineInputBlock {
    encoding_block encoding;
    i8x4 error;
    u32x4 secret_key[NUM_PHYSICAL_PIPELINES];
    u32x4 c1[NUM_PHYSICAL_PIPELINES];
};

struct PerModulusOutputBlock {
    u32x4 c0;
    u32x4 ntt_s;
    u32x4 ntt_pte;
};

template <typename Scalar>
inline void pack_scalar_to_block(const Scalar* src, size_t block_idx, u32x4& dst)
{
    size_t base = block_idx * LANES;
    dst.element0 = static_cast<uint32_t>(src[base + 0]);
    dst.element1 = static_cast<uint32_t>(src[base + 1]);
    dst.element2 = static_cast<uint32_t>(src[base + 2]);
    dst.element3 = static_cast<uint32_t>(src[base + 3]);
}

inline void pack_error_to_block(const int8_t* src, size_t block_idx, i8x4& dst)
{
    size_t base = block_idx * LANES;
    dst.element0 = src[base + 0];
    dst.element1 = src[base + 1];
    dst.element2 = src[base + 2];
    dst.element3 = src[base + 3];
}

inline void pack_encoding_to_block(const complex_double* src, size_t block_idx, encoding_block& dst)
{
    size_t base = block_idx * LANES;
    dst.element0 = src[base + 0];
    dst.element1 = src[base + 1];
    dst.element2 = src[base + 2];
    dst.element3 = src[base + 3];
}

template <typename Scalar>
inline void unpack_block_to_scalar(const u32x4& src, size_t block_idx, Scalar* dst)
{
    size_t base = block_idx * LANES;
    dst[base + 0] = static_cast<Scalar>(src.element0);
    dst[base + 1] = static_cast<Scalar>(src.element1);
    dst[base + 2] = static_cast<Scalar>(src.element2);
    dst[base + 3] = static_cast<Scalar>(src.element3);
}

}
