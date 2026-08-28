#include "sycl_ckks_accelerator/SYCL_ckks_sym.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "SYCL_common.h"
#include "SYCL_data_types.h"
#include "SYCL_pipes.h"
#include "SYCL_entry.h"
#include "SYCL_pipeline_exit.h"
#include "SYCL_ntt.h"
#include "SYCL_ifft.h"
#include "SYCL_scale_and_reduce.h"
#include "SYCL_poly_mult_neg_add.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace sycl;
using namespace sycl_ckks;

static_assert(POLY_N == SYCL_POLY_N, "public and internal polynomial degrees differ");
static_assert(NUM_MODULI == SYCL_NUM_MODULI, "public and internal modulus counts differ");
static_assert(NUM_PHYSICAL_PIPELINES == SYCL_NUM_PHYSICAL_PIPELINES,
              "public and internal physical-pipeline counts differ");

struct ModulusParams {
    double scale;
    uint32_t mod_value;
    uint32_t const_ratio[2];
    uint8_t modulus_selector;
    bool save_ntt_s;
    bool save_ntt_pte;
};

static void pack_input(
    size_t n,
    const complex_double* encoding_buffer,
    const int8_t* error_samples,
    const uint32_t* const* secret_keys,
    const uint32_t* const* c1_polys,
    std::vector<PipelineInputBlock>& input_blocks)
{
    size_t num_blocks = n / LANES;
    input_blocks.resize(num_blocks);

    for (size_t blk = 0; blk < num_blocks; ++blk) {
        PipelineInputBlock& block = input_blocks[blk];
        pack_encoding_to_block(encoding_buffer, blk, block.encoding);
        pack_error_to_block(error_samples, blk, block.error);

        for (size_t p = 0; p < NUM_MODULI; ++p) {
            pack_scalar_to_block(secret_keys[p], blk, block.secret_key[p]);
            pack_scalar_to_block(c1_polys[p], blk, block.c1[p]);
        }
    }
}

static void unpack_u32_blocks(
    size_t n,
    const std::vector<u32x4>& blocks,
    uint32_t* out)
{
    size_t num_blocks = n / LANES;
    for (size_t blk = 0; blk < num_blocks; ++blk) {
        unpack_block_to_scalar(blocks[blk], blk, out);
    }
}

static std::vector<event> run_pipeline(
    queue& q,
    buffer<PipelineInputBlock, 1>& input_buf,
    std::array<buffer<u32x4, 1>, NUM_MODULI>& c0_bufs,
    std::array<buffer<u32x4, 1>, NUM_MODULI>& ntt_s_bufs,
    std::array<buffer<u32x4, 1>, NUM_MODULI>& ntt_pte_bufs,
    const std::array<ModulusParams, NUM_MODULI>& mod_params)
{
    try {
        std::vector<event> events;

        events.push_back(q.submit([&](handler& h) { ExitC0Kernel<0> kernel(c0_bufs[0]); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ExitC0Kernel<1> kernel(c0_bufs[1]); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ExitC0Kernel<2> kernel(c0_bufs[2]); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ExitC0Kernel<3> kernel(c0_bufs[3]); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ExitC0Kernel<4> kernel(c0_bufs[4]); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ExitC0Kernel<5> kernel(c0_bufs[5]); kernel(h); }));

        if (mod_params[0].save_ntt_s) events.push_back(q.submit([&](handler& h) { ExitNTTASKernel<0> kernel(ntt_s_bufs[0]); kernel(h); }));
        if (mod_params[1].save_ntt_s) events.push_back(q.submit([&](handler& h) { ExitNTTASKernel<1> kernel(ntt_s_bufs[1]); kernel(h); }));
        if (mod_params[2].save_ntt_s) events.push_back(q.submit([&](handler& h) { ExitNTTASKernel<2> kernel(ntt_s_bufs[2]); kernel(h); }));
        if (mod_params[3].save_ntt_s) events.push_back(q.submit([&](handler& h) { ExitNTTASKernel<3> kernel(ntt_s_bufs[3]); kernel(h); }));
        if (mod_params[4].save_ntt_s) events.push_back(q.submit([&](handler& h) { ExitNTTASKernel<4> kernel(ntt_s_bufs[4]); kernel(h); }));
        if (mod_params[5].save_ntt_s) events.push_back(q.submit([&](handler& h) { ExitNTTASKernel<5> kernel(ntt_s_bufs[5]); kernel(h); }));

        if (mod_params[0].save_ntt_pte) events.push_back(q.submit([&](handler& h) { ExitNTTBKernel<0> kernel(ntt_pte_bufs[0]); kernel(h); }));
        if (mod_params[1].save_ntt_pte) events.push_back(q.submit([&](handler& h) { ExitNTTBKernel<1> kernel(ntt_pte_bufs[1]); kernel(h); }));
        if (mod_params[2].save_ntt_pte) events.push_back(q.submit([&](handler& h) { ExitNTTBKernel<2> kernel(ntt_pte_bufs[2]); kernel(h); }));
        if (mod_params[3].save_ntt_pte) events.push_back(q.submit([&](handler& h) { ExitNTTBKernel<3> kernel(ntt_pte_bufs[3]); kernel(h); }));
        if (mod_params[4].save_ntt_pte) events.push_back(q.submit([&](handler& h) { ExitNTTBKernel<4> kernel(ntt_pte_bufs[4]); kernel(h); }));
        if (mod_params[5].save_ntt_pte) events.push_back(q.submit([&](handler& h) { ExitNTTBKernel<5> kernel(ntt_pte_bufs[5]); kernel(h); }));

        events.push_back(q.submit([&](handler& h) { PolyMultNegAddKernel<0> kernel(mod_params[0].mod_value, mod_params[0].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { PolyMultNegAddKernel<1> kernel(mod_params[1].mod_value, mod_params[1].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { PolyMultNegAddKernel<2> kernel(mod_params[2].mod_value, mod_params[2].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { PolyMultNegAddKernel<3> kernel(mod_params[3].mod_value, mod_params[3].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { PolyMultNegAddKernel<4> kernel(mod_params[4].mod_value, mod_params[4].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { PolyMultNegAddKernel<5> kernel(mod_params[5].mod_value, mod_params[5].const_ratio); kernel(h); }));

        // Imported NTT and IFFT service kernels are deliberately persistent;
        // their events are never collected into the waited vector.
        q.submit([&](handler& h) { NTTKernelA<0> kernel(mod_params[0].modulus_selector, mod_params[0].save_ntt_s); kernel(h); });
        q.submit([&](handler& h) { NTTKernelA<1> kernel(mod_params[1].modulus_selector, mod_params[1].save_ntt_s); kernel(h); });
        q.submit([&](handler& h) { NTTKernelA<2> kernel(mod_params[2].modulus_selector, mod_params[2].save_ntt_s); kernel(h); });
        q.submit([&](handler& h) { NTTKernelA<3> kernel(mod_params[3].modulus_selector, mod_params[3].save_ntt_s); kernel(h); });
        q.submit([&](handler& h) { NTTKernelA<4> kernel(mod_params[4].modulus_selector, mod_params[4].save_ntt_s); kernel(h); });
        q.submit([&](handler& h) { NTTKernelA<5> kernel(mod_params[5].modulus_selector, mod_params[5].save_ntt_s); kernel(h); });
        q.submit([&](handler& h) { NTTKernelB<0> kernel(mod_params[0].modulus_selector, mod_params[0].save_ntt_pte); kernel(h); });
        q.submit([&](handler& h) { NTTKernelB<1> kernel(mod_params[1].modulus_selector, mod_params[1].save_ntt_pte); kernel(h); });
        q.submit([&](handler& h) { NTTKernelB<2> kernel(mod_params[2].modulus_selector, mod_params[2].save_ntt_pte); kernel(h); });
        q.submit([&](handler& h) { NTTKernelB<3> kernel(mod_params[3].modulus_selector, mod_params[3].save_ntt_pte); kernel(h); });
        q.submit([&](handler& h) { NTTKernelB<4> kernel(mod_params[4].modulus_selector, mod_params[4].save_ntt_pte); kernel(h); });
        q.submit([&](handler& h) { NTTKernelB<5> kernel(mod_params[5].modulus_selector, mod_params[5].save_ntt_pte); kernel(h); });

        events.push_back(q.submit([&](handler& h) { ScaleAndReduceKernel<0> kernel(mod_params[0].scale, mod_params[0].mod_value, mod_params[0].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ScaleAndReduceKernel<1> kernel(mod_params[1].scale, mod_params[1].mod_value, mod_params[1].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ScaleAndReduceKernel<2> kernel(mod_params[2].scale, mod_params[2].mod_value, mod_params[2].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ScaleAndReduceKernel<3> kernel(mod_params[3].scale, mod_params[3].mod_value, mod_params[3].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ScaleAndReduceKernel<4> kernel(mod_params[4].scale, mod_params[4].mod_value, mod_params[4].const_ratio); kernel(h); }));
        events.push_back(q.submit([&](handler& h) { ScaleAndReduceKernel<5> kernel(mod_params[5].scale, mod_params[5].mod_value, mod_params[5].const_ratio); kernel(h); }));

        q.submit([&](handler& h) { IFFTKernel kernel; kernel(h); });
        events.push_back(q.submit([&](handler& h) { IFFTFanoutKernel<> kernel; kernel(h); }));
        events.push_back(q.submit([&](handler& h) { EntryKernel<> kernel(input_buf); kernel(h); }));

        return events;
    } catch (std::exception const& e) {
        std::cerr << "[SYCL_encrypt] Pipeline exception: " << e.what() << std::endl;
        std::exit(1);
    }
    return {};
}

extern "C" void SYCL_encrypt(
    size_t n,
    const double* scales,
    const uint32_t* mod_values,
    const uint32_t* const_ratios,
    const complex_double* encoding_buffer,
    const int8_t* error_samples,
    const uint32_t* const* secret_keys,
    const uint32_t* const* uniform_polys,
    uint32_t** c0_outputs,
    uint32_t** c1_outputs,
    uint32_t** s_save,
    uint32_t** c1_save,
    uint32_t** ntt_pte_outputs)
{
    if (n != POLY_N || n != SYCL_POLY_N) {
        std::cerr << "[SYCL_encrypt] this accelerator image requires polynomial degree "
                  << SYCL_POLY_N << "; received " << n << "\n";
        std::exit(1);
    }
    if (n % LANES != 0) {
        std::cerr << "[SYCL_encrypt] polynomial degree must be divisible by " << LANES << "\n";
        std::exit(1);
    }
    if (!scales || !mod_values || !const_ratios || !encoding_buffer || !error_samples ||
        !secret_keys || !uniform_polys || !c0_outputs) {
        std::cerr << "[SYCL_encrypt] required 8K accelerator argument is null\n";
        std::exit(1);
    }

    std::array<ModulusParams, NUM_MODULI> mod_params;
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        if (!secret_keys[p] || !uniform_polys[p] || !c0_outputs[p]) {
            std::cerr << "[SYCL_encrypt] required modulus buffer is null at index " << p << "\n";
            std::exit(1);
        }

        uint8_t selector = get_modulus_selector(mod_values[p]);
        if (selector == 0xff) {
            std::cerr << "[SYCL_encrypt] modulus " << mod_values[p]
                      << " is not supported by the six-channel 8K NTT RTL\n";
            std::exit(1);
        }

        uint32_t expected_cr0 = 0;
        uint32_t expected_cr1 = 0;
        if (!get_barrett_constants(mod_values[p], expected_cr0, expected_cr1) ||
            expected_cr0 != const_ratios[p * 2] || expected_cr1 != const_ratios[p * 2 + 1]) {
            std::cerr << "[SYCL_encrypt] Barrett constants do not match modulus index " << p << "\n";
            std::exit(1);
        }

        mod_params[p].scale = scales[p];
        mod_params[p].mod_value = mod_values[p];
        mod_params[p].const_ratio[0] = const_ratios[p * 2];
        mod_params[p].const_ratio[1] = const_ratios[p * 2 + 1];
        mod_params[p].modulus_selector = selector;
        mod_params[p].save_ntt_s = (s_save && s_save[p]);
        mod_params[p].save_ntt_pte = (ntt_pte_outputs && ntt_pte_outputs[p]);

        if (c1_save && c1_save[p]) {
            std::memcpy(c1_save[p], uniform_polys[p], n * sizeof(uint32_t));
        }
    }

    size_t num_blocks = n / LANES;
    std::vector<PipelineInputBlock> input_blocks;
    pack_input(n, encoding_buffer, error_samples, secret_keys, uniform_polys, input_blocks);

    std::array<std::vector<u32x4>, NUM_MODULI> c0_blocks;
    std::array<std::vector<u32x4>, NUM_MODULI> ntt_s_blocks;
    std::array<std::vector<u32x4>, NUM_MODULI> ntt_pte_blocks;
    for (size_t p = 0; p < NUM_MODULI; ++p) {
        c0_blocks[p].resize(num_blocks);
        ntt_s_blocks[p].resize(num_blocks);
        ntt_pte_blocks[p].resize(num_blocks);
    }

    {
        buffer<PipelineInputBlock, 1> input_buf(input_blocks.data(), range(num_blocks));
        std::array<buffer<u32x4, 1>, NUM_MODULI> c0_bufs = {
            buffer<u32x4, 1>(c0_blocks[0].data(), range(num_blocks)),
            buffer<u32x4, 1>(c0_blocks[1].data(), range(num_blocks)),
            buffer<u32x4, 1>(c0_blocks[2].data(), range(num_blocks)),
            buffer<u32x4, 1>(c0_blocks[3].data(), range(num_blocks)),
            buffer<u32x4, 1>(c0_blocks[4].data(), range(num_blocks)),
            buffer<u32x4, 1>(c0_blocks[5].data(), range(num_blocks))
        };
        std::array<buffer<u32x4, 1>, NUM_MODULI> ntt_s_bufs = {
            buffer<u32x4, 1>(ntt_s_blocks[0].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_s_blocks[1].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_s_blocks[2].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_s_blocks[3].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_s_blocks[4].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_s_blocks[5].data(), range(num_blocks))
        };
        std::array<buffer<u32x4, 1>, NUM_MODULI> ntt_pte_bufs = {
            buffer<u32x4, 1>(ntt_pte_blocks[0].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_pte_blocks[1].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_pte_blocks[2].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_pte_blocks[3].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_pte_blocks[4].data(), range(num_blocks)),
            buffer<u32x4, 1>(ntt_pte_blocks[5].data(), range(num_blocks))
        };

#if FPGA_SIMULATOR
        auto selector = ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
        auto selector = ext::intel::fpga_selector_v;
#else
        auto selector = ext::intel::fpga_emulator_selector_v;
#endif
        queue q{selector, property::queue::enable_profiling()};
        auto events = run_pipeline(q, input_buf, c0_bufs, ntt_s_bufs, ntt_pte_bufs, mod_params);
        for (auto& ev : events) ev.wait_and_throw();
    }

    for (size_t p = 0; p < NUM_MODULI; ++p) {
        unpack_u32_blocks(n, c0_blocks[p], c0_outputs[p]);
        if (mod_params[p].save_ntt_s) {
            unpack_u32_blocks(n, ntt_s_blocks[p], s_save[p]);
        }
        if (mod_params[p].save_ntt_pte) {
            unpack_u32_blocks(n, ntt_pte_blocks[p], ntt_pte_outputs[p]);
        }
        if (c1_outputs && c1_outputs[p]) {
            std::memcpy(c1_outputs[p], uniform_polys[p], n * sizeof(uint32_t));
        }
    }
}
