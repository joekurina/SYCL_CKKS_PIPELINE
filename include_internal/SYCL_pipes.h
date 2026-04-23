#pragma once

#include "SYCL_data_types.h"
#include "SYCL_common.h"
#include "pipe_utils.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

// ============================================================================
// Pipe depth rationale:
//
// PIPE_DEPTH_BUFFERED (1024): Used when producer runs far ahead of consumer.
//   The IFFT kernel is a batch operation (read all -> compute -> write all),
//   creating a latency gap. Pipes feeding kernels downstream of the IFFT that
//   are written by Entry must buffer a full polynomial while the IFFT computes.
//
// PIPE_DEPTH_STREAMING (64): Used when producer and consumer both run at II=1
//   in lock-step. The compiler may increase this for stall-freedom or
//   clock-domain crossing overhead.
//
// Pipes requiring BUFFERED depth:
//   - ErrorToScaleReducePipes: Entry writes all blocks before ScaleAndReduce
//     can start (blocked on IFFT output)
//   - EntryToPolyMultNegPipe: Entry writes all c1 blocks before PolyMultNegAdd
//     can consume (blocked on NTT outputs)
//   - NTTAToPolyMultNegPipe: NTTA finishes all output while NTTB hasn't started
//     (NTTB waits for IFFT -> ScaleAndReduce -> RTL NTT chain)
// ============================================================================

// --- Shared pipes (Entry -> IFFT) ---
struct SharedToIFFTPipeId {};
using SharedToIFFTPipe = sycl::ext::intel::pipe<SharedToIFFTPipeId, encoding_block, PIPE_DEPTH_STREAMING>;

// --- IFFT output fanout (IFFT -> ScaleAndReduce x3) ---
struct IFFTToScaleReducePipeArrayId {};
using IFFTToScaleReducePipes = fpga_tools::PipeArray<
    IFFTToScaleReducePipeArrayId,
    encoding_block,
    PIPE_DEPTH_STREAMING,
    NUM_MODULI
>;

// --- Error fanout (Entry -> ScaleAndReduce x3) --- BUFFERED: blocked on IFFT
struct ErrorToScaleReducePipeArrayId {};
using ErrorToScaleReducePipes = fpga_tools::PipeArray<
    ErrorToScaleReducePipeArrayId,
    i8x4,
    PIPE_DEPTH_BUFFERED,
    NUM_MODULI
>;

template <int P>
struct PipeSet
{
    struct EntryToNTTAPipeID {};
    struct EntryToPolyMultNegPipeID {};
    struct ScaleReduceToNTTBPipeID {};
    struct NTTAToPolyMultNegPipeID {};
    struct NTTBToPolyAddPipeID {};
    struct NTTAToExitPipeID {};
    struct NTTBToExitPipeID {};
    struct PolyAddToExitPipeID {};
    struct NTTAInputPipeID {};
    struct NTTAModSelectorPipeID {};
    struct NTTAOutputPipeID {};
    struct NTTBInputPipeID {};
    struct NTTBModSelectorPipeID {};
    struct NTTBOutputPipeID {};

    // STREAMING: Entry and NTTKernelA both consume at II=1
    using EntryToNTTAPipe = sycl::ext::intel::pipe<EntryToNTTAPipeID, u32x4, PIPE_DEPTH_STREAMING>;
    // BUFFERED: Entry writes all c1 before PolyMultNegAdd can consume
    using EntryToPolyMultNegPipe = sycl::ext::intel::pipe<EntryToPolyMultNegPipeID, u32x4, PIPE_DEPTH_BUFFERED>;

    // STREAMING: ScaleAndReduce and NTTKernelB both at II=1
    using ScaleReduceToNTTBPipe = sycl::ext::intel::pipe<ScaleReduceToNTTBPipeID, u32x4, PIPE_DEPTH_STREAMING>;

    // BUFFERED: NTTA finishes before NTTB (NTTA gets data from Entry, NTTB waits for IFFT chain)
    using NTTAToPolyMultNegPipe = sycl::ext::intel::pipe<NTTAToPolyMultNegPipeID, u32x4, PIPE_DEPTH_BUFFERED>;
    // STREAMING: NTTB is the late arrival, PolyMultNegAdd waits on it
    using NTTBToPolyAddPipe = sycl::ext::intel::pipe<NTTBToPolyAddPipeID, u32x4, PIPE_DEPTH_STREAMING>;

    // STREAMING: lock-step with ExitKernel
    using NTTAToExitPipe = sycl::ext::intel::pipe<NTTAToExitPipeID, u32x4, PIPE_DEPTH_STREAMING>;
    using NTTBToExitPipe = sycl::ext::intel::pipe<NTTBToExitPipeID, u32x4, PIPE_DEPTH_STREAMING>;
    using PolyAddToExitPipe = sycl::ext::intel::pipe<PolyAddToExitPipeID, u32x4, PIPE_DEPTH_STREAMING>;

    struct NTTRTLInputData {
        int32_t x0;
        int32_t x1;
        int32_t x2;
        int32_t x3;
    };

    struct NTTRTLOutputData {
        int32_t q0;
        int32_t q1;
        int32_t q2;
        int32_t q3;
    };

    // STREAMING: RTL NTT interface, lock-step with wrapper kernel
    using NTTAInputPipe = sycl::ext::intel::pipe<NTTAInputPipeID, NTTRTLInputData, PIPE_DEPTH_STREAMING>;
    using NTTAModSelectorPipe = sycl::ext::intel::pipe<NTTAModSelectorPipeID, uint8_t, PIPE_DEPTH_STREAMING>;
    using NTTAOutputPipe = sycl::ext::intel::pipe<NTTAOutputPipeID, NTTRTLOutputData, PIPE_DEPTH_STREAMING>;
    using NTTBInputPipe = sycl::ext::intel::pipe<NTTBInputPipeID, NTTRTLInputData, PIPE_DEPTH_STREAMING>;
    using NTTBModSelectorPipe = sycl::ext::intel::pipe<NTTBModSelectorPipeID, uint8_t, PIPE_DEPTH_STREAMING>;
    using NTTBOutputPipe = sycl::ext::intel::pipe<NTTBOutputPipeID, NTTRTLOutputData, PIPE_DEPTH_STREAMING>;
};

}
