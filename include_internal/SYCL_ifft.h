#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include "rtl/fhe_ifft_8k_4lanes_double_261_sycl.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

static_assert(sizeof(fhe_ifft_8k_4lanes_double_261_input_t) == 65,
              "unexpected fhe_ifft_8k input ABI size");
static_assert(sizeof(fhe_ifft_8k_4lanes_double_261_output_t) == 65,
              "unexpected fhe_ifft_8k output ABI size");

class IFFTKernelTask;

class IFFTKernel {
public:
    IFFTKernel() {}

    void operator()(sycl::handler& h) const {
        h.single_task<IFFTKernelTask>([=]() [[intel::kernel_args_restrict]] {

#ifdef FPGA_EMULATOR
            ifft8k_base_DUT* rtl_instance = fhe_ifft_8k_4lanes_double_261_new_instance();
#endif

            // The imported RTL IFFT is a fixed-schedule streaming core that
            // cannot be paused once a frame begins. Each output beat is handed
            // off with a single write into IFFTRawOutputPipe (BUFFERED, depth
            // NUM_BLOCKS) so this loop never shares a per-iteration schedule
            // with the slower downstream 6-way fanout, which runs as a
            // separate, decoupled six-way kernel (IFFTFanoutKernel).
            //
            // The RTL core must be fed every single cycle with no exceptions,
            // so the input read is a non-blocking, unconditional read every
            // iteration (matching known-good RTL-component-import designs)
            // rather than a conditional blocking read.
            //
            // This loop is deliberately free-running (`while (1)`) and never
            // terminates on its own. An earlier version gated the loop on
            // `output_count < NUM_BLOCKS`, incrementing output_count only
            // after observing hw_out.port_v_out_s for *that same* iteration.
            // That created a loop-carried dependency: the compiler could not
            // launch iteration N+1's input transaction (assert ivalid again)
            // without first resolving iteration N's own output. But this RTL
            // does not produce beat N's output until an entire NUM_BLOCKS-beat
            // frame has been fed -- which can never happen if beat N+1 is
            // blocked waiting on beat N's result. That's a deadlock by
            // construction, independent of how well-behaved the RTL wrapper's
            // ivalid/iready/ovalid/oready handshake is. Decoupling loop
            // continuation from the RTL's own output (as here) is required
            // for the compiler to keep issuing new input transactions while
            // earlier ones are still in flight, matching the known-good
            // free-running/non-blocking pattern validated in
            // ifft_simple_kernel_fpga_test.cpp.
            //
            // Because this kernel never completes, the host must never wait
            // on its own completion event -- only on the bounded downstream
            // kernels (IFFTFanoutKernel, drain/exit kernels).
            while (1) {
                bool input_valid = false;
                encoding_block block = SharedToIFFTPipe::read(input_valid);

                fhe_ifft_8k_4lanes_double_261_input_t hw_in;
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
                fhe_ifft_8k_4lanes_double_261_output_t hw_out = fhe_ifft_8k_4lanes_double_261(rtl_instance, hw_in);
#else
                fhe_ifft_8k_4lanes_double_261_output_t hw_out = fhe_ifft_8k_4lanes_double_261(hw_in);
#endif

                if (hw_out.port_v_out_s == 1) {
                    encoding_block output_block;
                    output_block.element0 = complex_double(hw_out.port_data_out_0re, hw_out.port_data_out_0im);
                    output_block.element1 = complex_double(hw_out.port_data_out_1re, hw_out.port_data_out_1im);
                    output_block.element2 = complex_double(hw_out.port_data_out_2re, hw_out.port_data_out_2im);
                    output_block.element3 = complex_double(hw_out.port_data_out_3re, hw_out.port_data_out_3im);

                    // Single write per RTL output beat, into a dedicated BUFFERED
                    // pipe. The RTL core cannot be paused once started, so this
                    // loop must be able to run start-to-finish at whatever pace
                    // the DUT delivers, with no other work (like the downstream
                    // 6-way fanout) sharing its per-iteration schedule.
                    IFFTRawOutputPipe::write(output_block);
                }
            }
            // Unreachable: the loop above never exits by design (see comment
            // above it). In FPGA_EMULATOR mode this means rtl_instance is
            // intentionally never explicitly deleted; it's reclaimed when the
            // emulator process exits.
        });
    }
};

template <size_t TotalBeats>
class IFFTFanoutKernelTask;

// Decoupled from IFFTKernel on purpose: reads the raw RTL output at its own
// pace and fans it out to the six per-modulus ScaleAndReduce pipes. See the
// comment on IFFTRawOutputPipe in SYCL_pipes.h for why this is a separate
// kernel rather than folded into IFFTKernel's loop.
//
// Templated on the total beat count (defaulting to a single NUM_BLOCKS
// frame, matching every existing call site's behavior unchanged) rather than
// hardcoding NUM_BLOCKS, so a multi-frame consumer (e.g. a back-to-back
// stress-test harness) can instantiate IFFTFanoutKernel<TotalBeats> as its
// own distinct kernel-name specialization instead of hand-rolling a second
// kernel that reads the same IFFTRawOutputPipe -- the FPGA device compiler
// rejects two different kernel-task definitions touching the same pipe
// identity within one compiled translation unit, even if only one of them is
// ever actually submitted to a queue.
template <size_t TotalBeats = NUM_BLOCKS>
class IFFTFanoutKernel {
public:
    IFFTFanoutKernel() {}

    void operator()(sycl::handler& h) const {
        h.single_task<IFFTFanoutKernelTask<TotalBeats>>([=]() [[intel::kernel_args_restrict]] {
            for (size_t i = 0; i < TotalBeats; ++i) {
                encoding_block output_block = IFFTRawOutputPipe::read();
                IFFTToScaleReducePipes::write(output_block);
            }
        });
    }
};

}
