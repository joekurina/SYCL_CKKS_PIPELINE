#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include "rtl/the_nwc_4k_ntt_sycl.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

// TEMPORARY DIAGNOSTIC SWITCH -- set to 1 to bypass the imported external RTL
// library call (the_nwc_4k_ntt) entirely and replace it with a trivial
// pass-through. This isolates whether the "input data goes to X forever
// after ~32 real beats" symptom is caused by the compiler's marshaling of
// repeated calls to the external RTL component, or by something in this
// kernel's loop/pipe usage itself. Set back to 0 to restore real NTT
// behavior once the root cause is found.
#define NTT_DIAGNOSTIC_BYPASS_RTL 0

struct NTT_A_Tag {};
struct NTT_B_Tag {};

template <int P, typename Tag>
struct NTTPipeTraits;

template <int P>
struct NTTPipeTraits<P, NTT_A_Tag> {
    using InputPipe = typename PipeSet<P>::EntryToNTTAPipe;
    using RTLInputPipe = typename PipeSet<P>::NTTAInputPipe;
    using RTLModSelectorPipe = typename PipeSet<P>::NTTAModSelectorPipe;
    using RTLOutputPipe = typename PipeSet<P>::NTTAOutputPipe;
    using DownstreamPipe = typename PipeSet<P>::NTTAToPolyMultNegPipe;
    using ExitPipe = typename PipeSet<P>::NTTAToExitPipe;
};

template <int P>
struct NTTPipeTraits<P, NTT_B_Tag> {
    using InputPipe = typename PipeSet<P>::ScaleReduceToNTTBPipe;
    using RTLInputPipe = typename PipeSet<P>::NTTBInputPipe;
    using RTLModSelectorPipe = typename PipeSet<P>::NTTBModSelectorPipe;
    using RTLOutputPipe = typename PipeSet<P>::NTTBOutputPipe;
    using DownstreamPipe = typename PipeSet<P>::NTTBToPolyAddPipe;
    using ExitPipe = typename PipeSet<P>::NTTBToExitPipe;
};

template <int P, typename Tag>
class NTTKernelTask;

template <int P, typename Tag>
class NTTKernel {
private:
    uint8_t modulus_selector;
    bool write_to_exit;

public:
    NTTKernel(uint8_t mod_sel, bool save_output = false)
        : modulus_selector(mod_sel), write_to_exit(save_output) {}

    void operator()(sycl::handler& h) const {
        uint8_t kernel_mod_sel = modulus_selector;
        bool kernel_write_exit = write_to_exit;

        h.single_task<NTTKernelTask<P, Tag>>([=]() [[intel::kernel_args_restrict]] {
            using Traits = NTTPipeTraits<P, Tag>;

#if !NTT_DIAGNOSTIC_BYPASS_RTL
#ifdef FPGA_EMULATOR
            reg_test_verifyNTT_multi_DUT* rtl_instance = the_nwc_4k_ntt_new_instance();
#endif
#endif

            // The imported RTL NTT is a frame-oriented block, just like the IFFT
            // core in SYCL_ifft.h: it must absorb an entire NUM_BLOCKS-beat input
            // frame before it produces any valid output at all, then bursts the
            // full frame of output back out. This kernel mirrors IFFTKernel in
            // being free-running (`while (1)`, never terminates, never waited on
            // by the host -- see IFFTKernel's comment for that part of the
            // story), but deliberately does NOT mirror the non-blocking,
            // unconditional-call-every-iteration pattern IFFTKernel uses:
            //
            //   - An earlier version here used a non-blocking read
            //     (Traits::InputPipe::read(input_valid)) and called the RTL
            //     unconditionally every iteration, forwarding the pipe's
            //     success flag as port_in_v_s so the DUT would see a real
            //     beat vs. a bubble. That makes the compiler's generated call
            //     site assert ivalid every single iteration regardless of
            //     whether real data was available -- confirmed directly via
            //     RTL-level $display instrumentation, which showed ivalid
            //     pinned at 1 (with the payload showing X/garbage and
            //     port_in_v_s=0) straight through long stretches where the
            //     upstream feed pipe was genuinely empty.
            //   - The external-RTL-component call site (IS_STALL_FREE="no" in
            //     the_nwc_4k_ntt.xml) enforces CAPACITY as a hard cap on the
            //     number of *issued* calls (ivalid=1, real or not) that
            //     haven't yet retired (produced a corresponding ovalid). This
            //     DUT never retires anything until a full NUM_BLOCKS-beat
            //     frame has been absorbed. So every idle/bubble iteration
            //     silently burns one of CAPACITY's credits for nothing, and
            //     once genuinely-idle cycles + genuinely-real cycles exceeds
            //     CAPACITY before the first frame completes, ivalid
            //     permanently deadlocks -- confirmed empirically: it froze at
            //     issued-call #~2048, an exact match for CAPACITY=2048.
            //     Raising CAPACITY just moves this ceiling; the input feed's
            //     real-data duty cycle degrades over time in practice (bursty
            //     memory arbitration upstream), so no fixed CAPACITY is a
            //     robust fix -- the actual bug is spending credits on cycles
            //     with no real data at all.
            //   - The fix: block on the input read. A blocking read means the
            //     loop (and therefore the RTL call) only ever advances once
            //     genuine data is available, so every call the compiler issues
            //     corresponds to a real beat -- CAPACITY = 2 * NUM_BLOCKS then
            //     means exactly what it says, two full *real* frames in
            //     flight, regardless of how sparse or bursty upstream
            //     admission becomes. This does not resurrect the old
            //     loop-carried-dependency deadlock (see IFFTKernel's comment):
            //     that one was caused by bounding the *loop itself* on
            //     observing this same iteration's output; here the loop
            //     remains an unconditional `while (1)` and only the read
            //     statement blocks. Once the feeder finishes writing all
            //     NUM_BLOCKS real elements, this read blocks forever on the
            //     (NUM_BLOCKS+1)th iteration -- harmless, since this kernel is
            //     never waited on and by then every real beat has already
            //     been fed.
            while (1) {
                u32x4 input_block = Traits::InputPipe::read();

                the_nwc_4k_ntt_input_t hw_in;
                hw_in.port_in_v_s = 1;
                hw_in.port_in_c_s = kernel_mod_sel;
                hw_in.port_x_in_0 = static_cast<int32_t>(input_block.element0);
                hw_in.port_x_in_1 = static_cast<int32_t>(input_block.element1);
                hw_in.port_x_in_2 = static_cast<int32_t>(input_block.element2);
                hw_in.port_x_in_3 = static_cast<int32_t>(input_block.element3);

#if NTT_DIAGNOSTIC_BYPASS_RTL
                // Trivial pass-through: no external RTL call at all. Valid-in
                // simply forwards to valid-out one cycle later, and the data
                // is echoed unchanged. If this still shows the same
                // "X after ~32 beats" symptom, the bug is in this kernel's
                // loop/pipe usage rather than in the RTL call marshaling.
                the_nwc_4k_ntt_output_t hw_out;
                hw_out.port_out_v_s = hw_in.port_in_v_s;
                hw_out.port_out_c_s = hw_in.port_in_c_s;
                hw_out.port_out_q_0 = hw_in.port_x_in_0;
                hw_out.port_out_q_1 = hw_in.port_x_in_1;
                hw_out.port_out_q_2 = hw_in.port_x_in_2;
                hw_out.port_out_q_3 = hw_in.port_x_in_3;
#else
#ifdef FPGA_EMULATOR
                the_nwc_4k_ntt_output_t hw_out = the_nwc_4k_ntt(rtl_instance, hw_in);
#else
                the_nwc_4k_ntt_output_t hw_out = the_nwc_4k_ntt(hw_in);
#endif
#endif

                if (hw_out.port_out_v_s == 1) {
                    u32x4 output_block;
                    output_block.element0 = static_cast<uint32_t>(hw_out.port_out_q_0);
                    output_block.element1 = static_cast<uint32_t>(hw_out.port_out_q_1);
                    output_block.element2 = static_cast<uint32_t>(hw_out.port_out_q_2);
                    output_block.element3 = static_cast<uint32_t>(hw_out.port_out_q_3);

                    Traits::DownstreamPipe::write(output_block);
                    if (kernel_write_exit) {
                        Traits::ExitPipe::write(output_block);
                    }
                }
            }
            // Unreachable: the loop above never exits by design (see comment
            // above it). In FPGA_EMULATOR mode this means rtl_instance is
            // intentionally never explicitly deleted; it's reclaimed when the
            // emulator process exits.
        });
    }
};

template <int P>
using NTTKernelA = NTTKernel<P, NTT_A_Tag>;

template <int P>
using NTTKernelB = NTTKernel<P, NTT_B_Tag>;

}
