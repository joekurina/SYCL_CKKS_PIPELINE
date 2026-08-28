#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

template <int P, size_t TotalBeats>
class ExitC0KernelTask;

template <int P, size_t TotalBeats>
class ExitNTTASKernelTask;

template <int P, size_t TotalBeats>
class ExitNTTBKernelTask;

template <int P, size_t TotalBeats = NUM_BLOCKS>
class ExitC0Kernel {
private:
    mutable sycl::buffer<u32x4, 1> output_buf;

public:
    explicit ExitC0Kernel(sycl::buffer<u32x4, 1>& buf)
        : output_buf(buf) {}

    void operator()(sycl::handler& h) const {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);

        h.single_task<ExitC0KernelTask<P, TotalBeats>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;
            for (size_t blk = 0; blk < TotalBeats; ++blk) {
                output[blk] = Pipes::PolyAddToExitPipe::read();
            }
        });
    }
};

template <int P, size_t TotalBeats = NUM_BLOCKS>
class ExitNTTASKernel {
private:
    mutable sycl::buffer<u32x4, 1> output_buf;

public:
    explicit ExitNTTASKernel(sycl::buffer<u32x4, 1>& buf)
        : output_buf(buf) {}

    void operator()(sycl::handler& h) const {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);

        h.single_task<ExitNTTASKernelTask<P, TotalBeats>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;
            for (size_t blk = 0; blk < TotalBeats; ++blk) {
                output[blk] = Pipes::NTTAToExitPipe::read();
            }
        });
    }
};

template <int P, size_t TotalBeats = NUM_BLOCKS>
class ExitNTTBKernel {
private:
    mutable sycl::buffer<u32x4, 1> output_buf;

public:
    explicit ExitNTTBKernel(sycl::buffer<u32x4, 1>& buf)
        : output_buf(buf) {}

    void operator()(sycl::handler& h) const {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);

        h.single_task<ExitNTTBKernelTask<P, TotalBeats>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;
            for (size_t blk = 0; blk < TotalBeats; ++blk) {
                output[blk] = Pipes::NTTBToExitPipe::read();
            }
        });
    }
};

}
