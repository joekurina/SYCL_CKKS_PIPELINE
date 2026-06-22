#pragma once

#include "SYCL_common.h"
#include "SYCL_pipes.h"
#include "SYCL_data_types.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sycl_ckks {

template <int P>
class ExitC0KernelTask;

template <int P>
class ExitNTTASKernelTask;

template <int P>
class ExitNTTBKernelTask;

template <int P>
class ExitC0Kernel {
private:
    mutable sycl::buffer<u32x4, 1> output_buf;

public:
    explicit ExitC0Kernel(sycl::buffer<u32x4, 1>& buf)
        : output_buf(buf) {}

    void operator()(sycl::handler& h) const {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);

        h.single_task<ExitC0KernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;

            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                output[blk] = Pipes::PolyAddToExitPipe::read();
            }
        });
    }
};

template <int P>
class ExitNTTASKernel {
private:
    mutable sycl::buffer<u32x4, 1> output_buf;

public:
    explicit ExitNTTASKernel(sycl::buffer<u32x4, 1>& buf)
        : output_buf(buf) {}

    void operator()(sycl::handler& h) const {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);

        h.single_task<ExitNTTASKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;

            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                output[blk] = Pipes::NTTAToExitPipe::read();
            }
        });
    }
};

template <int P>
class ExitNTTBKernel {
private:
    mutable sycl::buffer<u32x4, 1> output_buf;

public:
    explicit ExitNTTBKernel(sycl::buffer<u32x4, 1>& buf)
        : output_buf(buf) {}

    void operator()(sycl::handler& h) const {
        auto output = output_buf.template get_access<sycl::access::mode::write>(h);

        h.single_task<ExitNTTBKernelTask<P>>([=]() [[intel::kernel_args_restrict]] {
            using Pipes = PipeSet<P>;

            for (size_t blk = 0; blk < NUM_BLOCKS; ++blk) {
                output[blk] = Pipes::NTTBToExitPipe::read();
            }
        });
    }
};

}
