#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdlib>
#include <iostream>

namespace sycl_ckks::test {

inline void kernel_test_exception_handler(sycl::exception_list exceptions)
{
    for (const std::exception_ptr& eptr : exceptions) {
        try {
            std::rethrow_exception(eptr);
        } catch (const sycl::exception& e) {
            std::cerr << "Caught asynchronous SYCL exception:\n" << e.what() << "\n";
            std::exit(1);
        }
    }
}

inline sycl::queue make_queue()
{
#if FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
    return sycl::queue(selector, kernel_test_exception_handler);
}

}  // namespace sycl_ckks::test
