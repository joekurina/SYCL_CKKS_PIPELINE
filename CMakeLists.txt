cmake_minimum_required(VERSION 3.20)
project(sycl_ckks_accelerator VERSION 0.1.0 LANGUAGES CXX)

option(SYCL_CKKS_FPGA_HARDWARE "Build for FPGA hardware instead of emulator" OFF)
set(SYCL_CKKS_FPGA_DEVICE "Agilex7" CACHE STRING "FPGA target device for hardware/report builds")

add_library(sycl_ckks_accelerator STATIC
    src/SYCL_ckks_sym.cpp
)

add_library(sycl_ckks_accelerator::sycl_ckks_accelerator ALIAS sycl_ckks_accelerator)

target_compile_features(sycl_ckks_accelerator PUBLIC cxx_std_17)

target_include_directories(sycl_ckks_accelerator
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include_internal
        ${CMAKE_CURRENT_SOURCE_DIR}/rtl
)

target_compile_options(sycl_ckks_accelerator PRIVATE
    -fsycl
    -fintelfpga
    -Rno-debug-disables-optimization
    -DCSL_USE_GMP
    -DCSL_SYCL
    -Wno-return-type-c-linkage
    -qactypes
)

if(SYCL_CKKS_FPGA_HARDWARE)
    target_compile_definitions(sycl_ckks_accelerator PRIVATE FPGA_HARDWARE=1)
else()
    target_compile_definitions(sycl_ckks_accelerator PRIVATE FPGA_EMULATOR=1)
endif()

# The RTL archives are consumed by downstream executables that link this static
# library. They intentionally remain separate link inputs rather than being
# repacked into libsycl_ckks_accelerator.a.
target_link_libraries(sycl_ckks_accelerator
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/rtl/the_nwc_4k_ntt.a
        ${CMAKE_CURRENT_SOURCE_DIR}/rtl/fhe_ifft_4k_4lanes_double_253.a
)

target_link_options(sycl_ckks_accelerator INTERFACE
    -fsycl
    -fintelfpga
    LINKER:--allow-multiple-definition
)

install(TARGETS sycl_ckks_accelerator
    EXPORT sycl_ckks_acceleratorTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
install(EXPORT sycl_ckks_acceleratorTargets
    NAMESPACE sycl_ckks_accelerator::
    DESTINATION lib/cmake/sycl_ckks_accelerator
)
