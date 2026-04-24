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

set(SYCL_CKKS_NTT_RTL_ARCHIVE ${CMAKE_CURRENT_SOURCE_DIR}/rtl/the_nwc_4k_ntt.a)
set(SYCL_CKKS_IFFT_RTL_ARCHIVE ${CMAKE_CURRENT_SOURCE_DIR}/rtl/fhe_ifft_4k_4lanes_double_253.a)
set(SYCL_CKKS_MERGED_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/libsycl_ckks_accelerator_merged.a)
set(SYCL_CKKS_MRI_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/merge_sycl_ckks_accelerator.mri)

file(WRITE ${SYCL_CKKS_MRI_SCRIPT}
    "CREATE ${SYCL_CKKS_MERGED_ARCHIVE}\n"
    "ADDLIB ${CMAKE_CURRENT_BINARY_DIR}/libsycl_ckks_accelerator.a\n"
    "ADDLIB ${SYCL_CKKS_NTT_RTL_ARCHIVE}\n"
    "ADDLIB ${SYCL_CKKS_IFFT_RTL_ARCHIVE}\n"
    "SAVE\n"
    "END\n")

add_custom_command(
    OUTPUT ${SYCL_CKKS_MERGED_ARCHIVE}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${SYCL_CKKS_MERGED_ARCHIVE}
    COMMAND /bin/sh -c "${CMAKE_AR} -M < ${SYCL_CKKS_MRI_SCRIPT}"
    COMMAND ${CMAKE_RANLIB} ${SYCL_CKKS_MERGED_ARCHIVE}
    DEPENDS sycl_ckks_accelerator ${SYCL_CKKS_NTT_RTL_ARCHIVE} ${SYCL_CKKS_IFFT_RTL_ARCHIVE} ${SYCL_CKKS_MRI_SCRIPT}
    COMMENT "Merging SYCL accelerator and active RTL archives into opaque static library"
    VERBATIM
)

add_custom_target(sycl_ckks_accelerator_merged ALL
    DEPENDS ${SYCL_CKKS_MERGED_ARCHIVE}
)

# Downstream executables must still use the Intel SYCL/FPGA emulator link path
# so icpx performs the final device-image link from this opaque static archive.
target_link_options(sycl_ckks_accelerator INTERFACE
    -fsycl
    -std=c++17
    -Rno-debug-disables-optimization
    -DCSL_USE_GMP
    -DCSL_SYCL
    -Wno-return-type-c-linkage
    -qactypes
    -fintelfpga
    -DFPGA_EMULATOR
    -O0
    LINKER:--allow-multiple-definition
)

install(FILES ${SYCL_CKKS_MERGED_ARCHIVE}
    DESTINATION lib
    RENAME libsycl_ckks_accelerator.a
)
install(DIRECTORY include/ DESTINATION include)
