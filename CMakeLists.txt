cmake_minimum_required(VERSION 3.20)
project(sycl_ckks_accelerator VERSION 0.2.0 LANGUAGES CXX)

option(SYCL_CKKS_FPGA_HARDWARE "Build for FPGA hardware instead of emulator" OFF)
option(SYCL_CKKS_FPGA_SIMULATION "Build for FPGA simulator instead of emulator/hardware" OFF)
option(SYCL_CKKS_BUILD_KERNEL_HARNESSES "Build individual FPGA Test harness executables for internal SYCL kernels" OFF)
set(SYCL_CKKS_SIMULATION_WAVEFORM_DEPTH "" CACHE STRING "Optional hierarchy depth for simulator -Xsghdl. Empty uses Intel oneAPI sample default -Xsghdl; 0 logs all hierarchy.")
set(SYCL_CKKS_FPGA_DEVICE "Agilex7" CACHE STRING "FPGA target device for hardware/report/simulation builds")

set(SYCL_CKKS_QUARTUS_ROOT "/opt/intelFPGA_pro/23.1/quartus" CACHE PATH "Quartus Pro root used for FPGA simulator/hardware flows")
set(SYCL_CKKS_QUESTA_ROOT "/opt/intelFPGA_pro/23.1/questa_fe" CACHE PATH "Questa Intel FPGA Edition root used for FPGA simulator flows")

if(SYCL_CKKS_FPGA_HARDWARE AND SYCL_CKKS_FPGA_SIMULATION)
    message(FATAL_ERROR "Choose only one FPGA backend: SYCL_CKKS_FPGA_HARDWARE or SYCL_CKKS_FPGA_SIMULATION")
endif()

if(SYCL_CKKS_FPGA_HARDWARE OR SYCL_CKKS_FPGA_SIMULATION)
    find_program(SYCL_CKKS_QUARTUS_SH_EXECUTABLE
        NAMES quartus_sh
        HINTS "${SYCL_CKKS_QUARTUS_ROOT}/bin"
        NO_DEFAULT_PATH
    )
    if(NOT SYCL_CKKS_QUARTUS_SH_EXECUTABLE)
        message(FATAL_ERROR "FPGA builds require quartus_sh under SYCL_CKKS_QUARTUS_ROOT=${SYCL_CKKS_QUARTUS_ROOT}")
    endif()
    message(STATUS "SYCL_CKKS Quartus Pro root: ${SYCL_CKKS_QUARTUS_ROOT}")
    message(STATUS "SYCL_CKKS quartus_sh: ${SYCL_CKKS_QUARTUS_SH_EXECUTABLE}")
endif()

if(SYCL_CKKS_FPGA_SIMULATION)
    find_program(SYCL_CKKS_VSIM_EXECUTABLE
        NAMES vsim
        HINTS
            "${SYCL_CKKS_QUESTA_ROOT}/linux_x86_64"
            "${SYCL_CKKS_QUESTA_ROOT}/bin"
        NO_DEFAULT_PATH
    )
    if(NOT SYCL_CKKS_VSIM_EXECUTABLE)
        message(FATAL_ERROR "SYCL_CKKS_FPGA_SIMULATION requires vsim under SYCL_CKKS_QUESTA_ROOT=${SYCL_CKKS_QUESTA_ROOT}")
    endif()
    foreach(required_simlib
            "${SYCL_CKKS_QUESTA_ROOT}/intel/vhdl/tennm"
            "${SYCL_CKKS_QUESTA_ROOT}/intel/vhdl/tennm_hssi_all"
            "${SYCL_CKKS_QUESTA_ROOT}/intel/verilog/tennm"
            "${SYCL_CKKS_QUESTA_ROOT}/intel/verilog/tennm_hssi_all")
        if(NOT EXISTS "${required_simlib}")
            message(FATAL_ERROR "SYCL_CKKS_FPGA_SIMULATION requires Agilex simulation library: ${required_simlib}")
        endif()
    endforeach()
    message(STATUS "SYCL_CKKS simulator Questa root: ${SYCL_CKKS_QUESTA_ROOT}")
    message(STATUS "SYCL_CKKS simulator vsim: ${SYCL_CKKS_VSIM_EXECUTABLE}")
endif()

# Compatibility alias so workstation invocations using -DFPGA_DEVICE=... are
# consumed by this project instead of being reported as unused by CMake.
if(DEFINED FPGA_DEVICE AND NOT FPGA_DEVICE STREQUAL "")
    set(SYCL_CKKS_FPGA_DEVICE "${FPGA_DEVICE}" CACHE STRING "FPGA target device for hardware/report/simulation builds" FORCE)
endif()

set(SYCL_CKKS_FPGA_BACKEND_FLAGS)
set(SYCL_CKKS_DEVICE_COMPILE_DEFINITION FPGA_EMULATOR=1)
set(SYCL_CKKS_DEVICE_DEFINE_FLAG -DFPGA_EMULATOR)
set(SYCL_CKKS_MERGED_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/libsycl_ckks_accelerator_merged.a)
set(SYCL_CKKS_MRI_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/merge_sycl_ckks_accelerator.mri)

if(SYCL_CKKS_FPGA_HARDWARE)
    message(STATUS "SYCL_CKKS hardware mode enabled; FPGA target: ${SYCL_CKKS_FPGA_DEVICE}")
    set(SYCL_CKKS_DEVICE_COMPILE_DEFINITION FPGA_HARDWARE=1)
    set(SYCL_CKKS_DEVICE_DEFINE_FLAG -DFPGA_HARDWARE)
    list(APPEND SYCL_CKKS_FPGA_BACKEND_FLAGS
        -Xshardware
        -Xstarget=${SYCL_CKKS_FPGA_DEVICE}
    )
elseif(SYCL_CKKS_FPGA_SIMULATION)
    message(STATUS "SYCL_CKKS simulation mode enabled; FPGA target: ${SYCL_CKKS_FPGA_DEVICE}")
    set(SYCL_CKKS_DEVICE_COMPILE_DEFINITION FPGA_SIMULATOR=1)
    set(SYCL_CKKS_DEVICE_DEFINE_FLAG -DFPGA_SIMULATOR)
    set(SYCL_CKKS_MERGED_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/sycl_ckks_accelerator_sim.a)
    list(APPEND SYCL_CKKS_FPGA_BACKEND_FLAGS
        -Xssimulation
    )
    if(SYCL_CKKS_SIMULATION_WAVEFORM_DEPTH STREQUAL "")
        list(APPEND SYCL_CKKS_FPGA_BACKEND_FLAGS -Xsghdl)
    else()
        list(APPEND SYCL_CKKS_FPGA_BACKEND_FLAGS -Xsghdl=${SYCL_CKKS_SIMULATION_WAVEFORM_DEPTH})
    endif()
    list(APPEND SYCL_CKKS_FPGA_BACKEND_FLAGS
        -Xstarget=${SYCL_CKKS_FPGA_DEVICE}
    )
else()
    message(STATUS "SYCL_CKKS emulator mode enabled")
endif()

add_library(sycl_ckks_accelerator_obj OBJECT
    src/SYCL_ckks_sym.cpp
)

add_library(sycl_ckks_accelerator STATIC
    $<TARGET_OBJECTS:sycl_ckks_accelerator_obj>
)

add_library(sycl_ckks_accelerator::sycl_ckks_accelerator ALIAS sycl_ckks_accelerator)

target_compile_features(sycl_ckks_accelerator_obj PUBLIC cxx_std_17)
target_compile_features(sycl_ckks_accelerator PUBLIC cxx_std_17)

target_include_directories(sycl_ckks_accelerator_obj
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include_internal
        ${CMAKE_CURRENT_SOURCE_DIR}/rtl
)

target_include_directories(sycl_ckks_accelerator
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(sycl_ckks_accelerator_obj PRIVATE
    -fsycl
    -fintelfpga
    -Rno-debug-disables-optimization
    -DCSL_USE_GMP
    -DCSL_SYCL
    -Wno-return-type-c-linkage
    -qactypes
)
target_compile_definitions(sycl_ckks_accelerator_obj PRIVATE ${SYCL_CKKS_DEVICE_COMPILE_DEFINITION})

set(SYCL_CKKS_NTT_RTL_ARCHIVE ${CMAKE_CURRENT_SOURCE_DIR}/rtl/the_nwc_8k_ntt.a)
set(SYCL_CKKS_IFFT_RTL_ARCHIVE ${CMAKE_CURRENT_SOURCE_DIR}/rtl/fhe_ifft_8k_4lanes_double_261.a)

if(SYCL_CKKS_FPGA_SIMULATION)
    # Simulator image archive. This uses -fsycl-link=image because the
    # accelerator has no main(), while downstream SEAL-Embedded needs a
    # reusable FPGA simulator image/archive. Runtime execution must enable the
    # simulator device with CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 or
    # INTELFPGA_SIM_DEVICE_SPEC_DIR=<project.prj>.
    add_custom_command(
        OUTPUT ${SYCL_CKKS_MERGED_ARCHIVE}
        COMMAND ${CMAKE_COMMAND} -E rm -f ${SYCL_CKKS_MERGED_ARCHIVE}
        COMMAND ${CMAKE_COMMAND} -E env
                "PATH=${SYCL_CKKS_QUESTA_ROOT}/linux_x86_64:${SYCL_CKKS_QUESTA_ROOT}/bin:$ENV{PATH}"
                ${CMAKE_CXX_COMPILER}
                -v
                -fsycl
                -std=c++17
                -Rno-debug-disables-optimization
                -DCSL_USE_GMP
                -DCSL_SYCL
                -Wno-return-type-c-linkage
                -qactypes
                -fintelfpga
                ${SYCL_CKKS_FPGA_BACKEND_FLAGS}
                ${SYCL_CKKS_DEVICE_DEFINE_FLAG}
                -fsycl-link=image
                -I${CMAKE_CURRENT_SOURCE_DIR}
                -I${CMAKE_CURRENT_SOURCE_DIR}/include
                -I${CMAKE_CURRENT_SOURCE_DIR}/include_internal
                -I${CMAKE_CURRENT_SOURCE_DIR}/rtl
                -o ${SYCL_CKKS_MERGED_ARCHIVE}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/SYCL_ckks_sym.cpp
                ${SYCL_CKKS_NTT_RTL_ARCHIVE}
                ${SYCL_CKKS_IFFT_RTL_ARCHIVE}
        DEPENDS
                ${CMAKE_CURRENT_SOURCE_DIR}/src/SYCL_ckks_sym.cpp
                ${SYCL_CKKS_NTT_RTL_ARCHIVE}
                ${SYCL_CKKS_IFFT_RTL_ARCHIVE}
        COMMENT "Running Intel FPGA simulation image compile for reusable archive ${SYCL_CKKS_MERGED_ARCHIVE}"
        VERBATIM
    )
else()
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
endif()

add_custom_target(sycl_ckks_accelerator_merged ALL
    DEPENDS ${SYCL_CKKS_MERGED_ARCHIVE}
)

if(SYCL_CKKS_FPGA_HARDWARE)
    add_custom_target(sycl_ckks_accelerator_report
        COMMAND ${CMAKE_CXX_COMPILER}
                -fsycl
                -fintelfpga
                -Xshardware
                -Xstarget=${SYCL_CKKS_FPGA_DEVICE}
                -fsycl-link=early
                -o ${CMAKE_CURRENT_BINARY_DIR}/sycl_ckks_accelerator_report.a
                $<TARGET_OBJECTS:sycl_ckks_accelerator_obj>
                ${SYCL_CKKS_NTT_RTL_ARCHIVE}
                ${SYCL_CKKS_IFFT_RTL_ARCHIVE}
        DEPENDS sycl_ckks_accelerator_obj ${SYCL_CKKS_NTT_RTL_ARCHIVE} ${SYCL_CKKS_IFFT_RTL_ARCHIVE}
        COMMENT "Running Intel FPGA early device link for board ${SYCL_CKKS_FPGA_DEVICE}"
        VERBATIM
    )
elseif(SYCL_CKKS_FPGA_SIMULATION)
    add_custom_target(sycl_ckks_accelerator_sim_archive
        DEPENDS ${SYCL_CKKS_MERGED_ARCHIVE}
        COMMENT "Simulation accelerator archive ready for SEAL-Embedded: ${SYCL_CKKS_MERGED_ARCHIVE}"
    )
endif()

if(SYCL_CKKS_BUILD_KERNEL_HARNESSES)
    if(NOT SYCL_CKKS_FPGA_SIMULATION)
        message(FATAL_ERROR "SYCL_CKKS_BUILD_KERNEL_HARNESSES requires SYCL_CKKS_FPGA_SIMULATION=ON so harnesses use the simulator path")
    endif()

    function(add_sycl_ckks_kernel_harness target source)
        add_executable(${target} ${source})
        target_compile_features(${target} PRIVATE cxx_std_17)
        target_include_directories(${target}
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/include_internal
                ${CMAKE_CURRENT_SOURCE_DIR}/rtl
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/kernel_harnesses
        )
        target_compile_options(${target}
            PRIVATE
                -fsycl
                -fintelfpga
                -Rno-debug-disables-optimization
                -DCSL_USE_GMP
                -DCSL_SYCL
                -Wno-return-type-c-linkage
                -qactypes
        )
        target_compile_definitions(${target} PRIVATE ${SYCL_CKKS_DEVICE_COMPILE_DEFINITION})
        set_property(TARGET ${target} PROPERTY RULE_LAUNCH_LINK
            "${CMAKE_COMMAND} -E env PATH=${SYCL_CKKS_QUESTA_ROOT}/linux_x86_64:${SYCL_CKKS_QUESTA_ROOT}/bin:$ENV{PATH}")
        target_link_options(${target}
            PRIVATE
                -fsycl
                -std=c++17
                -Rno-debug-disables-optimization
                -DCSL_USE_GMP
                -DCSL_SYCL
                -Wno-return-type-c-linkage
                -qactypes
                -fintelfpga
                ${SYCL_CKKS_FPGA_BACKEND_FLAGS}
                ${SYCL_CKKS_DEVICE_DEFINE_FLAG}
                LINKER:--allow-multiple-definition
        )
        foreach(rtl_archive IN LISTS ARGN)
            target_link_libraries(${target} PRIVATE ${rtl_archive})
        endforeach()
    endfunction()

    add_sycl_ckks_kernel_harness(sycl_ckks_entry_kernel_fpga_test tests/kernel_harnesses/entry_kernel_fpga_test.cpp)
    add_sycl_ckks_kernel_harness(sycl_ckks_ifft_kernel_fpga_test tests/kernel_harnesses/ifft_kernel_fpga_test.cpp ${SYCL_CKKS_IFFT_RTL_ARCHIVE})
    add_sycl_ckks_kernel_harness(sycl_ckks_scale_and_reduce_kernel_fpga_test tests/kernel_harnesses/scale_and_reduce_kernel_fpga_test.cpp)
    add_sycl_ckks_kernel_harness(sycl_ckks_ntt_a_kernel_fpga_test tests/kernel_harnesses/ntt_a_kernel_fpga_test.cpp ${SYCL_CKKS_NTT_RTL_ARCHIVE})
    add_sycl_ckks_kernel_harness(sycl_ckks_ntt_b_kernel_fpga_test tests/kernel_harnesses/ntt_b_kernel_fpga_test.cpp ${SYCL_CKKS_NTT_RTL_ARCHIVE})
    add_sycl_ckks_kernel_harness(sycl_ckks_poly_mult_neg_add_kernel_fpga_test tests/kernel_harnesses/poly_mult_neg_add_kernel_fpga_test.cpp)
    add_sycl_ckks_kernel_harness(sycl_ckks_exit_c0_kernel_fpga_test tests/kernel_harnesses/exit_c0_kernel_fpga_test.cpp)
    add_sycl_ckks_kernel_harness(sycl_ckks_exit_ntt_a_kernel_fpga_test tests/kernel_harnesses/exit_ntt_a_kernel_fpga_test.cpp)
    add_sycl_ckks_kernel_harness(sycl_ckks_exit_ntt_b_kernel_fpga_test tests/kernel_harnesses/exit_ntt_b_kernel_fpga_test.cpp)

    add_custom_target(sycl_ckks_kernel_harnesses
        DEPENDS
            sycl_ckks_entry_kernel_fpga_test
            sycl_ckks_ifft_kernel_fpga_test
            sycl_ckks_scale_and_reduce_kernel_fpga_test
            sycl_ckks_ntt_a_kernel_fpga_test
            sycl_ckks_ntt_b_kernel_fpga_test
            sycl_ckks_poly_mult_neg_add_kernel_fpga_test
            sycl_ckks_exit_c0_kernel_fpga_test
            sycl_ckks_exit_ntt_a_kernel_fpga_test
            sycl_ckks_exit_ntt_b_kernel_fpga_test
    )
endif()

# Downstream executables must still use the Intel SYCL/FPGA link path
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
    LINKER:--allow-multiple-definition
)

if(SYCL_CKKS_FPGA_HARDWARE OR SYCL_CKKS_FPGA_SIMULATION)
    target_link_options(sycl_ckks_accelerator INTERFACE
        ${SYCL_CKKS_FPGA_BACKEND_FLAGS}
        ${SYCL_CKKS_DEVICE_DEFINE_FLAG}
    )
else()
    target_link_options(sycl_ckks_accelerator INTERFACE
        -DFPGA_EMULATOR
        -O0
    )
endif()

install(FILES ${SYCL_CKKS_MERGED_ARCHIVE}
    DESTINATION lib
    RENAME libsycl_ckks_accelerator.a
)
install(DIRECTORY include/ DESTINATION include)
