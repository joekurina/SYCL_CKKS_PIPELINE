function(add_ckks_kernel_test NAME SOURCE)
    set(options)
    set(oneValueArgs CKKS_TEST_P CKKS_TEST_USES_NTT_RTL CKKS_TEST_USES_IFFT_RTL)
    set(multiValueArgs)
    cmake_parse_arguments(AKT "" "${oneValueArgs}" "" ${ARGN})

    set(TEST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE})
    set(BASE_COMPILE_OPTIONS
        -fsycl
        -fintelfpga
        -Rno-debug-disables-optimization
        -DCSL_USE_GMP
        -DCSL_SYCL
        -Wno-return-type-c-linkage
        -qactypes
    )
    set(BASE_LINK_OPTIONS
        -fsycl
        -fintelfpga
        -qactypes
        LINKER:--allow-multiple-definition
    )
    set(BASE_INCLUDES
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include_internal
        ${CMAKE_SOURCE_DIR}/rtl
        ${CMAKE_CURRENT_SOURCE_DIR}/common
    )
    set(RTL_LIBS)
    if(AKT_CKKS_TEST_USES_NTT_RTL)
        list(APPEND RTL_LIBS ${CMAKE_SOURCE_DIR}/rtl/the_nwc_4k_ntt.a)
    endif()
    if(AKT_CKKS_TEST_USES_IFFT_RTL)
        list(APPEND RTL_LIBS ${CMAKE_SOURCE_DIR}/rtl/fhe_ifft_4k_4lanes_double_253.a)
    endif()
    set(IS_RTL_TEST FALSE)
    if(AKT_CKKS_TEST_USES_NTT_RTL OR AKT_CKKS_TEST_USES_IFFT_RTL)
        set(IS_RTL_TEST TRUE)
    endif()

    foreach(MODE IN ITEMS emu report fpga)
        set(TARGET ${NAME}_${MODE})
        if(MODE STREQUAL emu)
            add_executable(${TARGET} ${TEST_SOURCE})
        else()
            add_executable(${TARGET} EXCLUDE_FROM_ALL ${TEST_SOURCE})
        endif()

        target_compile_features(${TARGET} PRIVATE cxx_std_17)
        target_include_directories(${TARGET} PRIVATE ${BASE_INCLUDES})
        target_compile_options(${TARGET} PRIVATE ${BASE_COMPILE_OPTIONS})
        target_link_options(${TARGET} PRIVATE ${BASE_LINK_OPTIONS})
        target_link_libraries(${TARGET} PRIVATE ${RTL_LIBS})

        if(DEFINED AKT_CKKS_TEST_P AND NOT "${AKT_CKKS_TEST_P}" STREQUAL "")
            target_compile_definitions(${TARGET} PRIVATE CKKS_TEST_P=${AKT_CKKS_TEST_P})
        endif()

        if(MODE STREQUAL emu)
            target_compile_definitions(${TARGET} PRIVATE FPGA_EMULATOR=1)
            if((NOT IS_RTL_TEST) OR SYCL_CKKS_REGISTER_RTL_EMULATOR_CTESTS)
                add_test(NAME ${TARGET} COMMAND ${TARGET})
            endif()
        elseif(MODE STREQUAL report)
            target_compile_definitions(${TARGET} PRIVATE FPGA_HARDWARE=1)
            target_link_options(${TARGET} PRIVATE
                -Xshardware
                -Xstarget=${SYCL_CKKS_FPGA_DEVICE}
                -fsycl-link=early
            )
        else()
            target_compile_definitions(${TARGET} PRIVATE FPGA_HARDWARE=1)
            target_link_options(${TARGET} PRIVATE
                -Xshardware
                -Xstarget=${SYCL_CKKS_FPGA_DEVICE}
                -reuse-exe=${CMAKE_CURRENT_BINARY_DIR}/${TARGET}
            )
        endif()
    endforeach()
endfunction()
