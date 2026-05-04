include(${CMAKE_CURRENT_LIST_DIR}/common/kernel_test_cmake.cmake)

set(CKKS_KERNEL_TEST_COMMON_INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

add_ckks_kernel_test(ckks_test_entry entry/entry_harness.cpp)

foreach(P RANGE 0 2)
    add_ckks_kernel_test(ckks_test_exit_c0_p${P} exit/exit_c0_harness.cpp CKKS_TEST_P ${P})
    add_ckks_kernel_test(ckks_test_exit_ntt_a_p${P} exit/exit_ntt_a_harness.cpp CKKS_TEST_P ${P})
    add_ckks_kernel_test(ckks_test_exit_ntt_b_p${P} exit/exit_ntt_b_harness.cpp CKKS_TEST_P ${P})
    add_ckks_kernel_test(ckks_test_scale_reduce_p${P} scale_reduce/scale_reduce_harness.cpp CKKS_TEST_P ${P})
    add_ckks_kernel_test(ckks_test_poly_mult_neg_add_p${P} poly_mult_neg_add/poly_mult_neg_add_harness.cpp CKKS_TEST_P ${P})
    add_ckks_kernel_test(ckks_test_ntt_a_p${P} ntt_a/ntt_a_harness.cpp CKKS_TEST_P ${P} CKKS_TEST_USES_NTT_RTL 1)
    add_ckks_kernel_test(ckks_test_ntt_b_p${P} ntt_b/ntt_b_harness.cpp CKKS_TEST_P ${P} CKKS_TEST_USES_NTT_RTL 1)
endforeach()

add_ckks_kernel_test(ckks_test_ifft ifft/ifft_harness.cpp CKKS_TEST_USES_IFFT_RTL 1)

set(CKKS_KERNEL_TEST_EMU_TARGETS
    ckks_test_entry_emu
    ckks_test_ifft_emu
)
foreach(P RANGE 0 2)
    list(APPEND CKKS_KERNEL_TEST_EMU_TARGETS
        ckks_test_exit_c0_p${P}_emu
        ckks_test_exit_ntt_a_p${P}_emu
        ckks_test_exit_ntt_b_p${P}_emu
        ckks_test_scale_reduce_p${P}_emu
        ckks_test_poly_mult_neg_add_p${P}_emu
        ckks_test_ntt_a_p${P}_emu
        ckks_test_ntt_b_p${P}_emu
    )
endforeach()

add_custom_target(ckks_kernel_tests_emu DEPENDS ${CKKS_KERNEL_TEST_EMU_TARGETS})

set(CKKS_KERNEL_TEST_REPORT_TARGETS
    ckks_test_entry_report
    ckks_test_ifft_report
)
foreach(P RANGE 0 2)
    list(APPEND CKKS_KERNEL_TEST_REPORT_TARGETS
        ckks_test_exit_c0_p${P}_report
        ckks_test_exit_ntt_a_p${P}_report
        ckks_test_exit_ntt_b_p${P}_report
        ckks_test_scale_reduce_p${P}_report
        ckks_test_poly_mult_neg_add_p${P}_report
        ckks_test_ntt_a_p${P}_report
        ckks_test_ntt_b_p${P}_report
    )
endforeach()

add_custom_target(ckks_kernel_tests_report DEPENDS ${CKKS_KERNEL_TEST_REPORT_TARGETS})
