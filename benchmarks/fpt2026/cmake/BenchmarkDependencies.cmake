# Path-only dependency contract for the FPT 2026 driver. This module never
# downloads a dependency and never searches an unrelated checkout.

function(fpt2026_require_absolute_existing_directory variable)
    if(NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
        message(FATAL_ERROR "${variable} must be an explicit absolute path")
    endif()
    if(NOT IS_ABSOLUTE "${${variable}}")
        message(FATAL_ERROR "${variable} is not absolute: ${${variable}}")
    endif()
    if(NOT IS_DIRECTORY "${${variable}}")
        message(FATAL_ERROR "${variable} is not a directory: ${${variable}}")
    endif()
endfunction()

function(fpt2026_require_absolute_existing_file variable)
    if(NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
        message(FATAL_ERROR "${variable} must be an explicit absolute file path")
    endif()
    if(NOT IS_ABSOLUTE "${${variable}}")
        message(FATAL_ERROR "${variable} is not absolute: ${${variable}}")
    endif()
    if(NOT EXISTS "${${variable}}" OR IS_DIRECTORY "${${variable}}")
        message(FATAL_ERROR "${variable} is not a file: ${${variable}}")
    endif()
endfunction()

set(SYCL_CKKS_ACCELERATOR_ROOT "" CACHE PATH "Pinned SYCL accelerator source root")
set(SYCL_CKKS_ACCELERATOR_LIBRARY "" CACHE FILEPATH "Pinned accelerator static archive")
set(SEAL_EMBEDDED_ROOT "" CACHE PATH "Pinned SEAL-Embedded source root")
set(SEAL_EMBEDDED_LIBRARY "" CACHE FILEPATH "Unaccelerated SEAL-Embedded static archive")
set(SEAL_EMBEDDED_KEY "" CACHE FILEPATH "Pinned compact sk_8192.dat")
set(MICROSOFT_SEAL_ROOT "" CACHE PATH "Pinned Microsoft SEAL source/install root")
set(MICROSOFT_SEAL_LIBRARY "" CACHE FILEPATH "Pinned Microsoft SEAL static archive")
set(FPGA_REPORT_ROOT "" CACHE PATH "Pinned seed-7 report root")
set(FPT2026_REUSE_EXE "" CACHE FILEPATH "Optional seed-7 anchor executable")
set(FPT2026_FPGA_DEVICE "Agilex7" CACHE STRING "Exact FPGA target passed to the final hardware link")
option(FPT2026_ALLOW_DIRTY_DIAGNOSTIC
       "Allow dirty dependency trees for non-paper diagnostic builds" OFF)

set(SYCL_CKKS_ACCELERATOR_EXPECTED_COMMIT
    "" CACHE STRING "Exact committed 8k_benchmarks implementation revision")
set(SEAL_EMBEDDED_EXPECTED_COMMIT
    "0913fa9afe1f2bdc0d995f853a685aceea6d3cd0" CACHE STRING "Expected SEAL-Embedded revision")
set(MICROSOFT_SEAL_EXPECTED_COMMIT
    "79234726053c45eede688400aa219fdec0810bd8" CACHE STRING "Expected Microsoft SEAL revision")

if(NOT SYCL_CKKS_ACCELERATOR_EXPECTED_COMMIT MATCHES "^[0-9a-f]{40}$")
    message(FATAL_ERROR
        "SYCL_CKKS_ACCELERATOR_EXPECTED_COMMIT must name the exact committed benchmark implementation")
endif()

fpt2026_require_absolute_existing_directory(SYCL_CKKS_ACCELERATOR_ROOT)
fpt2026_require_absolute_existing_file(SYCL_CKKS_ACCELERATOR_LIBRARY)
fpt2026_require_absolute_existing_directory(SEAL_EMBEDDED_ROOT)
fpt2026_require_absolute_existing_file(SEAL_EMBEDDED_LIBRARY)
fpt2026_require_absolute_existing_file(SEAL_EMBEDDED_KEY)
fpt2026_require_absolute_existing_directory(MICROSOFT_SEAL_ROOT)
fpt2026_require_absolute_existing_file(MICROSOFT_SEAL_LIBRARY)
fpt2026_require_absolute_existing_directory(FPGA_REPORT_ROOT)

find_package(Git REQUIRED)
function(fpt2026_require_git_state repository expected_head expected_branch label)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${repository}" rev-parse HEAD
        RESULT_VARIABLE result OUTPUT_VARIABLE observed_head
        ERROR_VARIABLE git_error OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT result EQUAL 0 OR NOT observed_head STREQUAL expected_head)
        message(FATAL_ERROR
            "${label} HEAD mismatch: expected ${expected_head}, observed ${observed_head}; ${git_error}")
    endif()
    if(NOT "${expected_branch}" STREQUAL "")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${repository}" branch --show-current
            RESULT_VARIABLE result OUTPUT_VARIABLE observed_branch
            ERROR_VARIABLE git_error OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT result EQUAL 0 OR NOT observed_branch STREQUAL expected_branch)
            message(FATAL_ERROR
                "${label} branch mismatch: expected ${expected_branch}, observed ${observed_branch}; ${git_error}")
        endif()
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${repository}" status --porcelain=v1 --untracked-files=all
        RESULT_VARIABLE result OUTPUT_VARIABLE dirty_state
        ERROR_VARIABLE git_error OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Cannot inspect ${label} worktree: ${git_error}")
    endif()
    if(NOT "${dirty_state}" STREQUAL "" AND NOT FPT2026_ALLOW_DIRTY_DIAGNOSTIC)
        message(FATAL_ERROR
            "${label} worktree is dirty; paper builds require committed sources")
    endif()
endfunction()

fpt2026_require_git_state(
    "${SYCL_CKKS_ACCELERATOR_ROOT}" "${SYCL_CKKS_ACCELERATOR_EXPECTED_COMMIT}"
    "8k_benchmarks" "SYCL accelerator")
fpt2026_require_git_state(
    "${SEAL_EMBEDDED_ROOT}" "${SEAL_EMBEDDED_EXPECTED_COMMIT}"
    "" "SEAL-Embedded")
fpt2026_require_git_state(
    "${MICROSOFT_SEAL_ROOT}" "${MICROSOFT_SEAL_EXPECTED_COMMIT}"
    "" "Microsoft SEAL")

set(_accelerator_header
    "${SYCL_CKKS_ACCELERATOR_ROOT}/include/sycl_ckks_accelerator/SYCL_ckks_benchmark.h")
if(NOT EXISTS "${_accelerator_header}")
    message(FATAL_ERROR "Missing public accelerator benchmark API: ${_accelerator_header}")
endif()
set(_seal_embedded_header "${SEAL_EMBEDDED_ROOT}/device/lib/seal_embedded.h")
if(NOT EXISTS "${_seal_embedded_header}")
    message(FATAL_ERROR "Missing SEAL-Embedded public header: ${_seal_embedded_header}")
endif()

set(_seal_include_candidates
    "${MICROSOFT_SEAL_ROOT}/native/src"
    "${MICROSOFT_SEAL_ROOT}/include"
    "${MICROSOFT_SEAL_ROOT}")
set(MICROSOFT_SEAL_INCLUDE_DIR "")
foreach(candidate IN LISTS _seal_include_candidates)
    if(EXISTS "${candidate}/seal/seal.h")
        set(MICROSOFT_SEAL_INCLUDE_DIR "${candidate}")
        break()
    endif()
endforeach()
if(NOT MICROSOFT_SEAL_INCLUDE_DIR)
    message(FATAL_ERROR
        "MICROSOFT_SEAL_ROOT does not contain native/src/seal/seal.h or include/seal/seal.h")
endif()

add_library(fpt2026_accelerator STATIC IMPORTED GLOBAL)
set_target_properties(fpt2026_accelerator PROPERTIES
    IMPORTED_LOCATION "${SYCL_CKKS_ACCELERATOR_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SYCL_CKKS_ACCELERATOR_ROOT}/include")

add_library(seal_embedded_imported STATIC IMPORTED GLOBAL)
set_target_properties(seal_embedded_imported PROPERTIES
    IMPORTED_LOCATION "${SEAL_EMBEDDED_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SEAL_EMBEDDED_ROOT}/device/lib")

add_library(microsoft_seal_imported STATIC IMPORTED GLOBAL)
set_target_properties(microsoft_seal_imported PROPERTIES
    IMPORTED_LOCATION "${MICROSOFT_SEAL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MICROSOFT_SEAL_INCLUDE_DIR}")

set(FPT2026_ACCELERATOR_LINK_OPTIONS
    "-fsycl;-std=c++17;-Rno-debug-disables-optimization;-DCSL_USE_GMP;-DCSL_SYCL;-Wno-return-type-c-linkage;-qactypes;-fintelfpga;-Xshardware;-Xstarget=${FPT2026_FPGA_DEVICE};-DFPGA_HARDWARE;LINKER:--allow-multiple-definition"
    CACHE STRING "Pinned host-side accelerator final-link options")
