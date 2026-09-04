cmake_minimum_required(VERSION 3.20)

project(fpt2026_benchmark VERSION 1.0.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(BenchmarkDependencies)

add_library(fpt2026_host STATIC
    src/benchmark_vectors.cpp
    src/correctness_metrics.cpp
    src/provenance.cpp
    src/benchmark_record.cpp
    src/seal_oracle.cpp
    src/seal_embedded_runner.cpp
    src/accelerator_runner.cpp
    src/seal_embedded_c_runner.c)

target_include_directories(fpt2026_host
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${SEAL_EMBEDDED_ROOT}/device/lib")

target_compile_definitions(fpt2026_host PRIVATE
    FPT2026_ACCELERATOR_COMMIT="${SYCL_CKKS_ACCELERATOR_EXPECTED_COMMIT}"
    FPT2026_SEAL_EMBEDDED_COMMIT="${SEAL_EMBEDDED_EXPECTED_COMMIT}"
    FPT2026_MICROSOFT_SEAL_COMMIT="${MICROSOFT_SEAL_EXPECTED_COMMIT}")

set_source_files_properties(src/seal_embedded_c_runner.c PROPERTIES
    LANGUAGE C
    C_STANDARD 11
    C_STANDARD_REQUIRED YES)

target_link_libraries(fpt2026_host
    PUBLIC fpt2026_accelerator microsoft_seal_imported
    PRIVATE seal_embedded_imported m)

target_compile_options(fpt2026_host PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

add_executable(fpt2026_benchmark src/benchmark_main.cpp)
target_link_libraries(fpt2026_benchmark PRIVATE fpt2026_host)

# These are final-link options only. They do not compile an FPGA image. The
# caller supplies -reuse-exe through FPT2026_REUSE_EXE when image reuse is used.
target_link_options(fpt2026_benchmark PRIVATE ${FPT2026_ACCELERATOR_LINK_OPTIONS})
if(FPT2026_REUSE_EXE)
    fpt2026_require_absolute_existing_file(FPT2026_REUSE_EXE)
    target_link_options(fpt2026_benchmark PRIVATE "-reuse-exe=${FPT2026_REUSE_EXE}")
endif()

option(FPT2026_ENABLE_SOURCE_TEST_TARGETS
       "Declare host source-test executables (never enabled for paper builds)" OFF)
if(FPT2026_ENABLE_SOURCE_TEST_TARGETS)
    foreach(test_name IN ITEMS
            benchmark_vectors_test
            correctness_metrics_test
            result_writer_test
            seal_ciphertext_bridge_test
            seal_embedded_runner_test)
        add_executable(${test_name} "tests/${test_name}.cpp")
        target_link_libraries(${test_name} PRIVATE fpt2026_host)
    endforeach()
endif()
