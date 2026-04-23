# SYCL CKKS Accelerator

Standalone Intel oneAPI/SYCL FPGA-emulator library for the CKKS symmetric-encryption accelerator pipeline extracted from the invasive SEAL-Embedded thesis tree.

## Scope

This repository owns only the accelerator implementation and its active RTL artifacts:

- `src/SYCL_ckks_sym.cpp`
- public C/C++ ABI header in `include/sycl_ckks_accelerator/SYCL_ckks_sym.h`
- internal SYCL pipeline headers in `include_internal/`
- active RTL archives/headers in `rtl/`

It intentionally does not own SEAL-Embedded generated adapter data. `adapter_output_data` remains generated and owned by the SEAL adapter / SEAL-Embedded program. The accelerator should receive all needed runtime data through `SYCL_encrypt(...)` arguments.

`.OLD` RTL artifacts are intentionally excluded.

## Build

Always initialize the Intel oneAPI environment first:

```bash
source /opt/intel/oneapi/2025.0/oneapi-vars.sh --force
```

Configure and build:

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j
```

Install to a local prefix consumed by the thin SEAL-Embedded patch:

```bash
cmake --install build --prefix /home/joe/Projects/Thesis/new/SYCL_Pipeline/install
```

Expected installed artifacts:

```text
install/lib/libsycl_ckks_accelerator.a
install/include/sycl_ckks_accelerator/SYCL_ckks_sym.h
```

## SEAL-Embedded integration

The first-pass SEAL integration should use a simple path variable:

```bash
-DSYCL_CKKS_ACCELERATOR_ROOT=/home/joe/Projects/Thesis/new/SYCL_Pipeline/install
```

Do not add polished `find_package(...)` integration until the simple path-based build is working.

## Smoke test policy

Use the SEAL-side short-run mode once implemented:

```bash
SE_TEST_MODE=sycl-smoke /path/to/seal_embedded_tests
```

Do not run the full verbose test executable by default.
