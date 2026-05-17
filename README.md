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

This project has three distinct target paths:

- emulator: fast CPU-backed FPGA emulator build for functional debugging
- simulation: RTL simulator-backed build using `-Xssimulation`
- hardware/report: FPGA hardware report/early-link flow using `-Xshardware`

### Emulator archive

```bash
cmake -S . -B build-emu -DCMAKE_CXX_COMPILER=icpx
cmake --build build-emu -j
```

### Simulator archive

The simulator path uses `FPGA_SIMULATOR`, `sycl::ext::intel::fpga_simulator_selector_v`, and `-Xssimulation`. Use this when emulator behavior differs from expected hardware behavior or when you need cycle/bit-accurate simulator visibility without doing a full hardware compile.

```bash
cmake -S . -B build-sim \
  -DCMAKE_CXX_COMPILER=icpx \
  -DSYCL_CKKS_FPGA_SIMULATION=ON \
  -DSYCL_CKKS_FPGA_DEVICE=<family_or_part_or_board_target>
cmake --build build-sim --target sycl_ckks_accelerator_sim_archive -j
```

To enable waveform capture, add:

```bash
-DSYCL_CKKS_SIMULATION_WAVEFORMS=ON
```

Optionally set a hierarchy depth:

```bash
-DSYCL_CKKS_SIMULATION_WAVEFORM_DEPTH=0
```

`0` requests all hierarchy with `-Xsghdl=0`; leaving the depth empty uses Intel's default `-Xsghdl` depth.

To run a downstream executable against the simulator device, the Intel guide requires enabling the simulation runtime device search, for example:

```bash
export CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
# or, if automatic discovery fails:
export INTELFPGA_SIM_DEVICE_SPEC_DIR=/path/to/<project>.prj
# sometimes also needed:
export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
```

Unset those variables before returning to physical FPGA runs.

### Hardware report

```bash
cmake -S . -B build-hw \
  -DCMAKE_CXX_COMPILER=icpx \
  -DSYCL_CKKS_FPGA_HARDWARE=ON \
  -DSYCL_CKKS_FPGA_DEVICE=<family_or_part_or_board_target>
cmake --build build-hw --target sycl_ckks_accelerator_report -j
```

Install to a local prefix consumed by the thin SEAL-Embedded patch:

```bash
cmake --install build-emu --prefix /home/joe/Projects/Thesis/new/SYCL_Pipeline/install
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

## FPGA Test policy

For accelerator-enabled SEAL-Embedded builds, the default executable path should be the FPGA Test:

```bash
/path/to/seal_embedded_tests
```

You can also request it explicitly:

```bash
SE_TEST_MODE=fpga-test /path/to/seal_embedded_tests
```

Use `SE_TEST_MODE=all` only when you intentionally want the original CPU-oriented SEAL test suite instead of the FPGA Test path.
