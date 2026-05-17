# SEAL-Embedded integration patch

`seal-sycl-accelerator.patch` applies the SEAL-Embedded side of the split-repo SYCL CKKS accelerator integration.

It adds:

- `SE_ENABLE_SYCL_ACCELERATOR`
- `SE_SYCL_FPGA_HARDWARE`
- `SYCL_CKKS_ACCELERATOR_ROOT`
- `SYCL_CKKS_ACCELERATOR_LIBRARY`
- the SYCL FPGA Test entry point and test file

## Agilex7 path convention

The current Agilex7 accelerator clone is:

```bash
/home/uwb_student00/Joe/new/SYCL_CKKS_PIPELINE
```

Do not use the old pre-GitHub clone path:

```bash
/home/uwb_student00/Joe/new/SYCL_Pipeline
```

## Applying the patch

From the SEAL-Embedded repo root:

```bash
cd /home/uwb_student00/Joe/new/SEAL-Embedded
git apply /home/uwb_student00/Joe/new/SYCL_CKKS_PIPELINE/patches/seal-sycl-accelerator.patch
```

## Configuring SEAL against a full hardware accelerator archive

After the full accelerator hardware build finishes, configure SEAL with the exact archive path produced by that build:

```bash
cd /home/uwb_student00/Joe/new/SEAL-Embedded/device
source /etc/profile.d/quartus.sh
export OFS_OCL_SHIM_ROOT=/home/uwb_student00/IA-840f/IOFS_BUILD_ROOT/oneapi-asp/ia840f
source /opt/intel/oneapi/setvars.sh

cmake -S . -B build-sycl-hw-link \
  -DSE_BUILD_LOCAL=ON \
  -DSE_BUILD_TYPE=Tests \
  -DSE_ENABLE_SYCL_ACCELERATOR=ON \
  -DSE_SYCL_FPGA_HARDWARE=ON \
  -DSYCL_CKKS_ACCELERATOR_ROOT=/home/uwb_student00/Joe/new/SYCL_CKKS_PIPELINE \
  -DSYCL_CKKS_ACCELERATOR_LIBRARY=/home/uwb_student00/Joe/new/SYCL_CKKS_PIPELINE/build-full-hw-YYYYMMDD-HHMMSS/sycl_ckks_accelerator_hw.a \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx
```

Replace `build-full-hw-YYYYMMDD-HHMMSS` with the actual full-hardware build directory.

Then build the integrated test executable:

```bash
cmake --build build-sycl-hw-link --target seal_embedded_tests -j$(nproc)
```

## Running the FPGA Test

For accelerator-enabled builds, the default executable path is the FPGA Test:

```bash
./build-sycl-hw-link/bin/seal_embedded_tests
```

Explicit mode names:

```bash
SE_TEST_MODE=fpga-test ./build-sycl-hw-link/bin/seal_embedded_tests
SE_TEST_MODE=all ./build-sycl-hw-link/bin/seal_embedded_tests
```

`SE_TEST_MODE=all` runs the original CPU-oriented test suite instead of taking the default FPGA Test path.
