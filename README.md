# avx2rvv

## Overview

**avx2rvv** is a header-only translation layer that maps Intel x86 SIMD intrinsics (SSE, AVX, AVX2, AVX512) to RISC‑V Vector (RVV) intrinsics. It enables existing x86 SIMD code to run on RISC‑V platforms with minimal source changes, allowing rapid workload bring-up, performance profiling, and hot path identification on RISC‑V targets.

Built on and extending [sse2rvv](https://github.com/pattonkan/sse2rvv), avx2rvv provides broader coverage for the AVX/AVX2/AVX512 families, implementing many functions from Intel’s intrinsic headers (such as `<immintrin.h>`) using RVV equivalents to match x86 semantics.

## Mapping and Coverage

Header file | Extension |
---|---|
`<mmintrin.h>` | MMX |
`<xmmintrin.h>` | SSE |
`<emmintrin.h>` | SSE2 |
`<pmmintrin.h>` | SSE3 |
`<tmmintrin.h>` | SSSE3 |
`<smmintrin.h>` | SSE4.1 |
`<nmmintrin.h>` | SSE4.2 |
`<wmmintrin.h>` | AES  |
`<immintrin.h>` | AVX  |

**Supported instruction sets:**  
- SSE, SSE2, SSE3, SSSE3  
- SSE4.1, SSE4.2, AES  
- AVX, AVX2, AVX512

**Design principle:**  
- Prefer one-to-one mappings to RVV intrinsics whenever possible  
- Otherwise, emulate semantics using concise RVV instruction sequences

**Examples:**
- Direct mapping: `_mm_add_epi16` → `__riscv_vadd_vv_i16m1`
- Composed mapping: `_mm_maddubs_epi16` implemented via multiple RVV ops

---

## Integration

1. **Add Headers:**  
   Place `avx2rvv.h` or `sse2rvv.h` in your project's include path.

2. **Replace x86 SIMD Headers:**  
   Locate and replace x86 SIMD header inclusions:
   ```c
   #include <xmmintrin.h>
   #include <emmintrin.h>
   #include <immintrin.h>
   ```
   with:
   ```c
   #include "sse2rvv.h"
   #include "avx2rvv.h"
   ```
   - Replace `{x,e,p,t,s,n,w}mmintrin.h` with `"sse2rvv.h"`
   - Replace `{avx,avx2,avx512f,avx512vl}intrin.h` or `immintrin.h` with `"avx2rvv.h"`

3. **Compiler Options:**  
   For RISC‑V (example for riscv64, adjust for your toolchain/CPU features):  
   ```
   -march=rv64gcv_zba -mabi=lp64d
   ```
## Run Built-in Test Suite

`avx2rvv` ships with a built-in test suite under the `tests/` directory. You can run all tests or a single test. Test inputs are provided at runtime and results are printed to stdout.

### Prerequisites
- A host RISC‑V toolchain (native or cross), or a host compiler plus QEMU for RISC‑V emulation
- GNU Make

### Run all tests (native toolchain)
```bash
make test
```

### Run a single test (example: mm_crc32_u8)
```bash
# Build test binaries
make

# Run test case help:
AVX2RVV Test Suite
Usage: ./tests/main [OPTIONS] [TEST_NAME]

Options:
  -h, --help              Show this help message
  -l, --list              List all available test cases
  -v, --verbose           Enable verbose output
  -q, --quiet             Suppress output except for errors
  -i, --index INDEX       Run test by index number
  TEST_NAME               Run specific test by name (supports partial matching)

Examples:
  ./tests/main                      # Run all tests
  ./tests/main mm_add_ps            # Run mm_add_ps test
  ./tests/main --index 5            # Run test at index 5
  ./tests/main --list               # List all available tests
  ./tests/main --verbose add        # Run tests matching 'add' with verbose output

# Run one case by name
./tests/main mm_crc32_u8
```
Expected output (sample):
```
Test mm_crc32_u8                    passed
SSE2RVV_TEST Complete!
Passed:  1
Failed:  0
Ignored: 0
Coverage rate: 100.00%
```

### Cross-compile for RISC‑V and run with QEMU
```bash
# Build with a cross toolchain
make CROSS_COMPILE=riscv64-unknown-elf-

# Run with qemu-riscv64 (if your tests are built as Linux user binaries)
# Example (adjust path/binary as needed):
qemu-riscv64 ./tests/main
```

Notes:
- Use `tests/main` to run the entire test matrix.
- For single tests, pass the exact test name to `tests/main $CASE`.
- If you target bare‑metal outputs, integrate with your runner or board bring‑up scripts accordingly.

---

## References

- [sse2rvv](https://github.com/pattonkan/sse2rvv)
- [sse2neon](https://github.com/DLTcollab/sse2neon)
- [neon2rvv](https://github.com/howjmay/neon2rvv)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [Microsoft: x86 intrinsics list](https://learn.microsoft.com/en-us/cpp/intrinsics/x86-intrinsics-list)
- [riscv-v-spec](https://github.com/riscv/riscv-v-spec)
- [rvv-intrinsic-doc](https://github.com/riscv-non-isa/rvv-intrinsic-doc)
- [riscv-c-api](https://github.com/riscv-non-isa/riscv-c-api-doc/blob/master/riscv-c-api.md)

---

## License

"avx2rvv is freely redistributable under the Apache License 2.0."