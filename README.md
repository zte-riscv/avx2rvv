# AVX2RVV

AVX512 Intrinsics Implementation for RISC-V Vector Extension

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
  -s, --suites CASETYPE   Select test suite (default: all → run SSE first, then AVX)
  TEST_NAME               Run specific test by name (supports partial matching)

Examples:
  ./tests/main                        # Run all tests
  ./tests/main mm_add_ps              # Run mm_add_ps test
  ./tests/main --index 5              # Run test at index 5
  ./tests/main --list                 # List all available tests
  ./tests/main --suite avx            # Run only AVX tests\n
  ./tests/main --suite sse --index 5  # Run SSE test at index 5
  ./tests/main --verbose add          # Run tests matching 'add' with verbose output

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

## Real-World Migration Examples

### Case Study 1: Basic SSE&AVX Operations Migration

**Objective**: Demonstrate seamless migration from x86 SSE&AVX to RISC-V RVV

**Source Code** (`testsse.cpp`):
```c
#include <stdio.h>
#include <stdint.h>
#include "sse2rvv.h"  // Replace <pmmintrin.h> for RISC-V
#include "avx2rvv.h"  // Replace <immintrin.h> for RISC-V

void sse_example() {
    int32_t a[4] = {-5, 13, 4, -20};
    int32_t b[4] = {12, 3, 0, 7};
    int32_t c[4] = {0};

    // Load 128-bit data (4 x 32-bit integers)
    __m128i t1 = _mm_loadu_si128((const __m128i*)a);
    __m128i t2 = _mm_loadu_si128((const __m128i*)b);

    // Execute 128-bit parallel addition
    __m128i dst = _mm_add_epi32(t1, t2);

    // Store results
    _mm_storeu_si128((__m128i*)c, dst);

    printf("Result: %d %d %d %d\n", c[0], c[1], c[2], c[3]);
}

int main(void) {
    sse_example();
    return 0;
}
```

**Migration Steps**:
1. Replace `#include <pmmintrin.h>` with `#include "sse2rvv.h"`
2. No source code changes required
3. Update compiler flags: `-march=rv64gcv_zba`

**Compilation & Execution**:
```bash
# Compile for RISC-V
riscv64-unknown-linux-gnu-g++ testsse.cpp -o testsse -march=rv64gcv_zba

# Run with QEMU
QEMU_LD_PREFIX=/opt/riscv/sysroot/ qemu-riscv64 ./testsse
# Output: Result: 7 16 4 -13
```

### Case Study 2: Image Processing Application

**Project**: [Prefetcher](https://github.com/ryanwang522/prefetcher.git) - Image processing with SIMD optimization

**Migration Process**:

1. **Header Replacement**:
   ```c
   // Before (x86)
   #include <xmmintrin.h>
   
   // After (RISC-V)
   #include "sse2rvv.h"
   ```

2. **Build Configuration**:
   ```makefile
   CC=/path/to/riscv64-unknown-linux-gnu-gcc
   CFLAGS = -O3 -march=rv64gcv_zba
   ```

3. **Results**:
   ```
   Matrix transpose operation:
   0  1  2  3
   4  5  6  7
   8  9 10 11
   12 13 14 15
   
   Transposed:
    0  4  8 12
   1  5  9 13
   2  6 10 14
   3  7 11 15
   
   SSE processing time: 1553431 us
   ```

### Case Study 3: Base64 Encoding/Decoding

**Project**: [Base64](https://github.com/aklomp/base64.git) - High-performance encoding library

**Migration Steps**:

1. **Header Updates**:
   ```c
   // lib/arch/avx512/codec.c
   // #include <immintrin.h>
   #include "avx2rvv.h"
   
   // lib/arch/ssse3/codec.c  
   // #include <immintrin.h>
   #include "avx2rvv.h"
   ```

2. **Build Configuration**:
   ```makefile
   CC=/path/to/riscv64-unknown-linux-gnu-gcc
   CFLAGS += -O3 -march=rv64gcv_zba -Wall -Wextra -pedantic \
             -DBASE64_STATIC_DEFINE -DBASE64_SSSE3_USE_ASM=0 -I.
   LD=/path/to/riscv64-unknown-linux-gnu-ld
   OBJCOPY=/path/to/riscv64-unknown-linux-gnu-objcopy
   ```

3. **Compilation**:
   ```bash
   export SSSE3_CFLAGS=
   export AVX512_CFLAGS=
   make clean && OPENMP=1 make && OPENMP=1 make -C test
   ```

4. **Result** (10MB buffer):
   ```
   Plain   encode:  801.69 MB/sec
   Plain   decode: 1195.70 MB/sec
   SSSE3   encode:  503.53 MB/sec
   SSSE3   decode:  556.33 MB/sec
   SSE41   encode:  705.87 MB/sec
   SSE41   decode:  598.00 MB/sec
   AVX512  encode:  269.52 MB/sec
   ```

### Migration Best Practices

**1. Systematic Migration Approach**:
```bash
# Step 1: Identify SIMD usage
grep -r "_mm_" src/ | grep -E "(include|#include)"

# Step 2: Replace headers
find . -name "*.c" -o -name "*.cpp" | xargs sed -i 's/#include <.*mmintrin\.h>/#include "sse2rvv.h"/g'
find . -name "*.c" -o -name "*.cpp" | xargs sed -i 's/#include <immintrin\.h>/#include "avx2rvv.h"/g'

# Step 3: Update build system
sed -i 's/-march=native/-march=rv64gcv_zba/g' Makefile
```

**2. Compatibility Verification**:
```c
// Add runtime checks for unsupported functions
#ifdef __riscv
    // Check for RVV support
    if (__riscv_v_elen < 128) {
        fprintf(stderr, "Warning: RVV not supported, falling back to scalar\n");
        use_scalar_implementation();
        return;
    }
#endif
```

**3. Performance Validation**:
```c
// Benchmark both implementations
void benchmark_migration() {
    clock_t start, end;
    
    // Test original x86 implementation
    start = clock();
    x86_simd_function();
    end = clock();
    double x86_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Test RISC-V implementation  
    start = clock();
    riscv_simd_function();
    end = clock();
    double riscv_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Performance ratio (RISC-V/x86): %.2f\n", riscv_time / x86_time);
}
```

### Common Migration Challenges and Solutions

**1. Unsupported Intrinsics**:
```c
// Problem: _mm_prefetch not implemented
// Solution: Use compiler hints or manual prefetching
#ifdef __riscv
    // Manual prefetch simulation
    __builtin_prefetch(ptr, 0, 3);  // Read, high temporal locality
#else
    _mm_prefetch(ptr, _MM_HINT_T1);
#endif
```

**2. Assembly Code Compatibility**:
```c
// Problem: Inline assembly not portable
// Solution: Conditional compilation
#ifdef __riscv
    // Use RVV intrinsics instead of assembly
    __m512i result = _mm512_add_epi32(a, b);
#elif defined(__x86_64__)
    // Original x86 assembly
    asm volatile ("vpaddd %0, %1, %2" : "=x"(result) : "x"(a), "x"(b));
#endif
```

**3. Memory Alignment Issues**:
```c
// Problem: Different alignment requirements
// Solution: Use portable alignment
void* aligned_alloc_portable(size_t alignment, size_t size) {
#ifdef __riscv
    return aligned_alloc(alignment, size);
#else
    return _mm_malloc(size, alignment);
#endif
}
```

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

## Development Roadmap

This project is under active development with a clear roadmap for expanding SIMD intrinsic support across different Intel x86 architectures.

## Function Statistics & Priority Analysis

Based on comprehensive analysis of x86 SIMD intrinsic usage patterns, we have identified the following function distribution:

| Vector Type | Header Files | Total Functions | High-Frequency Functions | Priority Level |
|-------------|--------------|-----------------|-------------------------|----------------|
| **AVX** | `immintrin.h` | 191 | 191 (100%) | 🔴 **Critical** |
| **AVX2** | `immintrin.h` | 233 | 191 (82%) | 🔴 **Critical** |
| **AVX512** | `immintrin.h`<br>`avx512fintrin.h`<br>`avx512vfintrin.h` | 2,665 | 2,624 (98%) | 🟡 **High** |
| **AVX512 Extensions** | Various | 2,459 | 1,855 (75%) | 🟢 **Medium** |
| **AVX512 Compute** | Various | 132 | 64 (48%) | 🔵 **Low** |
| **Total** | - | **6,327** | **5,554 (88%)** | - |

### Phase 1: Foundation & High-Priority Implementation
**Target Completion: December, 2025**

- **Priority Focus**: Complete AVX and AVX2 intrinsic libraries (384 functions)
- **Scope**: Organize and submit existing developed functions and test cases
- **Deliverables**: 
  - Complete AVX/AVX2 function library (100% coverage)
  - Performance benchmarking suite
- **Rationale**: AVX/AVX2 show 100% and 82% high-frequency usage respectively

### Phase 2: AVX512 Core Implementation
**Target Completion: July 2026**

- **Implementation Strategy**: Focus on high-frequency AVX512 functions (2,624 functions)
- **Batch Development**: 200-300 functions per month
- **Coverage Goals**:
  - Complete core AVX512 intrinsic function library (98% high-frequency coverage)
  - Comprehensive test suite for all implemented functions
  - Performance optimization for RISC-V Vector Extension
- **Quality Assurance**: Each batch includes comprehensive testing, documentation, and benchmarking

### Phase 3: AVX512 Extensions & Specialized Functions
**Target Completion: October, 2026**

- **Implementation Strategy**: Batch development approach (200-300 functions per month)
- **Coverage Goals**:
  - Complete AVX intrinsic function library
  - Complete AVX2 intrinsic function library  
  - Complete AVX512 intrinsic function library
  - Full test suite coverage for all implemented functions
- **Quality Assurance**: Each batch includes comprehensive testing and documentation

## Development Philosophy

- **Data-Driven**: Prioritize functions based on comprehensive usage analysis (88% high-frequency coverage)
- **Quality-First**: Every function includes comprehensive test cases and documentation
- **Performance-Oriented**: Optimize for RISC-V Vector Extension efficiency
- **Community-Focused**: Open development process with regular milestone releases
- **Backward Compatible**: Maintain API compatibility throughout development phases

### Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:
- Report bugs and request features
- Submit pull requests
- Contribute to test cases
- Help with documentation

---

## License

"avx2rvv is freely redistributable under the Apache License 2.0."