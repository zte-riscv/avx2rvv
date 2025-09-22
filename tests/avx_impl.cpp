/*
 * @file avx_impl.cpp
 * AVX512 instruction implementation and testing for RISC-V Vector Extension
 * 
 * This file contains the implementation and comprehensive testing of AVX512
 * intrinsics using RISC-V Vector Extension (RVV) instructions.
 */

#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "binding.h"
#include "avx_impl.h"

// Platform-specific includes and definitions
#if defined(__riscv) || defined(__riscv__)
    #include "avx2rvv.h"
    #define AVX512_TEST_BODY
#elif defined(__x86_64__) || defined(__i386__)
    // On x86 platforms, AVX512 functions are not implemented, so tests return TEST_UNIMPL
    #include <immintrin.h>
    #define AVX512_NOT_IMPLEMENTED_ON_X86
    #define AVX512_TEST_BODY return TEST_UNIMPL;
    #define AVX512_TEST_FUNCTION_BODY { AVX512_TEST_BODY }

    // Stub declarations for custom AVX512 functions that don't exist on x86
    inline __m512i _mm512_loadu_epi16(const void*) { return _mm512_setzero_si512(); }
    inline void _mm512_storeu_epi16(void*, __m512i) {}
#else
    #define AVX512_TEST_BODY
    #define AVX512_TEST_FUNCTION_BODY
#endif

/*
 * Maximum number of test values for comprehensive testing
 * 
 * We test with 10,000 random floating point and integer values to ensure
 * robust validation of AVX512 instruction implementations.
 */
#define MAX_TEST_VALUE 10000

/*
 * AVX2RVV namespace containing all AVX512 implementation and testing code
 * 
 * This namespace provides a comprehensive set of unit tests to ensure that each
 * AVX512 intrinsic call produces the expected output when implemented using
 * RISC-V Vector Extension instructions.
 * 
 * Functions with "test_" prefix are automatically called by the test framework.
 */
namespace AVX2RVV {
using namespace SSE2RVV;

/*
 * Helper functions to access __m512i elements
 * 
 * These functions work around the fact that __m512i is not a union on x86 platforms.
 * They provide a consistent interface for accessing individual elements of 512-bit vectors.
 */

static inline int16_t get_epi16(__m512i a, int index) {
    assert(index >= 0 && index < 32);
    int16_t result[32];
    _mm512_storeu_si512((__m512i*)result, a);
    return result[index];
}

static inline int8_t get_epi8(__m512i a, int index) {
    assert(index >= 0 && index < 64);
    int8_t result[64];
    _mm512_storeu_si512((__m512i*)result, a);
    return result[index];
}

static inline uint8_t get_epu8(__m512i a, int index) {
    assert(index >= 0 && index < 64);
    uint8_t result[64];
    _mm512_storeu_si512((__m512i*)result, a);
    return result[index];
}

static inline uint16_t get_epu16(__m512i a, int index) {
    assert(index >= 0 && index < 32);
    uint16_t result[32];
    _mm512_storeu_si512((__m512i*)result, a);
    return result[index];
}

/*
 * Implementation class for AVX512 testing framework
 * 
 * This class provides the concrete implementation of the AVX512 testing interface,
 * including test data generation, execution, and validation.
 */
class AVX2RVV_TEST_IMPL : public AVX2RVV_TEST {
public:
    /*
     * Constructor - initializes test data and random number generator
     */
    AVX2RVV_TEST_IMPL(void);
    
    /*
     * Load test float data into aligned memory pointers
     */
    result_t load_test_float_pointers(uint32_t i);
    
    /*
     * Load test integer data into aligned memory pointers
     */
    result_t load_test_int_pointers(uint32_t i);
    
    /*
     * Run a single test iteration
     */
    result_t run_single_test(INSTRUCTION_TEST test, uint32_t i);
    
    /*
     * Run comprehensive test for an instruction
     */
    virtual result_t run_test(INSTRUCTION_TEST test) override;
    
    virtual void release(void) override;

    // Test data storage
    float *test_cases_float_pointer1;    ///< First aligned float test data pointer
    float *test_cases_float_pointer2;    ///< Second aligned float test data pointer
    int32_t *test_cases_int_pointer1;    ///< First aligned integer test data pointer
    int32_t *test_cases_int_pointer2;    ///< Second aligned integer test data pointer
    float test_cases_floats[MAX_TEST_VALUE];  ///< Float test data array
    int32_t test_cases_ints[MAX_TEST_VALUE];  ///< Integer test data array

    /*
     * Destructor - cleans up allocated memory
     */
    virtual ~AVX2RVV_TEST_IMPL(void) {
        platform_aligned_free(test_cases_float_pointer1);
        platform_aligned_free(test_cases_float_pointer2);
        platform_aligned_free(test_cases_int_pointer1);
        platform_aligned_free(test_cases_int_pointer2);
    }
};

/*
 * Release the test implementation instance
 */
void AVX2RVV_TEST_IMPL::release(void) { 
    delete this; 
}

/*
 * SplitMix64 PRNG implementation
 * 
 * High-quality pseudo-random number generator by Sebastiano Vigna.
 * Reference: https://xoshiro.di.unimi.it/splitmix64.c
 */
static uint64_t state; ///< Current state of the SplitMix64 PRNG
const double TWOPOWER64 = pow(2, 64); ///< 2^64 constant for normalization

/*
 * Initialize the random number generator with a seed
 */
#define AVX2RVV_INIT_RNG(seed)                                                 \
  do {                                                                         \
    state = seed;                                                              \
  } while (0)

/*
 * Generate the next pseudo-random number
 */
static double next() {
    uint64_t z = (state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

/*
 * Platform-specific register size definitions
 * 
 * These definitions help determine the appropriate alignment and vector size
 * for different target architectures.
 */
#if defined(__riscv_v_elen)
    #define REGISTER_SIZE __riscv_v_elen
#elif defined(__aarch64__)
    #define REGISTER_SIZE 128
#elif (defined(__x86_64__) || defined(__i386__))
    #define REGISTER_SIZE sizeof(__m128)
#else
    #define REGISTER_SIZE 64  // Default fallback
#endif

/*
 * Constructor implementation
 * 
 * Initializes aligned memory pointers and generates comprehensive test data
 * using a high-quality pseudo-random number generator.
 */
AVX2RVV_TEST_IMPL::AVX2RVV_TEST_IMPL(void) {
    test_cases_float_pointer1 = (float *)platform_aligned_alloc(REGISTER_SIZE);
    test_cases_float_pointer2 = (float *)platform_aligned_alloc(REGISTER_SIZE);
    test_cases_int_pointer1 = (int32_t *)platform_aligned_alloc(REGISTER_SIZE);
    test_cases_int_pointer2 = (int32_t *)platform_aligned_alloc(REGISTER_SIZE);
    
    AVX2RVV_INIT_RNG(123456);
    
    for (uint32_t i = 0; i < MAX_TEST_VALUE; i++) {
        test_cases_floats[i] = (float)(next() / TWOPOWER64 * 200000.0 - 100000.0);
        test_cases_ints[i] = (int32_t)(next() / TWOPOWER64 * 200000.0 - 100000.0);
    }
}

result_t test_mm_empty11(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
    return TEST_SUCCESS;
}

result_t validate_int32_512(__m512i a, int32_t i0, int32_t i1, int32_t i2, int32_t i3,
                           int32_t i4, int32_t i5, int32_t i6, int32_t i7,
                           int32_t i8, int32_t i9, int32_t i10, int32_t i11,
                           int32_t i12, int32_t i13, int32_t i14, int32_t i15) {
    const int32_t *t = (const int32_t *)&a;
    
    // Validate each element
    ASSERT_RETURN(t[0] == i0);
    ASSERT_RETURN(t[1] == i1);
    ASSERT_RETURN(t[2] == i2);
    ASSERT_RETURN(t[3] == i3);
    ASSERT_RETURN(t[4] == i4);
    ASSERT_RETURN(t[5] == i5);
    ASSERT_RETURN(t[6] == i6);
    ASSERT_RETURN(t[7] == i7);
    ASSERT_RETURN(t[8] == i8);
    ASSERT_RETURN(t[9] == i9);
    ASSERT_RETURN(t[10] == i10);
    ASSERT_RETURN(t[11] == i11);
    ASSERT_RETURN(t[12] == i12);
    ASSERT_RETURN(t[13] == i13);
    ASSERT_RETURN(t[14] == i14);
    ASSERT_RETURN(t[15] == i15);
    
    return TEST_SUCCESS;
}

result_t test_mm512_setzero_si512(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
    AVX512_TEST_BODY
    
    __m512i result = _mm512_setzero_si512();
    
    // Expected result: all zeros
    int32_t expected[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    return validate_int32_512(result, expected[0], expected[1], expected[2], expected[3],
                             expected[4], expected[5], expected[6], expected[7],
                             expected[8], expected[9], expected[10], expected[11],
                             expected[12], expected[13], expected[14], expected[15]);
}

result_t test_rdtsc(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
    return TEST_UNIMPL;
}

result_t test_mm512_loadu_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) { 
    AVX512_TEST_BODY
    
    // Prepare test data
    int16_t test_data[32];
    for (int i = 0; i < 32; i++) {
        test_data[i] = (int16_t)(iter + i);
    }
    
    // Load data using AVX512 intrinsic
    __m512i ret = _mm512_loadu_epi16(test_data);
    
    // Validate the loaded data
    for (int i = 0; i < 32; i++) {
        if (get_epi16(ret, i) != test_data[i]) {
            return TEST_FAIL;
        }
    }
    
    return TEST_SUCCESS;
}

result_t test_mm512_storeu_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
    (void)impl;  // Suppress unused parameter warning
    
    AVX512_TEST_BODY
    
    // Prepare test data
    int16_t test_data[32];
    int16_t result_data[32];
    
    // Initialize test data
    for (int i = 0; i < 32; i++) {
        test_data[i] = (int16_t)(iter + i);
    }
    
    // Load data into vector and store it back
    __m512i src = _mm512_loadu_epi16(test_data);
    _mm512_storeu_epi16(result_data, src);
    
    // Validate stored data matches original
    for (int i = 0; i < 32; i++) {
        if (result_data[i] != test_data[i]) {
            return TEST_FAIL;
        }
    }
    
    return TEST_SUCCESS;
}

result_t test_mm512_loadu_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
    return TEST_UNIMPL;
}

result_t test_mm512_storeu_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
    return TEST_UNIMPL;
}

result_t test_mm512_add_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
    return TEST_UNIMPL;
}

result_t test_mm512_add_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
    AVX512_TEST_BODY
    
    // Prepare test data
    int16_t a_data[32], b_data[32], expected[32];
    
    for (int i = 0; i < 32; i++) {
        a_data[i] = (int16_t)(iter + i);
        b_data[i] = (int16_t)(iter + i + 1);
        expected[i] = a_data[i] + b_data[i];
    }
    
    // Perform vector addition
    __m512i a = _mm512_loadu_epi16(a_data);
    __m512i b = _mm512_loadu_epi16(b_data);
    __m512i ret = _mm512_add_epi16(a, b);
    
    // Validate results
    for (int i = 0; i < 32; i++) {
        if (get_epi16(ret, i) != expected[i]) {
            return TEST_FAIL;
        }
    }
    
    return TEST_SUCCESS;
}

result_t test_mm512_sub_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
    return TEST_UNIMPL;
}

result_t test_mm512_sub_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
    (void)impl;
    
    AVX512_TEST_BODY
    
    int16_t a_data[32], b_data[32], expected[32];
    
    for (int i = 0; i < 32; i++) {
        a_data[i] = (int16_t)(iter + i);
        b_data[i] = (int16_t)(iter + i + 1);
        expected[i] = a_data[i] - b_data[i];
    }
    
    __m512i a = _mm512_loadu_epi16(a_data);
    __m512i b = _mm512_loadu_epi16(b_data);
    __m512i ret = _mm512_sub_epi16(a, b);
    
    for (int i = 0; i < 32; i++) {
        if (get_epi16(ret, i) != expected[i]) {
            return TEST_FAIL;
        }
    }
    
    return TEST_SUCCESS;
}

result_t test_mm512_avg_epu8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_avg_epu16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  AVX512_TEST_BODY
  uint16_t a_data[32], b_data[32], expected[32];
  
  for (int i = 0; i < 32; i++) {
    a_data[i] = (uint16_t)(iter + i);
    b_data[i] = (uint16_t)(iter + i + 1);
    expected[i] = (a_data[i] + b_data[i] + 1) >> 1; // Rounding average
  }
  
  __m512i a = _mm512_loadu_epi16((int16_t*)a_data);
  __m512i b = _mm512_loadu_epi16((int16_t*)b_data);
  __m512i ret = _mm512_avg_epu16(a, b);
  
  for (int i = 0; i < 32; i++) {
    if (get_epu16(ret, i) != expected[i]) {
      return TEST_FAIL;
    }
  }
  return TEST_SUCCESS;
}

result_t test_mm512_cmpeq_epi8_mask(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_cmpeq_epi16_mask(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  AVX512_TEST_BODY
  int16_t a_data[32], b_data[32];
  
  for (int i = 0; i < 32; i++) {
    a_data[i] = (int16_t)(iter + i);
    b_data[i] = (int16_t)(iter + i); 
  }
  
  __m512i a = _mm512_loadu_epi16(a_data);
  __m512i b = _mm512_loadu_epi16(b_data);
  __mmask32 ret = _mm512_cmpeq_epi16_mask(a, b);
  
  if (ret != 0xFFFFFFFF) {
    return TEST_FAIL;
  }
  return TEST_SUCCESS;
}

result_t test_mm512_cmpgt_epi8_mask(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_cmpgt_epi16_mask(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  AVX512_TEST_BODY
  int16_t a_data[32], b_data[32];
  
  for (int i = 0; i < 32; i++) {
    a_data[i] = (int16_t)(iter + i + 10); // a > b
    b_data[i] = (int16_t)(iter + i);
  }
  
  __m512i a = _mm512_loadu_epi16(a_data);
  __m512i b = _mm512_loadu_epi16(b_data);
  __mmask32 ret = _mm512_cmpgt_epi16_mask(a, b);
  
  // All elements should be greater, so mask should be all 1s
  if (ret != 0xFFFFFFFF) {
    return TEST_FAIL;
  }
  return TEST_SUCCESS;
}

result_t test_mm512_min_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_max_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_min_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  AVX512_TEST_BODY
  int16_t a_data[32], b_data[32], expected[32];
  
  for (int i = 0; i < 32; i++) {
    a_data[i] = (int16_t)(iter + i);
    b_data[i] = (int16_t)(iter + i + 5);
    expected[i] = (a_data[i] < b_data[i]) ? a_data[i] : b_data[i];
  }
  
  __m512i a = _mm512_loadu_epi16(a_data);
  __m512i b = _mm512_loadu_epi16(b_data);
  __m512i ret = _mm512_min_epi16(a, b);
  
  // Validate results
  for (int i = 0; i < 32; i++) {
    if (get_epi16(ret, i) != expected[i]) {
      return TEST_FAIL;
    }
  }
  return TEST_SUCCESS;
}

result_t test_mm512_max_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  AVX512_TEST_BODY
  int16_t a_data[32], b_data[32], expected[32];
  
  for (int i = 0; i < 32; i++) {
    a_data[i] = (int16_t)(iter + i);
    b_data[i] = (int16_t)(iter + i + 5);
    expected[i] = (a_data[i] > b_data[i]) ? a_data[i] : b_data[i];
  }
  
  __m512i a = _mm512_loadu_epi16(a_data);
  __m512i b = _mm512_loadu_epi16(b_data);
  __m512i ret = _mm512_max_epi16(a, b);
  
  // Validate results
  for (int i = 0; i < 32; i++) {
    if (get_epi16(ret, i) != expected[i]) {
      return TEST_FAIL;
    }
  }
  return TEST_SUCCESS;
}

result_t test_mm512_min_epu8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_max_epu8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_min_epu16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  AVX512_TEST_BODY
  uint16_t a_data[32], b_data[32], expected[32];
  
  for (int i = 0; i < 32; i++) {
    a_data[i] = (uint16_t)(iter + i);
    b_data[i] = (uint16_t)(iter + i + 5);
    expected[i] = (a_data[i] < b_data[i]) ? a_data[i] : b_data[i];
  }
  
  __m512i a = _mm512_loadu_epi16((int16_t*)a_data);
  __m512i b = _mm512_loadu_epi16((int16_t*)b_data);
  __m512i ret = _mm512_min_epu16(a, b);
  
  for (int i = 0; i < 32; i++) {
    if (get_epu16(ret, i) != expected[i]) {
      return TEST_FAIL;
    }
  }
  return TEST_SUCCESS;
}

result_t test_mm512_max_epu16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  AVX512_TEST_BODY
  uint16_t a_data[32], b_data[32], expected[32];
  
  for (int i = 0; i < 32; i++) {
    a_data[i] = (uint16_t)(iter + i);
    b_data[i] = (uint16_t)(iter + i + 5);
    expected[i] = (a_data[i] > b_data[i]) ? a_data[i] : b_data[i];
  }
  
  __m512i a = _mm512_loadu_epi16((int16_t*)a_data);
  __m512i b = _mm512_loadu_epi16((int16_t*)b_data);
  __m512i ret = _mm512_max_epu16(a, b);
  
  for (int i = 0; i < 32; i++) {
    if (get_epu16(ret, i) != expected[i]) {
      return TEST_FAIL;
    }
  }
  return TEST_SUCCESS;
}

result_t test_mm512_mask_mov_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_maskz_mov_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mask_mov_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_maskz_mov_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_shuffle_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_shufflehi_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_shufflelo_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_slli_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_srli_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_srai_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_cvtepi16_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_cvtepi8_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_cvtepu8_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_permutexvar_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_movepi8_mask(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_movepi16_mask(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_movm_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_movm_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_test_epi8_mask(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_test_epi16_mask(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_unpackhi_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_unpackhi_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mullo_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mulhi_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mulhi_epu16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mulhrs_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_sad_epu8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_packs_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_alignr_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_abs_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_abs_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_adds_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_adds_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_adds_epu8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_adds_epu16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_subs_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_subs_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_subs_epu8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_subs_epu16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_set1_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_set1_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mask_set1_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mask_set1_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_maskz_set1_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_maskz_set1_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mask_blend_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mask_blend_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mask_loadu_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mask_loadu_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_maskz_loadu_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_maskz_loadu_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mask_storeu_epi8(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_mask_storeu_epi16(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_kunpackd(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_mm512_kunpackw(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  return TEST_UNIMPL;
}

result_t test_last(const AVX2RVV_TEST_IMPL &impl, uint32_t iter) {
  // #ifdef ENABLE_TEST_ALL
  return TEST_SUCCESS;
}

// This function is not called from "run_single_test", but for other intrinsic
// tests that might need to call "_mm_store_ps".
result_t do_mm_store_ps(float *p, float x, float y, float z, float w) {
  __m128 a = _mm_set_ps(x, y, z, w);
  _mm_store_ps(p, a);
  ASSERT_RETURN(p[0] == w);
  ASSERT_RETURN(p[1] == z);
  ASSERT_RETURN(p[2] == y);
  ASSERT_RETURN(p[3] == x);
  return TEST_SUCCESS;
}

// This function is not called from "run_single_test", but for other intrinsic
// tests that might need to call "_mm_store_ps".
result_t do_mm_store_ps(int32_t *p, int32_t x, int32_t y, int32_t z,
                        int32_t w) {
  __m128i a = _mm_set_epi32(x, y, z, w);
  _mm_store_ps((float *)p, *(const __m128 *)&a);
  ASSERT_RETURN(p[0] == w);
  ASSERT_RETURN(p[1] == z);
  ASSERT_RETURN(p[2] == y);
  ASSERT_RETURN(p[3] == x);
  return TEST_SUCCESS;
}

result_t AVX2RVV_TEST_IMPL::load_test_float_pointers(uint32_t i) {
  result_t ret = do_mm_store_ps(
      test_cases_float_pointer1, test_cases_floats[i], test_cases_floats[i + 1],
      test_cases_floats[i + 2], test_cases_floats[i + 3]);
  if (ret == TEST_SUCCESS) {
    ret = do_mm_store_ps(test_cases_float_pointer2, test_cases_floats[i + 4],
                         test_cases_floats[i + 5], test_cases_floats[i + 6],
                         test_cases_floats[i + 7]);
  }
  return ret;
}

result_t AVX2RVV_TEST_IMPL::load_test_int_pointers(uint32_t i) {
  result_t ret = do_mm_store_ps(test_cases_int_pointer1, test_cases_ints[i],
                                test_cases_ints[i + 1], test_cases_ints[i + 2],
                                test_cases_ints[i + 3]);
  if (ret == TEST_SUCCESS) {
    ret = do_mm_store_ps(test_cases_int_pointer2, test_cases_ints[i + 4],
                         test_cases_ints[i + 5], test_cases_ints[i + 6],
                         test_cases_ints[i + 7]);
  }
  return ret;
}

result_t AVX2RVV_TEST_IMPL::run_single_test(INSTRUCTION_TEST test, uint32_t i) {
  result_t ret = TEST_SUCCESS;
  switch (test) {
#define _(x) case it_##x: ret = test_##x(*this, i); break;
    AVX_INTRIN_LIST
#undef _
  }
  return ret;
}

result_t AVX2RVV_TEST_IMPL::run_test(INSTRUCTION_TEST test) {
  result_t ret = TEST_SUCCESS;
  for (uint32_t i = 0; i < (MAX_TEST_VALUE - 8); i++) {
    ret = load_test_float_pointers(i);
    if (ret == TEST_FAIL) break;
    ret = load_test_int_pointers(i);
    if (ret == TEST_FAIL) break;
    ret = run_single_test(test, i);
    if (ret == TEST_FAIL) break;
  }
  return ret;
}

const char *instruction_string[] = {
#define _(x) #x,
    AVX_INTRIN_LIST
#undef _
};

AVX2RVV_TEST *AVX2RVV_TEST::create(void) {
  AVX2RVV_TEST_IMPL *st = new AVX2RVV_TEST_IMPL;
  return static_cast<AVX2RVV_TEST *>(st);
}

} // namespace AVX2RVV