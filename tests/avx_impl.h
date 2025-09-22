/**
 * @file avx_impl.h
 * AVX512 instruction testing framework header
 * @author AVX2RVV Project
 * @date 2024
 * 
 * This header defines the testing framework for AVX512 intrinsics implemented
 * using RISC-V Vector Extension (RVV) instructions.
 */

#ifndef AVX2RVV_TEST_H
#define AVX2RVV_TEST_H

#include "common.h"
#include "debug_tools.h"

/**
 * Comprehensive list of AVX512 intrinsics to be tested
 * 
 * This macro defines all the AVX512 intrinsics that are implemented and tested
 * in this framework. Each intrinsic is organized by functional category:
 * - Basic operations (load/store)
 * - Arithmetic operations
 * - Comparison operations
 * - Min/Max operations
 * - Shuffle and permute operations
 * - Shift operations
 * - Type conversion operations
 * - Mask operations
 * - And many more...
 */
#define AVX_INTRIN_LIST                                                        \
    /* MMX */                                                                  \
    _(mm_empty11)                                                              \
    /* AVX512 Basic Operations */                                              \
    _(mm512_setzero_si512)                                                     \
    _(mm512_loadu_epi16)                                                       \
    _(mm512_storeu_epi16)                                                      \
    _(mm512_loadu_epi8)                                                        \
    _(mm512_storeu_epi8)                                                       \
    _(mm512_mask_mov_epi16)                                                    \
    _(mm512_maskz_mov_epi16)                                                   \
    _(mm512_mask_mov_epi8)                                                     \
    _(mm512_maskz_mov_epi8)                                                    \
    /* AVX512 Arithmetic */                                                    \
    _(mm512_add_epi8)                                                          \
    _(mm512_add_epi16)                                                         \
    _(mm512_sub_epi8)                                                          \
    _(mm512_sub_epi16)                                                         \
    _(mm512_avg_epu8)                                                          \
    _(mm512_avg_epu16)                                                         \
    /* AVX512 Comparison */                                                    \
    _(mm512_cmpeq_epi8_mask)                                                   \
    _(mm512_cmpeq_epi16_mask)                                                  \
    _(mm512_cmpgt_epi8_mask)                                                   \
    _(mm512_cmpgt_epi16_mask)                                                  \
    /* AVX512 Min/Max */                                                       \
    _(mm512_min_epi8)                                                          \
    _(mm512_max_epi8)                                                          \
    _(mm512_min_epi16)                                                         \
    _(mm512_max_epi16)                                                         \
    _(mm512_min_epu8)                                                          \
    _(mm512_max_epu8)                                                          \
    _(mm512_min_epu16)                                                         \
    _(mm512_max_epu16)                                                         \
    /* AVX512 Shuffle */                                                       \
    _(mm512_shuffle_epi8)                                                      \
    _(mm512_shufflehi_epi16)                                                   \
    _(mm512_shufflelo_epi16)                                                   \
    /* AVX512 Shift */                                                         \
    _(mm512_slli_epi16)                                                        \
    _(mm512_srli_epi16)                                                        \
    _(mm512_srai_epi16)                                                        \
    /* AVX512 Type Conversion */                                               \
    _(mm512_cvtepi16_epi8)                                                     \
    _(mm512_cvtepi8_epi16)                                                     \
    _(mm512_cvtepu8_epi16)                                                     \
    /* AVX512 Permute */                                                       \
    _(mm512_permutexvar_epi16)                                                 \
    /* AVX512 Mask Operations */                                               \
    _(mm512_movepi8_mask)                                                      \
    _(mm512_movepi16_mask)                                                     \
    _(mm512_movm_epi8)                                                         \
    _(mm512_movm_epi16)                                                        \
    /* AVX512 Test Operations */                                               \
    _(mm512_test_epi8_mask)                                                    \
    _(mm512_test_epi16_mask)                                                   \
    /* AVX512 Unpack */                                                        \
    _(mm512_unpackhi_epi8)                                                     \
    _(mm512_unpackhi_epi16)                                                    \
    /* AVX512 Multiply */                                                      \
    _(mm512_mullo_epi16)                                                       \
    _(mm512_mulhi_epi16)                                                       \
    _(mm512_mulhi_epu16)                                                       \
    _(mm512_mulhrs_epi16)                                                      \
    /* AVX512 SAD */                                                           \
    _(mm512_sad_epu8)                                                          \
    /* AVX512 Pack */                                                          \
    _(mm512_packs_epi16)                                                       \
    /* AVX512 Align */                                                         \
    _(mm512_alignr_epi8)                                                       \
    /* AVX512 Abs */                                                           \
    _(mm512_abs_epi8)                                                          \
    _(mm512_abs_epi16)                                                         \
    /* AVX512 Saturating Operations */                                         \
    _(mm512_adds_epi8)                                                         \
    _(mm512_adds_epi16)                                                        \
    _(mm512_adds_epu8)                                                         \
    _(mm512_adds_epu16)                                                        \
    _(mm512_subs_epi8)                                                         \
    _(mm512_subs_epi16)                                                        \
    _(mm512_subs_epu8)                                                         \
    _(mm512_subs_epu16)                                                        \
    /* AVX512 Set Operations */                                                \
    _(mm512_set1_epi8)                                                         \
    _(mm512_set1_epi16)                                                        \
    _(mm512_mask_set1_epi8)                                                    \
    _(mm512_mask_set1_epi16)                                                   \
    _(mm512_maskz_set1_epi8)                                                   \
    _(mm512_maskz_set1_epi16)                                                  \
    /* AVX512 Blend */                                                         \
    _(mm512_mask_blend_epi8)                                                   \
    _(mm512_mask_blend_epi16)                                                  \
    /* AVX512 Mask Load/Store */                                               \
    _(mm512_mask_loadu_epi8)                                                   \
    _(mm512_mask_loadu_epi16)                                                  \
    _(mm512_maskz_loadu_epi8)                                                  \
    _(mm512_maskz_loadu_epi16)                                                 \
    _(mm512_mask_storeu_epi8)                                                  \
    _(mm512_mask_storeu_epi16)                                                 \
    /* AVX512 Mask Operations */                                               \
    _(mm512_kunpackd)                                                          \
    _(mm512_kunpackw)                                                          \
    /* Utility */                                                              \
    _(rdtsc)                                                                   \
    _(last) /* This indicates the end of macros */

/**
 * AVX2RVV namespace containing all testing framework components
 */
namespace AVX2RVV {
using SSE2RVV::result_t;

/**
 * Comprehensive testing methodology
 * 
 * The unit testing framework generates 10,000 random floating point and
 * integer vectors as test data. Each AVX512 intrinsic is implemented using
 * RISC-V Vector Extension instructions and validated against expected results.
 * 
 * When running on RISC-V platforms, results are compared against the RVV
 * implementation. On x86 platforms, tests return TEST_UNIMPL for unimplemented
 * intrinsics.
 */

extern const char *instruction_string[]; ///< Array of instruction names for reporting

/**
 * Enumeration of all testable AVX512 instructions
 */
enum INSTRUCTION_TEST {
#define _(x) it_##x,
    AVX_INTRIN_LIST
#undef _
};

/**
 * Abstract base class for AVX512 testing framework
 * 
 * This class provides the interface for running comprehensive tests on
 * AVX512 intrinsics implemented using RISC-V Vector Extension instructions.
 */
class AVX2RVV_TEST {
public:
    /**
     * Factory method to create a test instance
     * @return Pointer to new test instance
     */
    static AVX2RVV_TEST *create(void);

    /**
     * Run comprehensive test for a specific instruction
     * @param test The instruction test to run
     * @return Test result:
     *         - TEST_SUCCESS (1): Test passed
     *         - TEST_FAIL (0): Test failed
     *         - TEST_UNIMPL (-1): Not implemented on this platform
     */
    virtual result_t run_test(INSTRUCTION_TEST test) = 0;
    
    /**
     * Release the test instance and free resources
     */
    virtual void release(void) = 0;
};

} // namespace AVX2RVV

#endif // AVX2RVV_TEST_H
