#ifndef AVX2RVV_H
#define AVX2RVV_H

#if defined(__riscv) || defined(__riscv__)
#include <riscv_vector.h>
#define AVX2RVV_IMPLEMENTATION
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
// On x86 platforms, use native intrinsics, don't implement our own
#endif

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


/* Compiler adaptation */
#ifdef __GNUC__
#define FORCE_INLINE static inline __attribute__((always_inline))
#define ALIGN_STRUCT(x) __attribute__((aligned(x)))
#else
#define FORCE_INLINE static inline
#define ALIGN_STRUCT(x)
#endif

/* Platform type adaptation */
#if !(defined(_WIN32) || defined(_WIN64) || defined(__int64))
    #if (defined(__x86_64__) || defined(__i386__))
        #define _int64 long long
    #else
        #define _int64 int64_t
    #endif
#endif

/* ===== Type mapping ===== */
#ifdef AVX2RVV_IMPLEMENTATION
typedef vuint8m8_t  __m512u;     /* 512-bit unsigned integer vector */
typedef vfloat32m4_t __m512f;    /* 512-bit float vector (16 32-bit floats) */
typedef vfloat64m2_t __m512d;    /* 512-bit double precision vector (8 64-bit floats) */
typedef uint8_t __mmask8;
typedef uint16_t __mmask16;
typedef uint32_t __mmask32;
typedef uint64_t __mmask64;
typedef vfloat32m2_t __m256f;    /* 256-bit float vector (8 32-bit floats) */
typedef vfloat32m1_t __m128f;    /* 128-bit float vector (4 32-bit floats) */
typedef union {
    uint8_t u8[64] __attribute__((aligned(64)));
    uint16_t u16[32];
    uint32_t u32[16];
    uint64_t u64[8];
    int8_t i8[64];
    int16_t i16[32];
    int32_t i32[16];
    int64_t i64[8];
} __m512i;

typedef union {
    uint8_t u8[32] __attribute__((aligned(32)));
    uint16_t u16[16];
    uint32_t u32[8];
    uint64_t u64[4];
    int8_t  i8[32];
    int16_t i16[16];
    int32_t i32[8];
    int64_t i64[4];
} __m256i;
#endif

#define _MM_ROUND_TO_NEAREST_INT 0x00
#define _MM_ROUND_TO_NEG_INF     0x01
#define _MM_ROUND_TO_POS_INF     0x02
#define _MM_ROUND_TO_ZERO        0x03
#define _MM_ROUND_CUR_DIRECTION  0x04
#define _MM_ROUND_NO_EXC         0x08
#define _MM_ROUND_RAISE_EXC      0x00

#ifndef __riscv_frm_rne
#define __riscv_frm_rne 0
#define __riscv_frm_rdn 1
#define __riscv_frm_rup 2
#define __riscv_frm_rtz 3
#endif

#ifndef _MM_FROUND_TO_NEAREST_INT
#define _MM_FROUND_TO_NEAREST_INT 0x0
#define _MM_FROUND_TO_NEG_INF     0x1
#define _MM_FROUND_TO_POS_INF     0x2
#define _MM_FROUND_TO_ZERO        0x3
#endif

// #define _MM_FROUND_CUR_DIRECTION  0x4
// #define _MM_FROUND_NO_EXC 0x8
/* Map to RVV rounding mode (FRM field) */
FORCE_INLINE uint8_t aux512_round_mode_to_rvv(uint8_t aux_mode) {
    switch(aux_mode & 0x03) {
        case _MM_ROUND_TO_NEAREST_INT: return __riscv_frm_rne; /* Round to nearest even */
        case _MM_ROUND_TO_NEG_INF:     return __riscv_frm_rdn; /* Round down */
        case _MM_ROUND_TO_POS_INF:     return __riscv_frm_rup; /* Round up */
        case _MM_ROUND_TO_ZERO:        return __riscv_frm_rtz; /* Round to zero */
        default: return __riscv_frm_rne;
    }
}

FORCE_INLINE __m512i _mm512_loadu_epi8(const void* mem_addr) {
    __m512i result;
    size_t vl = __riscv_vsetvl_e8m1(64);
    
    vint8m1_t vec = __riscv_vle8_v_i8m1((const int8_t*)mem_addr, vl);
    
    __riscv_vse8_v_i8m1((int8_t*)result.u8, vec, vl);
    
    return result;
}

FORCE_INLINE __m512i _mm512_loadu_epi16(const void* mem_addr) {
    size_t vl = __riscv_vsetvlmax_e16m4();
    vint16m4_t v = __riscv_vle16_v_i16m4((const short int*)mem_addr, vl);
    
    int16_t buffer[32] __attribute__((aligned(64)));
    __riscv_vse16_v_i16m4(buffer, v, vl);
    return *(__m512i*)buffer;
}

FORCE_INLINE void _mm512_storeu_epi8(void* mem_addr, __m512i a) {
    for (int i = 0; i < 64; i += 8) {
        size_t vl = __riscv_vsetvl_e8m1(8);
        vint8m1_t vec = *(vint8m1_t*)((int8_t*)&a + i);
        __riscv_vse8_v_i8m1((int8_t*)mem_addr + i, vec, vl);
    }
}

FORCE_INLINE void _mm512_storeu_epi16(void *mem_addr, __m512i a) {
    size_t vl = __riscv_vsetvl_e16m1(8);
    int16_t *dst = (int16_t*)mem_addr;

    for (int i = 0; i < 4; i++) {
        vint16m1_t vec = __riscv_vle16_v_i16m1((const short int*)&a.u16[i*8], vl);
        __riscv_vse16_v_i16m1(dst + i*8, vec, vl);
    }
}

FORCE_INLINE __m512i _mm512_add_epi16(__m512i a, __m512i b) {
    __m512i result;
    size_t vl = __riscv_vsetvl_e16m4(32);
    vint16m4_t va = __riscv_vle16_v_i16m4(a.i16, vl);
    vint16m4_t vb = __riscv_vle16_v_i16m4(b.i16, vl);
    vint16m4_t vr = __riscv_vadd_vv_i16m4(va, vb, vl);
    __riscv_vse16_v_i16m4(result.i16, vr, vl);
    return result;
}

static inline __m512i _mm512_sub_epi16(__m512i a, __m512i b) {
    size_t vl = __riscv_vsetvl_e16m8(32);

    vint16m8_t va = __riscv_vle16_v_i16m8((const int16_t*)a.u16, vl);
    vint16m8_t vb = __riscv_vle16_v_i16m8((const int16_t*)b.u16, vl);

    vint16m8_t vsub = __riscv_vsub_vv_i16m8(va, vb, vl);

    __m512i result;
    __riscv_vse16_v_i16m8((int16_t*)result.u16, vsub, vl);
    return result;
}

FORCE_INLINE __m512i _mm512_avg_epu16(__m512i a, __m512i b) {
    size_t vl = __riscv_vsetvl_e16m8(32);

    vuint16m8_t va = __riscv_vle16_v_u16m8((const uint16_t*)&a, vl);
    vuint16m8_t vb = __riscv_vle16_v_u16m8((const uint16_t*)&b, vl);

    vuint16m8_t vsum = __riscv_vadd_vv_u16m8(va, vb, vl);
    vuint16m8_t vone = __riscv_vmv_v_x_u16m8(1, vl);
    vsum = __riscv_vadd_vv_u16m8(vsum, vone, vl);
    vuint16m8_t vavg = __riscv_vsrl_vx_u16m8(vsum, 1, vl);

    __m512i result;
    __riscv_vse16_v_u16m8((uint16_t*)&result, vavg, vl);

    return result;
}

FORCE_INLINE __mmask32 _mm512_cmpeq_epi16_mask(__m512i a, __m512i b) {
    size_t vl = __riscv_vsetvl_e16m4(32);

    vint16m4_t va = __riscv_vle16_v_i16m4((const int16_t*)&a, vl);
    vint16m4_t vb = __riscv_vle16_v_i16m4((const int16_t*)&b, vl);

    vbool4_t vcmp = __riscv_vmseq_vv_i16m4_b4(va, vb, vl);

    uint8_t mask_arr[4] = {0};
    __riscv_vsm_v_b4(mask_arr, vcmp, vl);

    return *(uint32_t*)mask_arr;
}

FORCE_INLINE __mmask32 _mm512_cmpgt_epi16_mask(__m512i a, __m512i b) {
    size_t vl = __riscv_vsetvl_e16m4(32);

    vint16m4_t va = __riscv_vle16_v_i16m4((const int16_t*)&a, vl);
    vint16m4_t vb = __riscv_vle16_v_i16m4((const int16_t*)&b, vl);

    vbool4_t vcmp = __riscv_vmslt_vv_i16m4_b4(vb, va, vl);

    uint8_t mask_arr[4] = {0};
    __riscv_vsm_v_b4(mask_arr, vcmp, vl);

    return *(uint32_t*)mask_arr;
}

FORCE_INLINE __m512i _mm512_min_epi16(__m512i a, __m512i b) {
    size_t vl = __riscv_vsetvl_e16m8(32);

    vint16m8_t va = __riscv_vle16_v_i16m8(a.i16, vl);
    vint16m8_t vb = __riscv_vle16_v_i16m8(b.i16, vl);

    vint16m8_t vmin = __riscv_vmin_vv_i16m8(va, vb, vl);

    __m512i result;
    __riscv_vse16_v_i16m8(result.i16, vmin, vl);
    return result;
}

FORCE_INLINE __m512i _mm512_max_epi16(__m512i a, __m512i b) {
    size_t vl = __riscv_vsetvl_e16m8(32);

    vint16m8_t va = __riscv_vle16_v_i16m8((const int16_t*)a.i16, vl);
    vint16m8_t vb = __riscv_vle16_v_i16m8((const int16_t*)b.i16, vl);

    vint16m8_t vmax = __riscv_vmax_vv_i16m8(va, vb, vl);

    __m512i dst;
    __riscv_vse16_v_i16m8((int16_t*)dst.i16, vmax, vl);
    return dst;
}

FORCE_INLINE __m512i _mm512_mask_min_epu8(__m512i src, __mmask64 k, __m512i a, __m512i b) {
    size_t vl = __riscv_vsetvl_e8m8(64);

    vuint8m8_t vsrc = __riscv_vle8_v_u8m8(src.u8, vl);
    vuint8m8_t va = __riscv_vle8_v_u8m8(a.u8, vl);
    vuint8m8_t vb = __riscv_vle8_v_u8m8(b.u8, vl);

    uint8_t mask_arr[64];
    for (int i = 0; i < 64; i++) {
        mask_arr[i] = (k & (1ULL << i)) ? 0xFF : 0x00;
    }
    vbool1_t mask = __riscv_vmseq_vx_u8m8_b1(
        __riscv_vle8_v_u8m8(mask_arr, vl), 0xFF, vl);

    vuint8m8_t vmin = __riscv_vminu_vv_u8m8(va, vb, vl);

    vuint8m8_t result = __riscv_vmerge_vvm_u8m8(vsrc, vmin, mask, vl);

    __m512i dst;
    __riscv_vse8_v_u8m8(dst.u8, result, vl);
    return dst;
}


FORCE_INLINE __m512i _mm512_min_epu16(__m512i a, __m512i b) {
    size_t vl = __riscv_vsetvl_e16m8(32);

    vuint16m8_t va = __riscv_vle16_v_u16m8((const uint16_t*)&a, vl);
    vuint16m8_t vb = __riscv_vle16_v_u16m8((const uint16_t*)&b, vl);

    vuint16m8_t vmin = __riscv_vminu_vv_u16m8(va, vb, vl);

    __m512i result;
    __riscv_vse16_v_u16m8((uint16_t*)&result, vmin, vl);
    return result;
}

FORCE_INLINE __m512i _mm512_mask_min_epu16(__m512i src, __mmask32 k, __m512i a, __m512i b) {
    size_t vl = __riscv_vsetvl_e16m8(32);
    
    vuint16m8_t vsrc = __riscv_vle16_v_u16m8((const uint16_t*)&src, vl);
    vuint16m8_t va = __riscv_vle16_v_u16m8((const uint16_t*)&a, vl);
    vuint16m8_t vb = __riscv_vle16_v_u16m8((const uint16_t*)&b, vl);
    
    uint16_t mask_arr[32];
    for (int i = 0; i < 32; i++) {
        mask_arr[i] = (k & (1 << i)) ? 0xFFFF : 0x0000;
    }
    vbool2_t mask = __riscv_vmseq_vx_u16m8_b2(
        __riscv_vle16_v_u16m8(mask_arr, vl), 0xFFFF, vl);
    
    vuint16m8_t vmin = __riscv_vminu_vv_u16m8(va, vb, vl);
    
    vuint16m8_t result = __riscv_vmerge_vvm_u16m8(
        vsrc,
        vmin,
        mask,
        vl
    );
    
    __m512i dst;
    __riscv_vse16_v_u16m8((uint16_t*)&dst, result, vl);
    return dst;
}

FORCE_INLINE __m512i _mm512_max_epu16(__m512i a, __m512i b) {
    size_t vl = __riscv_vsetvl_e16m8(32);

    vuint16m8_t va = __riscv_vle16_v_u16m8((const uint16_t*)a.u16, vl);
    vuint16m8_t vb = __riscv_vle16_v_u16m8((const uint16_t*)b.u16, vl);

    vuint16m8_t vmax = __riscv_vmaxu_vv_u16m8(va, vb, vl);

    __m512i dst;
    __riscv_vse16_v_u16m8((uint16_t*)dst.u16, vmax, vl);
    return dst;
}

FORCE_INLINE __m512i _mm512_setzero_si512(void) {
    __m512i result;
    size_t vl = __riscv_vsetvl_e8m8(64);
    vint8m8_t zero_vec = __riscv_vmv_v_x_i8m8(0, vl);
    __riscv_vse8_v_i8m8((int8_t*)&result, zero_vec, vl);
    return result;
}

FORCE_INLINE void _mm512_storeu_si512(void* mem_addr, __m512i a) {
    size_t vl = __riscv_vsetvl_e64m8(8);
    __riscv_vse64_v_i64m8((int64_t*)mem_addr, __riscv_vle64_v_i64m8(a.i64, vl), vl);
}

FORCE_INLINE __m512i _mm512_loadu_si512(void const* mem_addr) {
    size_t vl = __riscv_vsetvl_e64m8(8);
    vint64m8_t vec = __riscv_vle64_v_i64m8((int64_t const*)mem_addr, vl);
    __m512i result;
    __riscv_vse64_v_i64m8(result.i64, vec, vl);
    return result;
}
#endif 
