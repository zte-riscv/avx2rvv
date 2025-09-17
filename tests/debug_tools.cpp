#include "debug_tools.h"
#include <cmath>
#include <cstdint>

namespace SSE2RVV {

#if defined(__clang__)
#define INT64_ESCAPE "%20lld"
#define UINT64_ESCAPE "%20llu"
#else
#define INT64_ESCAPE "%20ld"
#define UINT64_ESCAPE "%20lu"
#endif

void print_64_bits_u8_arr(const char *var_name, const uint8_t *u) {
  printf("%s0: %3u, %s1: %3u, %s2: %3u, %s3: %3u, %s4: %3u, %s5: %3u, %s6: "
         "%3u, %s7: %3u\n",
         var_name, u[0], var_name, u[1], var_name, u[2], var_name, u[3],
         var_name, u[4], var_name, u[5], var_name, u[6], var_name, u[7]);
}
void print_64_bits_s8_arr(const char *var_name, const int8_t *u) {
  printf("%s0: %3d, %s1: %3d, %s2: %3d, %s3: %3d, %s4: %3d, %s5: %3d, %s6: "
         "%3d, %s7: %3d\n",
         var_name, u[0], var_name, u[1], var_name, u[2], var_name, u[3],
         var_name, u[4], var_name, u[5], var_name, u[6], var_name, u[7]);
}
void print_64_bits_u16_arr(const char *var_name, const uint16_t *u) {
  printf("%s0: %5u, %s1: %5u, %s2: %5u, %s3: %5u\n", var_name, u[0], var_name,
         u[1], var_name, u[2], var_name, u[3]);
}
void print_64_bits_s16_arr(const char *var_name, const int16_t *u) {
  printf("%s0: %5d, %s1: %5d, %s2: %5d, %s3: %5d\n", var_name, u[0], var_name,
         u[1], var_name, u[2], var_name, u[3]);
}
void print_64_bits_u32_arr(const char *var_name, const uint32_t *u) {
  printf("%s0: %10u, %s1: %10u\n", var_name, u[0], var_name, u[1]);
}
void print_64_bits_s32_arr(const char *var_name, const int32_t *u) {
  printf("%s0: %10d, %s1: %10d\n", var_name, u[0], var_name, u[1]);
}
void print_64_bits_u64_arr(const char *var_name, const uint64_t *u) {
  printf("%s0: " UINT64_ESCAPE "\n", var_name, u[0]);
}
void print_64_bits_s64_arr(const char *var_name, const int64_t *u) {
  printf("%s0: " INT64_ESCAPE "\n", var_name, u[0]);
}
void print_64_bits_f32_arr(const char *var_name, const float *f) {
  printf("%s0: %.3f, %s1: %.3f\n", var_name, f[0], var_name, f[1]);
}
void print_64_bits_f64_arr(const char *var_name, const double *f) {
  printf("%s0: %.6f\n", var_name, f[0]);
}
void print_128_bits_u8_arr(const char *var_name, const uint8_t *u) {
  printf("%s0: %3u, %s1: %3u, %s2: %3u, %s3: %3u, %s4: %3u, %s5: %3u, "
         "%s6: %3u, %s7: %3u, %s8: %3u, %s9: %3u, %s10: %3u, %s11: %3u, "
         "%s12: %3u, %s13: %3u, %s14: %3u, %s15: %3u\n",
         var_name, u[0], var_name, u[1], var_name, u[2], var_name, u[3],
         var_name, u[4], var_name, u[5], var_name, u[6], var_name, u[7],
         var_name, u[8], var_name, u[9], var_name, u[10], var_name, u[11],
         var_name, u[12], var_name, u[13], var_name, u[14], var_name, u[15]);
}
void print_128_bits_s8_arr(const char *var_name, const int8_t *u) {
  printf("%s0: %3d, %s1: %3d, %s2: %3d, %s3: %3d, %s4: %3d, %s5: %3d, "
         "%s6: %3d, %s7: %3d, %s8: %3d, %s9: %3d, %s10: %3d, %s11: %3d, "
         "%s12: %3d, %s13: %3d, %s14: %3d, %s15: %3d\n",
         var_name, u[0], var_name, u[1], var_name, u[2], var_name, u[3],
         var_name, u[4], var_name, u[5], var_name, u[6], var_name, u[7],
         var_name, u[8], var_name, u[9], var_name, u[10], var_name, u[11],
         var_name, u[12], var_name, u[13], var_name, u[14], var_name, u[15]);
}
void print_128_bits_u16_arr(const char *var_name, const uint16_t *u) {
  printf("%s0: %5u, %s1: %5u, %s2: %5u, %s3: %5u, %s4: %5u, %s5: %5u, %s6: "
         "%5u, %s7: %5u\n",
         var_name, u[0], var_name, u[1], var_name, u[2], var_name, u[3],
         var_name, u[4], var_name, u[5], var_name, u[6], var_name, u[7]);
}
void print_128_bits_s16_arr(const char *var_name, const int16_t *u) {
  printf("%s0: %5d, %s1: %5d, %s2: %5d, %s3: %5d, %s4: %5d, %s5: %5d, %s6: "
         "%5d, %s7: %5d\n",
         var_name, u[0], var_name, u[1], var_name, u[2], var_name, u[3],
         var_name, u[4], var_name, u[5], var_name, u[6], var_name, u[7]);
}
void print_128_bits_u32_arr(const char *var_name, const uint32_t *u) {
  printf("%s0: %10u, %s1: %10u, %s2: %10u, %s3: %10u\n", var_name, u[0],
         var_name, u[1], var_name, u[2], var_name, u[3]);
}
void print_128_bits_s32_arr(const char *var_name, const int32_t *u) {
  printf("%s0: %10d, %s1: %10d, %s2: %10d, %s3: %10d\n", var_name, u[0],
         var_name, u[1], var_name, u[2], var_name, u[3]);
}
void print_128_bits_u64_arr(const char *var_name, const uint64_t *u) {
  printf("%s0: " UINT64_ESCAPE ", %s1: " UINT64_ESCAPE "\n", var_name, u[0],
         var_name, u[1]);
}
void print_128_bits_s64_arr(const char *var_name, const int64_t *u) {
  printf("%s0: " INT64_ESCAPE ", %s1: " INT64_ESCAPE "\n", var_name, u[0],
         var_name, u[1]);
}
void print_128_bits_f32_arr(const char *var_name, const float *f) {
  printf("%s0: %.3f, %s1: %.3f, %s2: %.3f, %s3: %.3f\n", var_name, f[0],
         var_name, f[1], var_name, f[2], var_name, f[3]);
}
void print_128_bits_f64_arr(const char *var_name, const double *f) {
  printf("%s0: %.6f, %s1: %.6f\n", var_name, f[0], var_name, f[1]);
}

void print_u8_64(const char *var_name, uint8_t u0, uint8_t u1, uint8_t u2,
                 uint8_t u3, uint8_t u4, uint8_t u5, uint8_t u6, uint8_t u7) {
  uint8_t a[] = {u0, u1, u2, u3, u4, u5, u6, u7};
  const uint8_t *u = (const uint8_t *)&a;
  print_64_bits_u8_arr(var_name, u);
}
void print_u8_64(const char *var_name, int8_t i0, int8_t i1, int8_t i2,
                 int8_t i3, int8_t i4, int8_t i5, int8_t i6, int8_t i7) {
  int8_t a[] = {i0, i1, i2, i3, i4, i5, i6, i7};
  const uint8_t *u = (const uint8_t *)&a;
  print_64_bits_u8_arr(var_name, u);
}
void print_u8_64(const char *var_name, uint16_t u0, uint16_t u1, uint16_t u2,
                 uint16_t u3) {
  uint16_t a[] = {u0, u1, u2, u3};
  const uint8_t *u = (const uint8_t *)&a;
  print_64_bits_u8_arr(var_name, u);
}
void print_u8_64(const char *var_name, int16_t i0, int16_t i1, int16_t i2,
                 int16_t i3) {
  int16_t a[] = {i0, i1, i2, i3};
  const uint8_t *u = (const uint8_t *)&a;
  print_64_bits_u8_arr(var_name, u);
}
void print_u8_64(const char *var_name, uint32_t u0, uint32_t u1) {
  uint32_t a[] = {u0, u1};
  const uint8_t *u = (const uint8_t *)&a;
  print_64_bits_u8_arr(var_name, u);
}
void print_u8_64(const char *var_name, int32_t i0, int32_t i1) {
  int32_t a[] = {i0, i1};
  const uint8_t *u = (const uint8_t *)&a;
  print_64_bits_u8_arr(var_name, u);
}
void print_u8_64(const char *var_name, uint64_t u0) {
  uint64_t a[] = {u0};
  const uint8_t *u = (const uint8_t *)&a;
  print_64_bits_u8_arr(var_name, u);
}
void print_u8_64(const char *var_name, int64_t i0) {
  int64_t a[] = {i0};
  const uint8_t *u = (const uint8_t *)&a;
  print_64_bits_u8_arr(var_name, u);
}

void print_u8_128(const char *var_name, uint8_t u0, uint8_t u1, uint8_t u2,
                  uint8_t u3, uint8_t u4, uint8_t u5, uint8_t u6, uint8_t u7,
                  uint8_t u8, uint8_t u9, uint8_t u10, uint8_t u11, uint8_t u12,
                  uint8_t u13, uint8_t u14, uint8_t u15) {
  uint8_t a[] = {u0, u1, u2,  u3,  u4,  u5,  u6,  u7,
                 u8, u9, u10, u11, u12, u13, u14, u15};
  const uint8_t *u = (const uint8_t *)&a;
  print_128_bits_u8_arr(var_name, u);
}
void print_u8_128(const char *var_name, int8_t i0, int8_t i1, int8_t i2,
                  int8_t i3, int8_t i4, int8_t i5, int8_t i6, int8_t i7,
                  int8_t i8, int8_t i9, int8_t i10, int8_t i11, int8_t i12,
                  int8_t i13, int8_t i14, int8_t i15) {
  int8_t a[] = {i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                i8, i9, i10, i11, i12, i13, i14, i15};
  const uint8_t *u = (const uint8_t *)&a;
  print_128_bits_u8_arr(var_name, u);
}
void print_u8_128(const char *var_name, uint16_t u0, uint16_t u1, uint16_t u2,
                  uint16_t u3, uint16_t u4, uint16_t u5, uint16_t u6,
                  uint16_t u7) {
  uint16_t a[] = {u0, u1, u2, u3, u4, u5, u6, u7};
  const uint8_t *u = (const uint8_t *)&a;
  print_128_bits_u8_arr(var_name, u);
}
void print_u8_128(const char *var_name, int16_t i0, int16_t i1, int16_t i2,
                  int16_t i3, int16_t i4, int16_t i5, int16_t i6, int16_t i7) {
  int16_t a[] = {i0, i1, i2, i3, i4, i5, i6, i7};
  const uint8_t *u = (const uint8_t *)&a;
  print_128_bits_u8_arr(var_name, u);
}
void print_u8_128(const char *var_name, uint32_t u0, uint32_t u1, uint32_t u2,
                  uint32_t u3) {
  uint32_t a[] = {u0, u1, u2, u3};
  const uint8_t *u = (const uint8_t *)&a;
  print_128_bits_u8_arr(var_name, u);
}
void print_u8_128(const char *var_name, int32_t i0, int32_t i1, int32_t i2,
                  int32_t i3) {
  int32_t a[] = {i0, i1, i2, i3};
  const uint8_t *u = (const uint8_t *)&a;
  print_128_bits_u8_arr(var_name, u);
}
void print_u8_128(const char *var_name, uint64_t u0, uint64_t u1) {
  uint64_t a[] = {u0, u1};
  const uint8_t *u = (const uint8_t *)&a;
  print_128_bits_u8_arr(var_name, u);
}
void print_u8_128(const char *var_name, int64_t i0, int64_t i1) {
  int64_t a[] = {i0, i1};
  const uint8_t *u = (const uint8_t *)&a;
  print_128_bits_u8_arr(var_name, u);
}

} // namespace SSE2RVV
