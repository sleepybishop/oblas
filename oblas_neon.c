#include <arm_neon.h>

void oblas_axpy(uint8_t *a, const uint8_t *b, size_t k, const uint8_t *u_lo,
                const uint8_t *u_hi) {
  uint8_t *ap = a, *ae = a + k, *bp = (uint8_t *)b;
  uint8x16_t mask = vdupq_n_u8(0x0f);
  uint8x16_t urow_lo = vld1q_u8(u_lo);
  uint8x16_t urow_hi = vld1q_u8(u_hi);
  for (; ap < ae; ap += sizeof(uint8x16_t), bp += sizeof(uint8x16_t)) {
    uint8x16_t bx = vld1q_u8(bp);
    uint8x16_t lo = vandq_u8(bx, mask);
    bx = vshrq_n_u8(bx, 4);
    uint8x16_t hi = vandq_u8(bx, mask);
    lo = vqtbl1q_u8(urow_lo, lo);
    hi = vqtbl1q_u8(urow_hi, hi);
    uint8x16_t ux = veorq_u8(lo, hi);
    uint8x16_t ax = vld1q_u8(ap);
    vst1q_u8(ap, veorq_u8(ux, ax));
  }
}

void oblas_scal(uint8_t *a, size_t k, const uint8_t *u_lo,
                const uint8_t *u_hi) {
  uint8_t *ap = a, *ae = a + k;
  uint8x16_t mask = vdupq_n_u8(0x0f);
  uint8x16_t urow_lo = vld1q_u8(u_lo);
  uint8x16_t urow_hi = vld1q_u8(u_hi);
  for (; ap < ae; ap += sizeof(uint8x16_t)) {
    uint8x16_t ax = vld1q_u8(ap);
    uint8x16_t lo = vandq_u8(ax, mask);
    ax = vshrq_n_u8(ax, 4);
    uint8x16_t hi = vandq_u8(ax, mask);
    lo = vqtbl1q_u8(urow_lo, lo);
    hi = vqtbl1q_u8(urow_hi, hi);
    vst1q_u8(ap, veorq_u8(lo, hi));
  }
}

void oblas_axpy_gf2_gf256_32(uint8_t *a, uint32_t *b, size_t k, uint8_t u) {
  for (size_t idx = 0, p = 0; idx < k; idx += 32, p++) {
    uint32_t tmp = b[p];
    while (tmp > 0) {
      int tz = __builtin_ctz(tmp);
      tmp = tmp & (tmp - 1);
      a[tz + idx] ^= u;
    }
  }
}
