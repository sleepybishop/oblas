#include <arm_neon.h>

#include "oblas.h"
#include "octmul_hilo.h"

/*
 * AArch32 does not provide this intrinsic natively because it does not
 * implement the underlying instruction. AArch32 only provides a 64-bit
 * wide vtbl.8 instruction, so use that instead.
 */

#ifndef vqtbl1q_u8
static uint8x16_t vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
  union {
    uint8x16_t val;
    uint8x8x2_t pair;
  } __a = {a};

  return vcombine_u8(vtbl2_u8(__a.pair, vget_low_u8(b)),
                     vtbl2_u8(__a.pair, vget_high_u8(b)));
}
#endif

void ocopy(uint8_t *restrict a, uint8_t *restrict b, size_t i, size_t j,
           size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  for (size_t idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    vst1q_u8(ap + idx, vld1q_u8(bp + idx));
  }
}

void oswaprow(uint8_t *restrict a, size_t i, size_t j, size_t k) {
  if (i == j)
    return;
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = a + (j * ALIGNED_COLS(k));

  for (size_t idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    uint8x16_t atmp = vld1q_u8(ap + idx);
    uint8x16_t btmp = vld1q_u8(bp + idx);

    vst1q_u8(ap + idx, btmp);
    vst1q_u8(bp + idx, atmp);
  }
}

void oswapcol(octet *restrict a, size_t i, size_t j, size_t k, size_t l) {
  if (i == j)
    return;
  octet *ap = a;

  for (size_t idx = 0; idx < k; idx++, ap += ALIGNED_COLS(l)) {
    OCTET_SWAP(ap[i], ap[j]);
  }
}

void oaxpy(uint8_t *restrict a, uint8_t *restrict b, size_t i, size_t j,
           size_t k, uint8_t u) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  if (u == 0)
    return;

  if (u == 1)
    return oaddrow(a, b, i, j, k);

  uint8x16_t mask = vdupq_n_u8(0x0f);
  uint8x16_t urow_hi = vld1q_u8(OCT_MUL_HI[u]);
  uint8x16_t urow_lo = vld1q_u8(OCT_MUL_LO[u]);
  for (size_t idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    uint8x16_t bx = vld1q_u8(bp + idx);
    uint8x16_t lo = vandq_u8(bx, mask);
    bx = vshrq_n_u8(bx, 4);
    uint8x16_t hi = vandq_u8(bx, mask);
    lo = vqtbl1q_u8(urow_lo, lo);
    hi = vqtbl1q_u8(urow_hi, hi);
    uint8x16_t ux = veorq_u8(lo, hi);
    uint8x16_t ax = vld1q_u8(ap + idx);
    vst1q_u8(ap + idx, veorq_u8(ux, ax));
  }
}

void oaddrow(uint8_t *restrict a, uint8_t *restrict b, size_t i, size_t j,
             size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  for (size_t idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    uint8x16_t ap128 = vld1q_u8(ap + idx);
    uint8x16_t bp128 = vld1q_u8(bp + idx);

    vst1q_u8(ap + idx, veorq_u8(ap128, bp128));
  }
}

void oscal(uint8_t *restrict a, size_t i, size_t k, uint8_t u) {
  octet *ap = a + (i * ALIGNED_COLS(k));

  if (u < 2)
    return;

  uint8x16_t mask = vdupq_n_u8(0x0f);
  uint8x16_t urow_hi = vld1q_u8(OCT_MUL_HI[u]);
  uint8x16_t urow_lo = vld1q_u8(OCT_MUL_LO[u]);
  for (size_t idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    uint8x16_t ax = vld1q_u8(ap + idx);
    uint8x16_t lo = vandq_u8(ax, mask);
    ax = vshrq_n_u8(ax, 4);
    uint8x16_t hi = vandq_u8(ax, mask);
    lo = vqtbl1q_u8(urow_lo, lo);
    hi = vqtbl1q_u8(urow_hi, hi);
    vst1q_u8(ap + idx, veorq_u8(lo, hi));
  }
}

void ozero(uint8_t *restrict a, size_t i, size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));

  uint8x16_t z128 = vdupq_n_u8(0);
  for (size_t idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    vst1q_u8(ap + idx, z128);
  }
}

void ogemm(uint8_t *restrict a, uint8_t *restrict b, uint8_t *restrict c,
           size_t n, size_t k, size_t m) {
  octet *ap, *cp = c;

  for (size_t row = 0; row < n; row++, cp += ALIGNED_COLS(m)) {
    ap = a + (row * ALIGNED_COLS(k));

    ozero(cp, 0, m);
    for (size_t idx = 0; idx < k; idx++) {
      oaxpy(cp, b, 0, idx, m, ap[idx]);
    }
  }
}

size_t onnz(uint8_t *a, size_t i, size_t s, size_t e, size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  size_t nz = 0;
  for (size_t idx = s; idx < e; idx++) {
    nz += (ap[idx] != 0);
  }
  return nz;
}

void oaxpy_b32(uint8_t *a, uint32_t *b, size_t i, size_t k, uint8_t u) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  for (size_t idx = 0, p = 0; idx < k; idx += 8 * sizeof(uint32_t), p++) {
    uint32_t tmp = b[p];
    while (tmp > 0) {
      int tz = __builtin_ctz(tmp);
      tmp = tmp & (tmp - 1);
      ap[tz + idx] ^= u;
    }
  }
}
