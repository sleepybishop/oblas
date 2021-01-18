#include "oblas.h"
#include "oblas_compat.h"
#include <errno.h>

void ocopy(uint8_t *restrict a, uint8_t *restrict b, size_t i, size_t j,
           size_t k) {
  uint8_t *ap = a + (i * ALIGNED_COLS(k));
  uint8_t *bp = b + (j * ALIGNED_COLS(k));
  oblas_copy(ap, bp, k);
}

void oswaprow(uint8_t *restrict a, size_t i, size_t j, size_t k) {
  if (i == j)
    return;
  uint8_t *ap = a + (i * ALIGNED_COLS(k));
  uint8_t *bp = a + (j * ALIGNED_COLS(k));
  oblas_swap(ap, bp, k);
}

void oaxpy(uint8_t *restrict a, uint8_t *restrict b, size_t i, size_t j,
           size_t k, uint8_t u) {
  uint8_t *ap = a + (i * ALIGNED_COLS(k));
  uint8_t *bp = b + (j * ALIGNED_COLS(k));

  if (u == 0)
    return;

  if (u == 1)
    return oaddrow(a, b, i, j, k);

  const uint8_t *urow_hi = GF2_8_SHUF_HI + (u * 16);
  const uint8_t *urow_lo = GF2_8_SHUF_LO + (u * 16);
  oblas_axpy(ap, bp, k, urow_lo, urow_hi);
}

void oaddrow(uint8_t *restrict a, uint8_t *restrict b, size_t i, size_t j,
             size_t k) {
  uint8_t *ap = a + (i * ALIGNED_COLS(k));
  uint8_t *bp = b + (j * ALIGNED_COLS(k));
  oblas_xor(ap, bp, k);
}

void oscal(uint8_t *restrict a, size_t i, size_t k, uint8_t u) {
  uint8_t *ap = a + (i * ALIGNED_COLS(k));

  if (u < 2)
    return;

  const uint8_t *urow_lo = GF2_8_SHUF_LO + (u * 16);
  const uint8_t *urow_hi = GF2_8_SHUF_HI + (u * 16);
  oblas_scal(ap, k, urow_lo, urow_hi);
}

void ozero(uint8_t *restrict a, size_t i, size_t k) {
  uint8_t *ap = a + (i * ALIGNED_COLS(k));
  oblas_zero(ap, k);
}

void oaxpy_b32(uint8_t *a, uint32_t *b, size_t i, size_t k, uint8_t u) {
  uint8_t *ap = a + (i * ALIGNED_COLS(k));
  oblas_axpy_gf2_gf256_32(ap, b, k, u);
}
