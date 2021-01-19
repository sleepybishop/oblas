#include "octmat.h"
#include "oblas.h"
#include <errno.h>

#define ALIGN_TO(k, a) (((k) / (a)) + (((k) % (a)) ? 1 : 0)) * (a)

void om_resize(octmat *v, size_t rows, size_t cols) {
  void *aligned = NULL;

  v->rows = rows;
  v->cols = cols;
  v->cols_al = ALIGN_TO(v->cols, OCTMAT_ALIGN);
  aligned = (uint8_t *)oblas_alloc(v->rows, v->cols_al, OCTMAT_ALIGN);
  memset(aligned, 0, v->cols_al * rows);
  v->data = aligned;
}

void om_destroy(octmat *v) {
  v->rows = 0;
  v->cols = 0;
  v->cols_al = 0;
  oblas_free(v->data);
  v->data = NULL;
}

void oswaprow(uint8_t *restrict a, size_t i, size_t j, size_t k) {
  if (i == j)
    return;
  uint8_t *ap = a + (i * ALIGN_TO(k, OCTMAT_ALIGN));
  uint8_t *bp = a + (j * ALIGN_TO(k, OCTMAT_ALIGN));
  oblas_swap(ap, bp, k);
}

void oaxpy(uint8_t *restrict a, uint8_t *restrict b, size_t i, size_t j,
           size_t k, uint8_t u) {
  uint8_t *ap = a + (i * ALIGN_TO(k, OCTMAT_ALIGN));
  uint8_t *bp = b + (j * ALIGN_TO(k, OCTMAT_ALIGN));

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
  uint8_t *ap = a + (i * ALIGN_TO(k, OCTMAT_ALIGN));
  uint8_t *bp = b + (j * ALIGN_TO(k, OCTMAT_ALIGN));
  oblas_xor(ap, bp, k);
}

void oscal(uint8_t *restrict a, size_t i, size_t k, uint8_t u) {
  uint8_t *ap = a + (i * ALIGN_TO(k, OCTMAT_ALIGN));

  if (u < 2)
    return;

  const uint8_t *urow_lo = GF2_8_SHUF_LO + (u * 16);
  const uint8_t *urow_hi = GF2_8_SHUF_HI + (u * 16);
  oblas_scal(ap, k, urow_lo, urow_hi);
}

void oaxpy_b32(uint8_t *a, uint32_t *b, size_t i, size_t k, uint8_t u) {
  uint8_t *ap = a + (i * ALIGN_TO(k, OCTMAT_ALIGN));
  oblas_axpy_gf2_gf256_32(ap, b, k, u);
}
