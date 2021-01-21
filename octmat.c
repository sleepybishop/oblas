#include "octmat.h"
#include "oblas.h"
#include <errno.h>

#define ALIGN_TO(k, a) (((k) / (a)) + (((k) % (a)) ? 1 : 0)) * (a)

octmat *octmat_new(unsigned rows, unsigned cols) {
  octmat *m = calloc(1, sizeof(octmat));
  m->rows = rows;
  m->cols = cols;
  m->stride = ALIGN_TO(cols, OCTMAT_ALIGN);
  m->bits = oblas_alloc(rows, m->stride, OCTMAT_ALIGN);
  oblas_zero(m->bits, rows * m->stride);
  return m;
}

void octmat_free(octmat *m) {
  if (m && m->bits)
    oblas_free(m->bits);
  if (m)
    free(m);
}

void oswaprow(octmat *a, unsigned i, unsigned j) {
  if (i == j)
    return;
  uint8_t *ap = a->bits + i * a->stride;
  uint8_t *bp = a->bits + j * a->stride;
  oblas_swap(ap, bp, a->stride);
}

void oaxpy(octmat *a, octmat *b, unsigned i, unsigned j, uint8_t u) {
  uint8_t *ap = a->bits + i * a->stride;
  uint8_t *bp = b->bits + j * b->stride;

  if (u == 0)
    return;

  if (u == 1) {
    oblas_xor(ap, bp, a->stride);
  } else {
    const uint8_t *urow_hi = GF2_8_SHUF_HI + (u * 16);
    const uint8_t *urow_lo = GF2_8_SHUF_LO + (u * 16);
    oblas_axpy(ap, bp, a->stride, urow_lo, urow_hi);
  }
}

void oaddrow(octmat *a, octmat *b, unsigned i, unsigned j) {
  uint8_t *ap = a->bits + i * a->stride;
  uint8_t *bp = b->bits + j * b->stride;
  oblas_xor(ap, bp, a->stride);
}

void oscal(octmat *a, unsigned i, uint8_t u) {
  uint8_t *ap = a->bits + i * a->stride;

  if (u < 2)
    return;

  const uint8_t *urow_lo = GF2_8_SHUF_LO + (u * 16);
  const uint8_t *urow_hi = GF2_8_SHUF_HI + (u * 16);
  oblas_scal(ap, a->stride, urow_lo, urow_hi);
}

void oaxpy_b32(octmat *a, uint32_t *b, unsigned i, uint8_t u) {
  uint8_t *ap = a->bits + i * a->stride;
  oblas_axpy_gf2_gf256_32(ap, b, a->stride, u);
}
