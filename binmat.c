#include "binmat.h"
#include <errno.h>

#define ALIGN_TO(k, a) (((k) / (a)) + (((k) % (a)) ? 1 : 0)) * (a)

#define BM_VPW (8 * sizeof(oblas_word))

binmat *binmat_new(unsigned rows, unsigned cols) {
  binmat *m = calloc(1, sizeof(binmat));
  m->rows = rows;
  m->cols = cols;
  m->stride = ((cols / BM_VPW) + ((cols % BM_VPW) ? 1 : 0));
  m->stride = ALIGN_TO(m->stride, sizeof(void *));
  m->bits = oblas_alloc(rows, m->stride * sizeof(oblas_word), sizeof(void *));
  oblas_zero(m->bits, rows * m->stride * sizeof(oblas_word));
  return m;
}

void binmat_free(binmat *m) {
  if (m && m->bits)
    free(m->bits);
  if (m)
    free(m);
}

uint8_t binmat_get(binmat *m, unsigned i, unsigned j) {
  if (i >= m->rows || j >= m->cols)
    return 0;
  oblas_word *a = m->bits + i * m->stride;
  unsigned q = j / BM_VPW;
  unsigned r = j % BM_VPW;
  return (a[q] >> r) & 1;
}

void binmat_set(binmat *m, unsigned i, unsigned j, uint8_t b) {
  if (i >= m->rows || j >= m->cols)
    return;
  oblas_word *a = m->bits + i * m->stride;
  unsigned q = j / BM_VPW;
  unsigned r = j % BM_VPW;
  a[q] = (a[q] & ~(1 << r)) | (1 << r);
}

void binmat_add(binmat *a, binmat *b, unsigned i, unsigned j) {
  oblas_word *ap = a->bits + i * a->stride;
  oblas_word *bp = b->bits + j * b->stride;
  oblas_xor((uint8_t *)ap, (uint8_t *)bp, a->stride * sizeof(oblas_word));
}

void binmat_swaprow(binmat *m, unsigned i, unsigned j) {
  if (i == j)
    return;

  oblas_word *a = m->bits + i * m->stride;
  oblas_word *b = m->bits + j * m->stride;
  oblas_swap((uint8_t *)a, (uint8_t *)b, m->stride * sizeof(oblas_word));
}

void binmat_zero(binmat *a, unsigned i) {
  oblas_word *ap = a->bits + i * a->stride;
  oblas_zero((uint8_t *)ap, a->stride * sizeof(oblas_word));
}

void binmat_fill(binmat *m, unsigned i, uint8_t *dst) {
  oblas_word *a = m->bits + i * m->stride;
  for (unsigned idx = 0; idx < m->stride; idx++) {
    oblas_word tmp = a[idx];
    while (tmp > 0) {
      unsigned tz = __builtin_ctz(tmp);
      unsigned q = tz;
      tmp = tmp & (tmp - 1);
      dst[q + idx * BM_VPW] |= 1 << (tz % BM_VPW);
    }
  }
}

void binmat_expand(binmat *m, uint32_t *src, unsigned i, uint8_t u) {
  oblas_word *a = m->bits + i * m->stride;
  oblas_axpy_gf2_gf256_32((uint8_t *)a, src, m->stride * sizeof(oblas_word), u);
}

unsigned binmat_nnz(binmat *m, unsigned i, unsigned s, unsigned e) {
  if (i >= m->rows || s < 0 || s > e || e > (m->cols + 1))
    return 0;
  oblas_word *a = m->bits + i * m->stride;
  unsigned nnz = 0;
  unsigned sq = s / BM_VPW, eq = e / BM_VPW;
  unsigned sr = s % BM_VPW, er = e % BM_VPW;
  oblas_word masks[2] = {~((1 << sr) - 1), ((1 << er) - 1)};

  if (1) {
    oblas_word tmp = a[sq], z = 0;
    while (tmp > 0) {
      z = z | 1 << (__builtin_ctz(tmp));
      tmp = tmp & (tmp - 1);
    }
    nnz += __builtin_popcount(z & masks[0]);
  }
  for (unsigned idx = sq + 1; idx < eq; idx++) {
    oblas_word tmp = a[idx], z = 0;
    while (tmp > 0) {
      z = z | 1 << (__builtin_ctz(tmp));
      tmp = tmp & (tmp - 1);
    }
    nnz += __builtin_popcount(z);
  }
  if (e > eq) {
    oblas_word tmp = a[eq], z = 0;
    while (tmp > 0) {
      z = z | 1 << (__builtin_ctz(tmp));
      tmp = tmp & (tmp - 1);
    }
    nnz += __builtin_popcount(z & masks[1]);
  }
  return nnz;
}
