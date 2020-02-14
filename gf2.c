#include "gf2.h"
#include <stdlib.h>
#include <string.h>

gf2mat *gf2mat_new(size_t rows, size_t cols) {
  gf2mat *gf2 = calloc(1, sizeof(gf2mat));
  gf2->rows = rows;
  gf2->cols = cols;
  gf2->stride = ((cols / gf2wsz) + ((cols % gf2wsz) ? 1 : 0));
  gf2->bits = calloc(sizeof(gf2word), gf2->stride * rows);
  return gf2;
}

void gf2mat_free(gf2mat *gf2) {
  if (gf2 && gf2->bits)
    free(gf2->bits);
  if (gf2)
    free(gf2);
}

void gf2mat_print(gf2mat *gf2, FILE *stream) {
  fprintf(stream, "gf2 [%ux%u]\n", (unsigned)gf2->rows, (unsigned)gf2->cols);
  fprintf(stream, "|     ");
  for (int j = 0; j < gf2->cols; j++) {
    fprintf(stream, "| %03d ", j);
  }
  fprintf(stream, "|\n");
  for (int i = 0; i < gf2->rows; i++) {
    fprintf(stream, "| %03d | %3d ", i, gf2mat_get(gf2, i, 0));
    for (int j = 1; j < gf2->cols; j++) {
      fprintf(stream, "| %3d ", gf2mat_get(gf2, i, j));
    }
    fprintf(stream, "|\n");
  }
}

gf2mat *gf2mat_copy(gf2mat *_gf2) {
  if (!_gf2 || !_gf2->bits)
    return NULL;
  gf2mat *gf2 = calloc(1, sizeof(gf2mat));
  gf2->rows = _gf2->rows;
  gf2->cols = _gf2->cols;
  gf2->stride = _gf2->stride;
  gf2->bits = calloc(sizeof(gf2word), gf2->stride * gf2->rows);
  memcpy(gf2->bits, _gf2->bits, sizeof(gf2word) * gf2->stride * gf2->cols);
  return gf2;
}

void gf2mat_axpy(gf2mat *gf2, int i, uint8_t *dst, int beta) {
  gf2word *a = gf2->bits + i * gf2->stride;
  int stride = gf2->stride;
  for (int p = 0; p < stride; p++) {
    gf2word tmp = a[p];
    while (tmp > 0) {
      int tz = __builtin_ctz(tmp);
      tmp = tmp & (tmp - 1);
      dst[tz + p * gf2wsz] ^= beta;
    }
  }
}

void gf2mat_fill(gf2mat *gf2, int i, uint8_t *dst) {
  gf2word *a = gf2->bits + i * gf2->stride;
  int stride = gf2->stride;
  for (int p = 0; p < stride; p++) {
    gf2word tmp = a[p];
    while (tmp > 0) {
      int tz = __builtin_ctz(tmp);
      tmp = tmp & (tmp - 1);
      dst[tz + p * gf2wsz] = 1;
    }
  }
}

int gf2mat_get(gf2mat *gf2, int i, int j) {
  if (i >= gf2->rows || j >= gf2->cols)
    return 0;

  gf2word *a = gf2->bits + i * gf2->stride;
  div_t p = div(j, gf2wsz);

  gf2word mask = 1 << p.rem;

  return !!(a[p.quot] & mask);
}

void gf2mat_set(gf2mat *gf2, int i, int j, uint8_t b) {
  if (i >= gf2->rows || j >= gf2->cols)
    return;

  gf2word *a = gf2->bits + i * gf2->stride;
  div_t p = div(j, gf2wsz);

  gf2word mask = 1 << p.rem;
  a[p.quot] = (b) ? (a[p.quot] | mask) : (a[p.quot] & ~mask);
}

void gf2mat_xor(gf2mat *a, gf2mat *b, int i, int j) {
  gf2word *ap = a->bits + i * a->stride;
  gf2word *bp = b->bits + j * b->stride;
  int stride = a->stride;
  for (int idx = 0; idx < stride; idx++) {
    ap[idx] ^= bp[idx];
  }
}

void gf2mat_and(gf2mat *a, gf2mat *b, int i, int j) {
  gf2word *ap = a->bits + i * a->stride;
  gf2word *bp = b->bits + j * b->stride;
  int stride = a->stride;
  for (int idx = 0; idx < stride; idx++) {
    ap[idx] &= bp[idx];
  }
}

int gf2mat_nnz(gf2mat *gf2, int i, int s, int e) {
  if (i >= gf2->rows || s < 0 || s > e || e > gf2->cols)
    return 0;
  gf2word *a = gf2->bits + i * gf2->stride;
  div_t ed = div(e, gf2wsz);
  int nnz = 0, p = 0, srem = s % gf2wsz;

  if (srem) {
    nnz += __builtin_popcount(-1 ^ ((1 << (srem - 1)) - 1));
    p++;
  }
  for (; p < ed.quot; p++) {
    nnz += __builtin_popcount(a[p]);
  }
  if (e > ed.quot) {
    nnz += __builtin_popcount(a[p] & ((1 << ed.rem) - 1));
  }

  return nnz;
}

void gf2mat_swaprow(gf2mat *gf2, int i, int j) {
  if (i == j)
    return;

  gf2word *a = gf2->bits + i * gf2->stride;
  gf2word *b = gf2->bits + j * gf2->stride;
  int stride = gf2->stride;
  for (int p = 0; p < stride; p++) {
    gf2word __tmp = a[p];
    a[p] = b[p];
    b[p] = __tmp;
  }
}

void gf2mat_swapcol(gf2mat *gf2, int i, int j) {
  if (i == j)
    return;

  gf2word *a = gf2->bits;
  gf2word mask = 0;

  int p = i / gf2wsz;
  int q = j / gf2wsz;
  int stride = gf2->stride;
  int m = gf2->rows;
  for (int r = 0; r < m; r++, p += stride, q += stride) {
    int ibit = a[p] & (1 << (i % gf2wsz));
    int jbit = a[q] & (1 << (j % gf2wsz));
    mask = 1 << (j % gf2wsz);
    a[p] = (ibit) ? (a[p] | mask) : (a[p] & ~mask);
    mask = 1 << (i % gf2wsz);
    a[q] = (jbit) ? (a[q] | mask) : (a[q] & ~mask);
  }
}

void gf2mat_zero(gf2mat *gf2, int i) {
  gf2word *a = gf2->bits + i * gf2->stride;
  int stride = gf2->stride;
  for (int p = 0; p < stride; p++) {
    a[p] = 0;
  }
}
