#include "gf4.h"
#include "util.h"
#include <stdlib.h>

gf4mat *gf4mat_new(unsigned rows, unsigned cols) {
  gf4mat *gf4 = calloc(1, sizeof(gf4mat));
  gf4->rows = rows;
  gf4->cols = cols;
  gf4->stride = ((cols / gf4bpw) + ((cols % gf4bpw) ? 1 : 0));
  gf4->bits = oblas_alloc(rows, gf4->stride * sizeof(gf4word), sizeof(void *));
  oblas_zero(gf4->bits, rows * gf4->stride * sizeof(gf4word));
  return gf4;
}

void gf4mat_free(gf4mat *gf4) {
  if (gf4 && gf4->bits)
    oblas_free(gf4->bits);
  if (gf4)
    free(gf4);
}

void gf4mat_print(gf4mat *gf4, FILE *stream) {
  fprintf(stream, "gf4 [%ux%u]\n", (unsigned)gf4->rows, (unsigned)gf4->cols);
  fprintf(stream, "|     ");
  for (int j = 0; j < gf4->cols; j++) {
    fprintf(stream, "| %03d ", j);
  }
  fprintf(stream, "|\n");
  for (int i = 0; i < gf4->rows; i++) {
    fprintf(stream, "| %03d | %3d ", i, gf4mat_get(gf4, i, 0));
    for (int j = 1; j < gf4->cols; j++) {
      fprintf(stream, "| %3d ", gf4mat_get(gf4, i, j));
    }
    fprintf(stream, "|\n");
  }
}

gf4mat *gf4mat_copy(gf4mat *_gf4) {
  if (!_gf4 || !_gf4->bits)
    return NULL;
  gf4mat *gf4 = calloc(1, sizeof(gf4mat));
  gf4->rows = _gf4->rows;
  gf4->cols = _gf4->cols;
  gf4->stride = _gf4->stride;
  gf4->bits =
      oblas_alloc(_gf4->rows, _gf4->stride * sizeof(gf4word), sizeof(void *));
  memcpy(gf4->bits, _gf4->bits, sizeof(gf4word) * gf4->stride * gf4->cols);
  return gf4;
}

void gf4mat_fill(gf4mat *gf4, int i, uint8_t *dst) {
  gf4word *a = gf4->bits + i * gf4->stride;
  unsigned stride = gf4->stride;
  for (int idx = 0; idx < stride; idx++) {
    gf4word tmp = a[idx];
    while (tmp > 0) {
      unsigned tz = __builtin_ctz(tmp);
      div_t q = div(tz, 2);
      tmp = tmp & (tmp - 1);
      dst[q.quot + idx * gf4bpw] |= (q.rem + 1);
    }
  }
}

int gf4mat_get(gf4mat *gf4, int i, int j) {
  if (i >= gf4->rows || j >= gf4->cols)
    return 0;

  gf4word *a = gf4->bits + i * gf4->stride;
  div_t p = div(j, gf4bpw);
  return bfx_32(a[p.quot], p.rem << 1, 2);
}

void gf4mat_set(gf4mat *gf4, int i, int j, uint8_t b) {
  if (i >= gf4->rows || j >= gf4->cols)
    return;

  gf4word *a = gf4->bits + i * gf4->stride;
  div_t p = div(j, gf4bpw);
  a[p.quot] = bfd_32(a[p.quot], p.rem << 1, 2, b % 4);
}

void gf4mat_add(gf4mat *a, gf4mat *b, int i, int j) {
  gf4word *ap = a->bits + i * a->stride;
  gf4word *bp = b->bits + j * b->stride;
  unsigned stride = a->stride;
  for (int idx = 0; idx < stride; idx++) {
    ap[idx] ^= bp[idx];
  }
}

void gf4mat_mul(gf4mat *a, gf4mat *b, int i, int j) {
  gf4word *ap = a->bits + i * a->stride;
  gf4word *bp = b->bits + j * b->stride;
  unsigned stride = a->stride;
  for (int idx = 0; idx < stride; idx++) {
    /*TODO*/
  }
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
    }
  }
}

void gf4mat_scal(gf4mat *a, int i, int u) {
  gf4word *ap = a->bits + i * a->stride;
  unsigned stride = a->stride;
  for (int idx = 0; idx < stride; idx++) {
    /*TODO*/
  }
}

int gf4mat_nnz(gf4mat *gf4, int i, int s, int e) {
  if (i >= gf4->rows || s < 0 || s > e || e > gf4->cols)
    return 0;
  gf4word *a = gf4->bits + i * gf4->stride;
  unsigned nnz = 0;
  div_t sd = div(s, gf4wsz), ed = div(e, gf4wsz);
  gf4word masks[2] = {~((1 << sd.rem) - 1), ((1 << ed.rem) - 1)};

  for (int idx = sd.quot; idx <= (ed.quot + 1); idx++) {
    gf4word tmp = a[idx], z = 0, mask = -1;
    while (tmp > 0) {
      unsigned tz = __builtin_ctz(tmp) >> 1;
      z |= 1 << tz;
      tmp = tmp & (tmp - 1);
    }
    if (sd.rem && idx == sd.quot)
      mask = masks[0];
    else if (e > ed.quot && idx == ed.quot + 1)
      mask = masks[1];
    nnz += __builtin_popcount(z & mask);
  }
  return nnz;
}

void gf4mat_swaprow(gf4mat *gf4, int i, int j) {
  if (i == j)
    return;

  gf4word *a = gf4->bits + i * gf4->stride;
  gf4word *b = gf4->bits + j * gf4->stride;
  unsigned stride = gf4->stride;
  for (int idx = 0; idx < stride; idx++) {
    gf4word __tmp = a[idx];
    a[idx] = b[idx];
    b[idx] = __tmp;
  }
}

void gf4mat_swapcol(gf4mat *gf4, int i, int j) {
  if (i == j)
    return;

  gf4word *a = gf4->bits;
  unsigned stride = gf4->stride;
  unsigned p = i / gf4bpw;
  unsigned q = j / gf4bpw;
  unsigned p_at = ((i % gf4bpw) - 1) << 1;
  unsigned q_at = ((j % gf4bpw) - 1) << 1;
  unsigned m = gf4->rows;
  for (int r = 0; r < m; r++, p += stride, q += stride) {
    uint8_t ival = bfx_32(a[p], p_at, 2);
    uint8_t jval = bfx_32(a[q], q_at, 2);
    a[p] = bfd_32(a[p], p_at, 2, jval);
    a[q] = bfd_32(a[q], q_at, 2, ival);
  }
}
