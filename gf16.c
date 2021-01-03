#include "gf16.h"
#include "util.h"
#include <stdlib.h>

gf16mat *gf16mat_new(unsigned rows, unsigned cols) {
  gf16mat *gf16 = calloc(1, sizeof(gf16mat));
  gf16->rows = rows;
  gf16->cols = cols;
  gf16->stride = ((cols / gf16bpw) + ((cols % gf16bpw) ? 1 : 0));
  gf16->bits =
      oblas_alloc(rows, gf16->stride * sizeof(gf16word), sizeof(void *));
  oblas_zero(gf16->bits, rows * gf16->stride * sizeof(gf16word));
  return gf16;
}

void gf16mat_free(gf16mat *gf16) {
  if (gf16 && gf16->bits)
    oblas_free(gf16->bits);
  if (gf16)
    free(gf16);
}

void gf16mat_print(gf16mat *gf16, FILE *stream) {
  fprintf(stream, "gf16 [%ux%u]\n", (unsigned)gf16->rows, (unsigned)gf16->cols);
  fprintf(stream, "|     ");
  for (int j = 0; j < gf16->cols; j++) {
    fprintf(stream, "| %03d ", j);
  }
  fprintf(stream, "|\n");
  for (int i = 0; i < gf16->rows; i++) {
    fprintf(stream, "| %03d | %3d ", i, gf16mat_get(gf16, i, 0));
    for (int j = 1; j < gf16->cols; j++) {
      fprintf(stream, "| %3d ", gf16mat_get(gf16, i, j));
    }
    fprintf(stream, "|\n");
  }
}

gf16mat *gf16mat_copy(gf16mat *_gf16) {
  if (!_gf16 || !_gf16->bits)
    return NULL;
  gf16mat *gf16 = calloc(1, sizeof(gf16mat));
  gf16->rows = _gf16->rows;
  gf16->cols = _gf16->cols;
  gf16->stride = _gf16->stride;
  gf16->bits = oblas_alloc(_gf16->rows, _gf16->stride * sizeof(gf16word),
                           sizeof(void *));
  memcpy(gf16->bits, _gf16->bits, sizeof(gf16word) * gf16->stride * gf16->cols);
  return gf16;
}

int gf16mat_get(gf16mat *gf16, int i, int j) {
  if (i >= gf16->rows || j >= gf16->cols)
    return 0;

  gf16word *a = gf16->bits + i * gf16->stride;
  div_t p = div(j, gf16bpw);
  return bfx_32(a[p.quot], p.rem << 2, 4);
}

void gf16mat_set(gf16mat *gf16, int i, int j, uint8_t b) {
  if (i >= gf16->rows || j >= gf16->cols)
    return;

  gf16word *a = gf16->bits + i * gf16->stride;
  div_t p = div(j, gf16bpw);
  a[p.quot] = bfd_32(a[p.quot], p.rem << 2, 4, b % 16);
}

void gf16mat_add(gf16mat *a, gf16mat *b, int i, int j) {
  gf16word *ap = a->bits + i * a->stride;
  gf16word *bp = b->bits + j * b->stride;
  unsigned stride = a->stride;
  for (int idx = 0; idx < stride; idx++) {
    ap[idx] ^= bp[idx];
  }
}

void gf16mat_mul(gf16mat *a, gf16mat *b, int i, int j) {
  gf16word *ap = a->bits + i * a->stride;
  gf16word *bp = b->bits + j * b->stride;
  unsigned stride = a->stride;
  for (int idx = 0; idx < stride; idx++) {
    /*TODO*/
  }
}

void gf16mat_scal(gf16mat *a, int i, int u) {
  gf16word *ap = a->bits + i * a->stride;
  unsigned stride = a->stride;
  for (int idx = 0; idx < stride; idx++) {
    /*TODO*/
  }
}

int gf16mat_nnz(gf16mat *gf16, int i, int s, int e) {
  if (i >= gf16->rows || s < 0 || s > e || e > (gf16->cols + 1))
    return 0;
  gf16word *a = gf16->bits + i * gf16->stride;
  unsigned nnz = 0;
  div_t sd = div(s, gf16bpw), ed = div(e, gf16bpw);
  gf16word masks[2] = {~((1 << sd.rem) - 1), ((1 << ed.rem) - 1)};

  for (int idx = sd.quot; idx <= ed.quot; idx++) {
    gf16word tmp = a[idx], z = 0, mask = -1;
    while (tmp > 0) {
      unsigned tz = __builtin_ctz(tmp) >> 2;
      z |= 1 << tz;
      tmp = tmp & (tmp - 1);
    }
    if (sd.rem && idx == sd.quot)
      mask = masks[0];
    else if (e > ed.quot && idx == ed.quot)
      mask = masks[1];
    nnz += __builtin_popcount(z & mask);
  }
  return nnz;
}

void gf16mat_fill(gf16mat *gf16, int i, uint8_t *dst) {
  gf16word *a = gf16->bits + i * gf16->stride;
  unsigned stride = gf16->stride;
  for (int idx = 0; idx < stride; idx++) {
    gf16word tmp = a[idx];
    while (tmp > 0) {
      unsigned tz = __builtin_ctz(tmp);
      div_t q = div(tz, 4);
      tmp = tmp & (tmp - 1);
      dst[q.quot + idx * gf16bpw] |= 1 << q.rem;
    }
  }
}

void gf16mat_swaprow(gf16mat *gf16, int i, int j) {
  if (i == j)
    return;

  gf16word *a = gf16->bits + i * gf16->stride;
  gf16word *b = gf16->bits + j * gf16->stride;
  unsigned stride = gf16->stride;
  for (int idx = 0; idx < stride; idx++) {
    gf16word __tmp = a[idx];
    a[idx] = b[idx];
    b[idx] = __tmp;
  }
}

void gf16mat_swapcol(gf16mat *gf16, int i, int j) {
  if (i == j)
    return;

  gf16word *a = gf16->bits;
  unsigned stride = gf16->stride;
  unsigned p = i / gf16bpw;
  unsigned q = j / gf16bpw;
  unsigned p_at = (i % gf16bpw) << 2;
  unsigned q_at = (j % gf16bpw) << 2;
  unsigned m = gf16->rows;
  for (int r = 0; r < m; r++, p += stride, q += stride) {
    uint8_t ival = bfx_32(a[p], p_at, 4);
    uint8_t jval = bfx_32(a[q], q_at, 4);
    a[p] = bfd_32(a[p], p_at, 4, jval);
    a[q] = bfd_32(a[q], q_at, 4, ival);
  }
}
