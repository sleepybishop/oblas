#include "gf2.h"
#include <stdlib.h>
#include <string.h>

gf2mat *gf2mat_new(int m, int n) {
  gf2mat *gf2 = calloc(1, sizeof(gf2mat));
  gf2->m = m;
  gf2->n = n;
  gf2->stride = ((n / 32) + ((n % 32) ? 1 : 0));
  gf2->bits = calloc(sizeof(gf2word), gf2->stride * m);
  return gf2;
}

void gf2mat_free(gf2mat *gf2) {
  if (gf2 && gf2->bits)
    free(gf2->bits);
  if (gf2)
    free(gf2);
}

void gf2mat_print(gf2mat *gf2, FILE *stream) {
  fprintf(stream, "gf2 [%ux%u]\n", gf2->m, gf2->n);
  fprintf(stream, "|     ");
  for (int j = 0; j < gf2->n; j++) {
    fprintf(stream, "| %03d ", j);
  }
  fprintf(stream, "|\n");
  for (int i = 0; i < gf2->m; i++) {
    fprintf(stream, "| %03d | %3d ", i, gf2mat_get(gf2, i, 0));
    for (int j = 1; j < gf2->n; j++) {
      fprintf(stream, "| %3d ", gf2mat_get(gf2, i, j));
    }
    fprintf(stream, "|\n");
  }
}

gf2mat *gf2mat_copy(gf2mat *_gf2) {
  if (!_gf2 || !_gf2->bits)
    return NULL;
  gf2mat *gf2 = calloc(1, sizeof(gf2mat));
  gf2->m = _gf2->m;
  gf2->n = _gf2->n;
  gf2->stride = _gf2->stride;
  gf2->bits = calloc(sizeof(gf2word), gf2->stride * gf2->m);
  memcpy(gf2->bits, _gf2->bits, sizeof(gf2word) * gf2->stride * gf2->m);
  return gf2;
}

int gf2mat_get(gf2mat *gf2, int i, int j) {
  if (i >= gf2->m || j >= gf2->n)
    return 0;

  gf2word *a = gf2->bits + i * gf2->stride;
  int p = j / gf2wsz;

  gf2word mask = 1 << (j % gf2wsz);

  return (a[p] & mask) != 0;
}

void gf2mat_set(gf2mat *gf2, int i, int j, uint8_t b) {
  if (i >= gf2->m || j >= gf2->n)
    return;

  gf2word *a = gf2->bits + i * gf2->stride;
  int p = j / gf2wsz;

  gf2word mask = 1 << (j % gf2wsz);
  a[p] = (b) ? (a[p] | mask) : (a[p] & ~mask);
}

void gf2mat_xor(gf2mat *gf2, int i, int j) {
  gf2word *a = gf2->bits + i * gf2->stride;
  gf2word *b = gf2->bits + j * gf2->stride;
  int stride = gf2->stride;
  for (int p = 0; p < stride; p++) {
    a[p] ^= b[p];
  }
}

void gf2mat_and(gf2mat *gf2, int i, int j) {
  gf2word *a = gf2->bits + i * gf2->stride;
  gf2word *b = gf2->bits + j * gf2->stride;
  int stride = gf2->stride;
  for (int p = 0; p < stride; p++) {
    a[p] &= b[p];
  }
}

void gf2mat_nnz_word(gf2mat *gf2, int i, int s, int e, int *nnz,
                     int ones_at[]) {
  gf2word *a = gf2->bits + i * gf2->stride;
  int p = s / gf2wsz;
  for (int j = s; j < e; j++) {
    gf2word mask = 1 << (j % gf2wsz);
    int bit = (a[p] & mask) != 0;
    if (bit) {
      *nnz += 1;
      if (*nnz <= 2) {
        ones_at[*nnz - 1] = j - s;
      }
    }
  }
}

void gf2mat_nnz(gf2mat *gf2, int i, int s, int e, int *nnz, int ones_at[]) {
  if (i >= gf2->m || s < 0 || s > e || e >= gf2->n)
    return;

  gf2word *a = gf2->bits + i * gf2->stride;

  int align_s = (s - (s % gf2wsz));
  if (align_s < s)
    align_s += gf2wsz;
  if (align_s > e)
    align_s = e;
  int align_e = e - (e % gf2wsz);
  if (align_e < align_s)
    align_e = e;

  gf2mat_nnz_word(gf2, i, s, align_s, nnz, ones_at);
  for (int l = align_s; l < align_e; l += gf2wsz) {
    int p = l / gf2wsz;
    int bits = __builtin_popcountll(a[p]);
    if (*nnz <= 2) {
      int tz = __builtin_ctz(a[p]);
      ones_at[*nnz] = tz - s;
      if (bits > 1 && *nnz < 2) {
        gf2word mask = (gf2word)(-1) << (tz + 1);
        tz = __builtin_ctz(a[p] & mask);
        ones_at[*nnz + 1] = tz - s;
      }
    }
    *nnz += bits;
  }
  gf2mat_nnz_word(gf2, i, align_e, e, nnz, ones_at);
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
  int m = gf2->m;
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
