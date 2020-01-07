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

void gf2mat_axpy(gf2mat *gf2, int i, uint8_t *dst, int beta) {
  gf2word *a = gf2->bits + i * gf2->stride;
  int stride = gf2->stride;
  for (int p = 0; p < stride; p++) {
    gf2word tmp = a[p];
    while (tmp > 0) {
      int tz = __builtin_ctz(tmp);
      tmp &= (tmp - 1);
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
      tmp &= (tmp - 1);
      dst[tz + p * gf2wsz] = 1;
    }
  }
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

int gf2mat_nnz(gf2mat *gf2, int i, int s, int e) {
  if (i >= gf2->m || s < 0 || s > e || e > gf2->n)
    return 0;
  gf2word *a = gf2->bits + i * gf2->stride;
  int stride = gf2->stride;
  int nnz = 0;
  for (int p = 0; p < stride; p++) {
    gf2word mask = -1;
    if (p == 0 && s % gf2wsz) {
      mask = (-1 << (s % gf2wsz));
    }
    if (p == e / gf2wsz && e % gf2wsz) {
      mask &= ~(1 << (gf2wsz - 1)) >> (gf2wsz - (e % gf2wsz) - 1); 
    }
    gf2word tmp = a[p] & mask;
    nnz += __builtin_popcount(tmp);
  } 
  return nnz;
}

int gf2mat_ones_at(gf2mat *gf2, int i, int s, int e, int at[], int at_len) {
  if (i >= gf2->m || s < 0 || s > e || e > gf2->n) {
    return 0;
  }
  gf2word *a = gf2->bits + i * gf2->stride;
  int stride = gf2->stride;
  int found = 0;
  for (int p = 0; p < stride; p++) {
    gf2word mask = -1;
    if (p == 0 && s % gf2wsz) {
      mask = (-1 << (s % gf2wsz));
    }
    if (p == e / gf2wsz && e % gf2wsz) {
      mask &= ~(1 << (gf2wsz - 1)) >> (gf2wsz - (e % gf2wsz) - 1);
    }
    gf2word tmp = a[p] & mask;
    while (tmp > 0 && found < at_len) {
      int tz = __builtin_ctz(tmp);
      tmp &= (tmp - 1);
      at[found++] = tz + p * gf2wsz - (s % gf2wsz);
    }
  }
  return found;
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
