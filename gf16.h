#ifndef GF16MAT_H
#define GF16MAT_H

#include <stdint.h>
#include <stdio.h>

#define gf16word uint32_t
#define gf16wsz (sizeof(gf16word) * 8)
#define gf16bpw (sizeof(gf16word) * 2)

typedef struct {
  unsigned rows;
  unsigned cols;
  unsigned stride;
  gf16word *bits;
} gf16mat;

gf16mat *gf16mat_new(unsigned rows, unsigned cols);
void gf16mat_free(gf16mat *gf16);
void gf16mat_print(gf16mat *gf16, FILE *stream);
gf16mat *gf16mat_copy(gf16mat *gf16);

int gf16mat_get(gf16mat *gf16, int i, int j);
void gf16mat_set(gf16mat *gf16, int i, int j, uint8_t b);

void gf16mat_add(gf16mat *a, gf16mat *b, int i, int j);
void gf16mat_mul(gf16mat *a, gf16mat *b, int i, int j);
void gf16mat_scal(gf16mat *a, int i, int u);

int gf16mat_nnz(gf16mat *gf16, int i, int s, int e);
void gf16mat_fill(gf16mat *gf16, int i, uint8_t *dst);

void gf16mat_swaprow(gf16mat *gf16, int i, int j);
void gf16mat_swapcol(gf16mat *gf16, int i, int j);

#endif
