#ifndef GF4MAT_H
#define GF4MAT_H

#include <stdint.h>
#include <stdio.h>

#define gf4word uint32_t
#define gf4wsz (sizeof(gf4word) * 8)
#define gf4bpw (sizeof(gf4word) * 4)

typedef struct {
  unsigned rows;
  unsigned cols;
  unsigned stride;
  gf4word *bits;
} gf4mat;

gf4mat *gf4mat_new(unsigned rows, unsigned cols);
void gf4mat_free(gf4mat *gf4);
void gf4mat_print(gf4mat *gf4, FILE *stream);
gf4mat *gf4mat_copy(gf4mat *gf4);

void gf4mat_axpy(gf4mat *gf4, int i, uint8_t *dst, int beta);
void gf4mat_fill(gf4mat *gf4, int i, uint8_t *dst);

int gf4mat_get(gf4mat *gf4, int i, int j);
void gf4mat_set(gf4mat *gf4, int i, int j, uint8_t b);

void gf4mat_add(gf4mat *a, gf4mat *b, int i, int j);
void gf4mat_mul(gf4mat *a, gf4mat *b, int i, int j);
void gf4mat_scal(gf4mat *a, int i, int u);

int gf4mat_nnz(gf4mat *gf4, int i, int s, int e);

void gf4mat_swaprow(gf4mat *gf4, int i, int j);
void gf4mat_swapcol(gf4mat *gf4, int i, int j);

#endif
