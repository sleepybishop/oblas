#ifndef GF2MAT_H
#define GF2MAT_H

#include <stdint.h>
#include <stdio.h>

#define gf2word uint32_t
#define gf2wsz (sizeof(gf2word) * 8)

typedef struct {
  int m;
  int n;
  int stride;
  gf2word *bits;
} gf2mat;

gf2mat *gf2mat_new(int m, int n);
void gf2mat_free(gf2mat *gf2);
void gf2mat_print(gf2mat *gf2, FILE *stream);
gf2mat *gf2mat_copy(gf2mat *gf2);

void gf2mat_axpy(gf2mat *gf2, int i, uint8_t *dst, int beta);
void gf2mat_fill(gf2mat *gf2, int i, uint8_t *dst);

int gf2mat_get(gf2mat *gf2, int i, int j);
void gf2mat_set(gf2mat *gf2, int i, int j, uint8_t b);

void gf2mat_xor(gf2mat *gf2, int i, int j);
void gf2mat_and(gf2mat *gf2, int i, int j);

void gf2mat_nnz(gf2mat *gf2, int i, int s, int e, int *nnz, int ones_at[]);
void gf2mat_swaprow(gf2mat *gf2, int i, int j);
void gf2mat_swapcol(gf2mat *gf2, int i, int j);
void gf2mat_zero(gf2mat *gf2, int i);

#endif
