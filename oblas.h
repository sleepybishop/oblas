#ifndef OBLAS_H
#define OBLAS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GF_ADD(a, b) ((a) ^ (b))
#define GF_SUB(a, b) ((a) ^ (b))
#define GF_MUL(t, a, b) (t##_EXP[t##_LOG[u] + t##_LOG[v]])
#define GF_DIV(t, a, b, s) (t##_EXP[t##_LOG[u] - t##_LOG[v] + (s - 1)])

#define oblas_word uint32_t

typedef enum { GF2_1 = 0, GF2_2 = 1, GF2_4 = 2, GF2_8 = 3 } gf_field;

typedef struct {
  gf_field field;
  unsigned vpw;
  unsigned exp;
  unsigned len;
  unsigned poly;
} gf;

typedef struct {
  gf_field field;
  unsigned rows;
  unsigned cols;
  unsigned stride;
  unsigned align;
  oblas_word *bits;
} gfmat;

gfmat *gfmat_new(gf_field field, unsigned rows, unsigned cols);
gfmat *gfmat_copy(gfmat *_m);
void gfmat_free(gfmat *m);
uint8_t gfmat_get(gfmat *m, unsigned i, unsigned j);
void gfmat_set(gfmat *m, unsigned i, unsigned j, uint8_t b);
void gfmat_print(gfmat *m, FILE *stream);
void gfmat_add(gfmat *a, gfmat *b, unsigned i, unsigned j);
void gfmat_swaprow(gfmat *m, unsigned i, unsigned j);
void gfmat_axpy(gfmat *a, gfmat *b, unsigned i, unsigned j, uint8_t u);
void gfmat_scal(gfmat *a, unsigned i, uint8_t u);
void gfmat_zero(gfmat *a, unsigned i);
void gfmat_fill(gfmat *m, unsigned i, uint8_t *dst);
void gfmat_expand(gfmat *m, uint32_t *src, unsigned i, uint8_t u);
int gfmat_nnz(gfmat *m, unsigned i, unsigned s, unsigned e);
void gfmat_swapcol(gfmat *m, unsigned i, unsigned j);

#endif
