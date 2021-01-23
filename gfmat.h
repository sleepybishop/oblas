#ifndef GFMAT_H
#define GFMAT_H

#include "oblas.h"

typedef enum { GF2_1 = 0, GF2_2 = 1, GF2_4 = 2, GF2_8 = 3 } gfmat_type;

typedef struct {
  gfmat_type field;
  unsigned vpw;
  unsigned exp;
  unsigned len;
  unsigned poly;
  const uint8_t *shuf_lo;
  const uint8_t *shuf_hi;
} gfmat_field;

typedef struct {
  gfmat_type field;
  unsigned rows;
  unsigned cols;
  unsigned stride;
  unsigned align;
  oblas_word *bits;
} gfmat;

gfmat *gfmat_new(gfmat_type field, unsigned rows, unsigned cols);
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
unsigned gfmat_nnz(gfmat *m, unsigned i, unsigned s, unsigned e);
void gfmat_swapcol(gfmat *m, unsigned i, unsigned j);

#endif
