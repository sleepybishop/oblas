#ifndef GFMAT_H
#define GFMAT_H

#include "oblas.h"

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
