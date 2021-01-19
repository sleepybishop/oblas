#ifndef BINMAT_H
#define BINMAT_H

#include "oblas.h"

typedef struct {
  unsigned rows;
  unsigned cols;
  unsigned stride;
  oblas_word *bits;
} binmat;

binmat *binmat_new(unsigned rows, unsigned cols);
void binmat_free(binmat *m);
uint8_t binmat_get(binmat *m, unsigned i, unsigned j);
void binmat_set(binmat *m, unsigned i, unsigned j, uint8_t b);
void binmat_add(binmat *a, binmat *b, unsigned i, unsigned j);
void binmat_swaprow(binmat *m, unsigned i, unsigned j);
void binmat_zero(binmat *a, unsigned i);
void binmat_fill(binmat *m, unsigned i, uint8_t *dst);
void binmat_expand(binmat *m, uint32_t *src, unsigned i, uint8_t u);
unsigned binmat_nnz(binmat *m, unsigned i, unsigned s, unsigned e);

#endif
