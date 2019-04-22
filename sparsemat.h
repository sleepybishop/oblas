#ifndef SPARSEMAT_H
#define SPARSEMAT_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "kvec.h"

#define UINT16_SWAP(u, v)                                                      \
  do {                                                                         \
    uint16_t __tmp = (u);                                                      \
    (u) = (v);                                                                 \
    (v) = __tmp;                                                               \
  } while (0)

typedef struct {
  kvec_t(uint16_t) idxs;
  kvec_t(uint8_t) vals;
} sparsevec;

typedef struct {
  uint16_t rows;
  uint16_t cols;

  sparsevec *r;

  uint16_t *rl2a;
  uint16_t *cl2a;
  uint16_t *ra2l;
  uint16_t *ca2l;
} sparsemat;

sparsevec sv_new(uint16_t initial);
uint16_t sv_get(sparsevec *v, uint16_t r);
void sv_set(sparsevec *v, uint16_t r, uint8_t val);
void sv_remove(sparsevec *v, uint16_t r);
void sv_destroy(sparsevec *v);

void sv_print(sparsevec *v, FILE *stream);
void sv_densify(sparsevec *v, uint8_t *d);
void sv_copy(sparsevec *v, sparsevec *w);
void sv_axpy(sparsevec *v, sparsevec *w, uint8_t u);
void sv_add(sparsevec *v, sparsevec *w);

void sv_scal(sparsevec *v, uint16_t k, uint8_t u);
void sv_nnz(sparsevec *v, uint16_t s, uint16_t e, int *nnz, int *ones,
            int ones_idx[]);

sparsemat sm_new(uint16_t rows, uint16_t cols);
void sm_destroy(sparsemat *a);

uint8_t sm_get(sparsemat *a, uint16_t i, uint16_t j);
void sm_set(sparsemat *a, uint16_t i, uint16_t j, uint8_t val);

void sm_swaprow(sparsemat *a, uint16_t i, uint16_t j);
void sm_swapcol(sparsemat *a, uint16_t i, uint16_t j);

void sm_axpy(sparsemat *a, sparsemat *b, uint16_t i, uint16_t j, uint8_t u);
void sm_addrow(sparsemat *a, sparsemat *b, uint16_t i, uint16_t j);
void sm_scal(sparsemat *a, uint16_t i, uint8_t u);
void sm_gemm(sparsemat *a, sparsemat *b, sparsemat *c);
void sm_nnz(sparsemat *a, uint16_t i, uint16_t s, uint16_t e, int *nnz,
            int *ones, int ones_idx[]);

void sm_print(sparsemat *a, FILE *stream);
void sm_densify(sparsemat *a, uint8_t *d, uint16_t cols);

#endif
