#ifndef SPARSEMAT_H
#define SPARSEMAT_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "kvec.h"

typedef struct {
  kvec_t(uint16_t) idxs;
  kvec_t(uint8_t) vals;
  uint16_t nnz;
} sparsevec;

typedef struct {
  uint16_t rows;
  uint16_t cols;

  kvec_t(sparsevec) r;

  kvec_t(uint16_t) rl2a;
  kvec_t(uint16_t) ra2l;

  kvec_t(uint16_t) cl2a;
  kvec_t(uint16_t) ca2l;
} sparsemat;

sparsevec sv_new(uint16_t initial);
uint16_t sv_get(sparsevec *v, uint16_t r);
void sv_set(sparsevec *v, uint16_t r, uint8_t val);
void sv_remove(sparsevec *v, uint16_t r);
void sv_destroy(sparsevec *v);

void sm_copy(sparsemat *a, sparsemat *b, uint16_t i, uint16_t j, uint16_t k);
void sm_resize(sparsemat *a, uint16_t r, uint16_t c);
void sm_destroy(sparsemat *a);
void sm_print(sparsemat *a, FILE *stream);

void sm_swaprow(sparsemat *a, uint16_t i, uint16_t j, uint16_t k);
void sm_swapcol(sparsemat *a, uint16_t i, uint16_t j, uint16_t k, uint16_t l);

void sm_axpy(sparsemat *a, uint16_t i, uint16_t j, uint16_t k, uint8_t u);
void sm_addrow(sparsemat *a, uint16_t i, uint16_t j, uint16_t k);
void sm_scal(sparsemat *a, uint16_t i, uint16_t k, uint8_t u);
void sm_zero(sparsemat *a, uint16_t i, size_t k);
void sm_gemm(sparsemat *a, sparsemat *b, sparsemat *c, uint16_t n, uint16_t k,
             uint16_t m);
void sm_nnz(sparsemat *a, uint16_t i, uint16_t s, uint16_t e, uint16_t k,
            int *nnz, int *ones, int ones_idx[]);

#endif
