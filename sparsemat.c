#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "sparsemat.h"

static int binsearch(uint16_t *a, int L, int R, uint16_t x) {
  int idx;
  while (L <= R) {
    idx = (L + R) / 2;
    if (a[idx] == x)
      return idx;
    else if (a[idx] < x)
      L = idx + 1;
    else if (a[idx] > x)
      R = idx - 1;
  }
  return L;
}

sparsevec sv_new(uint16_t initial) {
  sparsevec v = {0};
  kv_init(v.idxs);
  kv_init(v.vals);
  v.nnz = 0;

  if (initial > 0) {
    kv_resize(uint16_t, v.idxs, initial);
    kv_resize(uint8_t, v.vals, initial);
  }

  return v;
}

uint16_t sv_get(sparsevec *v, uint16_t i) {
  if (v->nnz == 0)
    return 0;
  int idx = binsearch(v->idxs.a, 0, kv_size(v->idxs) - 1, i);
  return (kv_A(v->idxs, idx) == i) ? kv_A(v->vals, idx) & 0xffff : 0;
}

void sv_set(sparsevec *v, uint16_t i, uint8_t val) {
  int idx = binsearch(v->idxs.a, 0, kv_size(v->idxs) - 1, i);
  if (kv_size(v->idxs) > idx && kv_A(v->idxs, idx) == i) {
    if (kv_A(v->vals, idx) > 0)
      v->nnz--;
    kv_A(v->vals, idx) = val;
    if (val > 0)
      v->nnz++;
  } else {
    if (kv_size(v->idxs) == 0 || idx > (kv_size(v->idxs))) {
      kv_push(uint16_t, v->idxs, i);
      kv_push(uint8_t, v->vals, val);
      if (val > 0)
        v->nnz++;
    } else {
      kv_push(uint16_t, v->idxs, 0);
      kv_push(uint8_t, v->vals, 0);
      memmove(v->idxs.a + idx + 1, v->idxs.a + idx,
              2 * (kv_size(v->idxs) - idx));
      memmove(v->vals.a + idx + 1, v->vals.a + idx, kv_size(v->vals) - idx);
      kv_A(v->idxs, idx) = i;
      kv_A(v->vals, idx) = val;
      if (val > 0)
        v->nnz++;
    }
  }
}

void sv_remove(sparsevec *v, uint16_t i) {
  int idx = binsearch(v->idxs.a, 0, kv_size(v->idxs) - 1, i);
  if (kv_size(v->idxs) > idx && kv_A(v->idxs, idx) == i) {
    if (kv_A(v->vals, idx) > 0)
      v->nnz--;
    memmove(v->idxs.a + idx, v->idxs.a + idx + 1, 2 * (kv_size(v->idxs) - idx));
    memmove(v->vals.a + idx, v->vals.a + idx + 1, kv_size(v->vals) - idx);
    v->idxs.n--;
    v->vals.n--;
  }
}

void sv_destroy(sparsevec *v) {
  kv_destroy(v->idxs);
  kv_destroy(v->vals);
}

void sv_copy(sparsevec *a, sparsevec *b, uint16_t i, uint16_t j, uint16_t k) {}

void sv_swap(sparsevec *a, uint16_t i, uint16_t j, uint16_t k, uint16_t l) {}

void sv_axpy(sparsevec *a, uint16_t i, uint16_t j, uint16_t k, uint8_t u) {}

void sv_add(sparsevec *a, uint16_t i, uint16_t j, uint16_t k) {}

void sv_scal(sparsevec *a, uint16_t i, uint16_t k, uint8_t u) {}

void sv_zero(sparsevec *a, uint16_t i, size_t k) {}

void sv_nnz(sparsevec *a, uint16_t i, uint16_t s, uint16_t e, uint16_t k,
            int *nnz, int *ones, int ones_idx[]) {}

void sm_copy(sparsemat *a, sparsemat *b, uint16_t i, uint16_t j, uint16_t k) {}

void sm_resize(sparsemat *a, uint16_t r, uint16_t c) {}

void sm_destroy(sparsemat *a) {}

void sm_print(sparsemat *a, FILE *stream) {}

void sm_swaprow(sparsemat *a, uint16_t i, uint16_t j, uint16_t k) {}

void sm_swapcol(sparsemat *a, uint16_t i, uint16_t j, uint16_t k, uint16_t l) {}

void sm_axpy(sparsemat *a, uint16_t i, uint16_t j, uint16_t k, uint8_t u) {}

void sm_addrow(sparsemat *a, uint16_t i, uint16_t j, uint16_t k) {}

void sm_scal(sparsemat *a, uint16_t i, uint16_t k, uint8_t u) {}

void sm_zero(sparsemat *a, uint16_t i, size_t k) {}

void sm_gemm(sparsemat *a, sparsemat *b, sparsemat *c, uint16_t n, uint16_t k,
             uint16_t m) {}

void sm_nnz(sparsemat *a, uint16_t i, uint16_t s, uint16_t e, uint16_t k,
            int *nnz, int *ones, int ones_idx[]) {}
