#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "octmul_hilo.h"
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

  if (initial > 0) {
    kv_resize(uint16_t, v.idxs, initial);
    kv_resize(uint8_t, v.vals, initial);
  }

  return v;
}

uint16_t sv_get(sparsevec *v, uint16_t i) {
  if (kv_size(v->idxs) == 0)
    return 0;
  int idx = binsearch(v->idxs.a, 0, kv_size(v->idxs) - 1, i);
  return (kv_size(v->idxs) > idx && kv_A(v->idxs, idx) == i)
             ? kv_A(v->vals, idx) & 0xffff
             : 0;
}

void sv_set(sparsevec *v, uint16_t i, uint8_t val) {
  int idx = binsearch(v->idxs.a, 0, kv_size(v->idxs) - 1, i);
  if (kv_size(v->idxs) == 0 || kv_size(v->idxs) <= idx) {
    if (val > 0) {
      kv_push(uint16_t, v->idxs, i);
      kv_push(uint8_t, v->vals, val);
    }
  } else if (kv_size(v->idxs) > idx) {
    if (kv_A(v->idxs, idx) == i) {
      kv_A(v->vals, idx) = val;
    } else {
      kv_push(uint16_t, v->idxs, 0);
      kv_push(uint8_t, v->vals, 0);
      memmove(v->idxs.a + idx + 1, v->idxs.a + idx,
              2 * (kv_size(v->idxs) - idx - 1));
      memmove(v->vals.a + idx + 1, v->vals.a + idx, kv_size(v->vals) - idx - 1);
      kv_A(v->idxs, idx) = i;
      kv_A(v->vals, idx) = val;
    }
  }
}

void sv_remove(sparsevec *v, uint16_t i) {
  int idx = binsearch(v->idxs.a, 0, kv_size(v->idxs) - 1, i);
  if (kv_size(v->idxs) > idx && kv_A(v->idxs, idx) == i) {
    memmove(v->idxs.a + idx, v->idxs.a + idx + 1,
            2 * (kv_size(v->idxs) - idx - 1));
    memmove(v->vals.a + idx, v->vals.a + idx + 1, kv_size(v->vals) - idx - 1);
    v->idxs.n--;
    v->vals.n--;
  }
}

void sv_destroy(sparsevec *v) {
  if (kv_max(v->idxs) > 0)
    kv_destroy(v->idxs);
  if (kv_max(v->vals) > 0)
    kv_destroy(v->vals);
}

void sv_copy(sparsevec *v, sparsevec *w) {
  kv_copy(uint16_t, v->idxs, w->idxs);
  kv_copy(uint8_t, v->vals, w->vals);
}

void sv_axpy(sparsevec *v, sparsevec *w, uint8_t u, uint16_t cols) {
  uint8_t tmp[cols];

  if (u == 0)
    return;

  memset(&tmp, 0, cols);
  const uint8_t *urow_hi = OCT_MUL_HI[u];
  const uint8_t *urow_lo = OCT_MUL_LO[u];
  for (int j = 0; j < kv_size(w->idxs); j++) {
    uint8_t w_lo = (kv_A(w->vals, j) & 0x0f);
    uint8_t w_hi = (kv_A(w->vals, j) & 0xf0) >> 4;
    uint8_t val = urow_hi[w_hi] ^ urow_lo[w_lo];
    tmp[kv_A(w->idxs, j)] = val;
  }
  for (int i = 0; i < kv_size(v->idxs); i++) {
    tmp[kv_A(v->idxs, i)] ^= kv_A(v->vals, i);
  }

  kv_size(v->idxs) = 0;
  kv_size(v->vals) = 0;
  for (int idx = 0; idx < cols; idx++) {
    if (tmp[idx] > 0) {
      kv_push(uint16_t, v->idxs, idx);
      kv_push(uint8_t, v->vals, tmp[idx]);
    }
  }
}

void sv_scal(sparsevec *v, uint16_t k, uint8_t u) {
  if (u < 2)
    return;

  int kidx = binsearch(v->idxs.a, 0, kv_size(v->idxs) - 1, k);
  if (kv_size(v->idxs) > kidx && kv_A(v->idxs, kidx) == k) {
    kidx++;
  }

  const uint8_t *urow_hi = OCT_MUL_HI[u];
  const uint8_t *urow_lo = OCT_MUL_LO[u];
  for (int idx = 0; idx < kidx; idx++) {
    uint8_t a_lo = (kv_A(v->vals, idx) & 0x0f);
    uint8_t a_hi = (kv_A(v->vals, idx) & 0xf0) >> 4;
    kv_A(v->vals, idx) = urow_hi[a_hi] ^ urow_lo[a_lo];
  }
}

sparsemat sm_new(uint16_t rows, uint16_t cols) {
  sparsemat a = {0};
  a.rows = rows;
  a.cols = cols;
  a.r = calloc(rows, sizeof(sparsevec));
  a.ra2l = calloc(rows, sizeof(uint16_t));
  a.rl2a = calloc(rows, sizeof(uint16_t));
  a.ca2l = calloc(cols, sizeof(uint16_t));
  a.cl2a = calloc(cols, sizeof(uint16_t));

  for (int i = 0; i < rows; i++) {
    a.r[i] = sv_new(cols / 4); // FIXME
    a.ra2l[i] = i;
    a.rl2a[i] = i;
  }

  for (int j = 0; j < cols; j++) {
    a.ca2l[j] = j;
    a.cl2a[j] = j;
  }

  return a;
}

void sm_destroy(sparsemat *a) {
  if (a->r) {
    for (int i = 0; i < a->rows; i++) {
      sparsevec *t = &a->r[i];
      sv_destroy(t);
    }
    free(a->r);
  }
  if (a->ra2l)
    free(a->ra2l);
  if (a->rl2a)
    free(a->rl2a);
  if (a->ca2l)
    free(a->ca2l);
  if (a->cl2a)
    free(a->cl2a);
}

void sm_swaprow(sparsemat *a, uint16_t i, uint16_t j) {
  uint16_t ia = a->rl2a[i];
  uint16_t ja = a->rl2a[j];
  UINT16_SWAP(a->rl2a[i], a->rl2a[j]);
  UINT16_SWAP(a->ra2l[ia], a->ra2l[ja]);
}

void sm_swapcol(sparsemat *a, uint16_t i, uint16_t j) {
  uint16_t ia = a->cl2a[i];
  uint16_t ja = a->cl2a[j];
  UINT16_SWAP(a->cl2a[i], a->cl2a[j]);
  UINT16_SWAP(a->ca2l[ia], a->ca2l[ja]);
}

uint8_t sm_get(sparsemat *a, uint16_t i, uint16_t j) {
  uint16_t ia = a->rl2a[i];
  uint16_t ja = a->cl2a[j];
  return sv_get(&a->r[ia], ja);
}

void sm_set(sparsemat *a, uint16_t i, uint16_t j, uint8_t val) {
  uint16_t ia = a->rl2a[i];
  uint16_t ja = a->cl2a[j];
  sv_set(&a->r[ia], ja, val);
}

void sm_axpy(sparsemat *a, sparsemat *b, uint16_t i, uint16_t j, uint8_t u) {
  uint16_t ia = a->rl2a[i];
  uint16_t ja = b->rl2a[j];
  sv_axpy(&a->r[ia], &b->r[ja], u, a->cols);
}

void sm_scal(sparsemat *a, uint16_t i, uint8_t u) {
  uint16_t ia = a->rl2a[i];
  sv_scal(&a->r[ia], a->cols, u);
}

void sm_gemm(sparsemat *a, sparsemat *b, sparsemat *c) {
  int k = 0;
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      uint8_t o = sm_get(a, i, j);
      sm_axpy(c, b, k, j, o);
    }
    k++;
  }
}

void sm_nnz(sparsemat *a, uint16_t i, uint16_t s, uint16_t e, int *nnz,
            int *ones, int ones_at[]) {
  *nnz = 0;
  *ones = 0;
  ones_at[0] = 0;
  ones_at[1] = 0;

  // FIXME: optimize
  for (int idx = s; idx < e; idx++) {
    uint8_t val = sm_get(a, i, idx);
    if (val != 0) {
      *nnz += 1;
      if (val == 1) {
        *ones += 1;
        if (*ones <= 2) {
          ones_at[*ones - 1] = idx - s;
        }
      }
    }
  }
}

void sm_densify(sparsemat *a, uint8_t *d, uint16_t cols) {
  // FIXME: optimize
  for (int i = 0; i < a->rows; i++) {
    uint8_t *dp = d + i * cols;
    memset(dp, 0, cols);
    for (int j = 0; j < cols; j++) {
      dp[j] = sm_get(a, i, j);
    }
  }
}

void sm_print(sparsemat *a, FILE *stream) {
  fprintf(stream, "sparse [%ux%u]\n", a->rows, a->cols);
  fprintf(stream, "|     ");
  for (int j = 0; j < a->cols; j++) {
    fprintf(stream, "| %03d ", j);
  }
  fprintf(stream, "|\n");
  for (int i = 0; i < a->rows; i++) {
    fprintf(stream, "| %03d | %3d ", i, sm_get(a, i, 0));
    for (int j = 1; j < a->cols; j++) {
      fprintf(stream, "| %3d ", sm_get(a, i, j));
    }
    fprintf(stream, "|\n");
  }
}
