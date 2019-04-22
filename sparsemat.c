#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "sparsemat.h"
#include "octmul_hilo.h"

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
    kv_push(uint16_t, v->idxs, i);
    kv_push(uint8_t, v->vals, val);
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

void sv_print(sparsevec *v, FILE *stream) {
  if (kv_size(v->idxs) == 0) {
    fprintf(stream, "| nul |\n\n");
    return;
  }
  fprintf(stream, "|");
  for (int idx = 0; idx <= v->idxs.a[v->idxs.n - 1]; idx++) {
    fprintf(stream, " %3d |", idx);
  }
  fprintf(stream, "\n|");
  for (int idx = 0; idx <= v->idxs.a[v->idxs.n - 1]; idx++) {
    fprintf(stream, "-----|");
  }
  fprintf(stream, "\n|");
  for (int idx = 0; idx <= v->idxs.a[v->idxs.n - 1]; idx++) {
    fprintf(stream, " %3d |", sv_get(v, idx));
  }
  fprintf(stream, "\n|");
  for (int idx = 0; idx <= v->idxs.a[v->idxs.n - 1]; idx++) {
    fprintf(stream, "=====|");
  }
  fprintf(stream, "\n\n");
}

void sv_densify(sparsevec *v, uint8_t *d) {
  if (kv_size(v->idxs) == 0) {
    return;
  }
  for (int idx = 0; idx < kv_size(v->idxs); idx++) {
    d[kv_A(v->idxs, idx)] = kv_A(v->vals, idx);
  }
}

void sv_copy(sparsevec *v, sparsevec *w) {
  kv_copy(uint16_t, v->idxs, w->idxs);
  kv_copy(uint8_t, v->vals, w->vals);
}

void sv_axpy(sparsevec *v, sparsevec *w, uint8_t u) {
  int i = 0, j = 0;
  int m = kv_size(v->idxs), n = kv_size(w->idxs);

  if (u == 0)
    return;

  if (u == 1)
    return sv_add(v, w);

  sparsevec za = sv_new(m + n);
  sparsevec *z = &za;

  const uint8_t *urow_hi = OCT_MUL_HI[u];
  const uint8_t *urow_lo = OCT_MUL_LO[u];
  while (i < m && j < n) {
    uint8_t w_lo = (kv_A(w->vals, j) & 0x0f);
    uint8_t w_hi = (kv_A(w->vals, j) & 0xf0) >> 4;

    if (kv_A(v->idxs, i) < kv_A(w->idxs, j)) {
      if (kv_A(v->vals, i) > 0) {
        kv_push(uint16_t, z->idxs, kv_A(v->idxs, i));
        kv_push(uint8_t, z->vals, kv_A(v->vals, i));
      }
      i++;
    } else if (kv_A(v->idxs, i) > kv_A(w->idxs, j)) {
      uint8_t val = urow_hi[w_hi] ^ urow_lo[w_lo];
      if (val > 0) {
        kv_push(uint16_t, z->idxs, kv_A(w->idxs, j));
        kv_push(uint8_t, z->vals, val);
      }
      j++;
    } else {
      uint8_t val = kv_A(v->vals, i) ^ urow_hi[w_hi] ^ urow_lo[w_lo];
      if (val > 0) {
        kv_push(uint16_t, z->idxs, kv_A(v->idxs, i));
        kv_push(uint8_t, z->vals, val);
      }
      i++;
      j++;
    }
  }

  while (i < m) {
    if (kv_A(v->vals, i) > 0) {
      kv_push(uint16_t, z->idxs, kv_A(v->idxs, i));
      kv_push(uint8_t, z->vals, kv_A(v->vals, i));
    }
    i++;
  }

  while (j < n) {
    uint8_t w_lo = (kv_A(w->vals, j) & 0x0f);
    uint8_t w_hi = (kv_A(w->vals, j) & 0xf0) >> 4;
    uint8_t val = urow_hi[w_hi] ^ urow_lo[w_lo];

    if (kv_A(w->vals, j) > 0) {
      kv_push(uint16_t, z->idxs, kv_A(w->idxs, j));
      kv_push(uint8_t, z->vals, val);
    }
    j++;
  }

  uint16_t *idxs = v->idxs.a;
  uint8_t *vals = v->vals.a;
  v->idxs.a = z->idxs.a;
  v->idxs.m = z->idxs.m;
  v->idxs.n = z->idxs.n;

  v->vals.a = z->vals.a;
  v->vals.m = z->vals.m;
  v->vals.n = z->vals.n;

  free(idxs);
  free(vals);
}

void sv_add(sparsevec *v, sparsevec *w) {
  int i = 0, j = 0;
  int m = kv_size(v->idxs), n = kv_size(w->idxs);
  sparsevec za = sv_new(m + n);
  sparsevec *z = &za;

  while (i < m && j < n) {
    if (kv_A(v->idxs, i) < kv_A(w->idxs, j)) {
      if (kv_A(v->vals, i) > 0) {
        kv_push(uint16_t, z->idxs, kv_A(v->idxs, i));
        kv_push(uint8_t, z->vals, kv_A(v->vals, i));
      }
      i++;
    } else if (kv_A(v->idxs, i) > kv_A(w->idxs, j)) {
      if (kv_A(w->vals, j) > 0) {
        kv_push(uint16_t, z->idxs, kv_A(w->idxs, j));
        kv_push(uint8_t, z->vals, kv_A(w->vals, j));
      }
      j++;
    } else {
      uint8_t o = kv_A(v->vals, i) ^ kv_A(w->vals, j);
      if (o > 0) {
        kv_push(uint16_t, z->idxs, kv_A(v->idxs, i));
        kv_push(uint8_t, z->vals, kv_A(v->vals, i) ^ kv_A(w->vals, j));
      }
      i++;
      j++;
    }
  }

  while (i < m) {
    if (kv_A(v->vals, i) > 0) {
      kv_push(uint16_t, z->idxs, kv_A(v->idxs, i));
      kv_push(uint8_t, z->vals, kv_A(v->vals, i));
    }
    i++;
  }

  while (j < n) {
    if (kv_A(w->vals, j) > 0) {
      kv_push(uint16_t, z->idxs, kv_A(w->idxs, j));
      kv_push(uint8_t, z->vals, kv_A(w->vals, j));
    }
    j++;
  }

  uint16_t *idxs = v->idxs.a;
  uint8_t *vals = v->vals.a;
  v->idxs.a = z->idxs.a;
  v->idxs.m = z->idxs.m;
  v->idxs.n = z->idxs.n;

  v->vals.a = z->vals.a;
  v->vals.m = z->vals.m;
  v->vals.n = z->vals.n;

  free(idxs);
  free(vals);
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

void sv_nnz(sparsevec *v, uint16_t s, uint16_t e, int *nnz, int *ones,
            int ones_at[]) {
  int sidx, eidx;

  *nnz = 0;
  *ones = 0;
  ones_at[0] = 0;
  ones_at[1] = 0;

  sidx = binsearch(v->idxs.a, 0, kv_size(v->idxs) - 1, s);
  eidx = binsearch(v->idxs.a, 0, kv_size(v->idxs) - 1, e);
  if (kv_size(v->idxs) > eidx && kv_A(v->idxs, eidx) == e) {
    eidx++;
  }

  for (int idx = sidx; idx < eidx; idx++) {
    if (kv_A(v->vals, idx) != 0) {
      *nnz += 1;
      if (kv_A(v->vals, idx) == 1) {
        *ones += 1;
        if (*ones <= 2) {
          ones_at[*ones - 1] = kv_A(v->idxs, idx) - s;
        }
      }
    }
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
    a.r[i] = sv_new(1); // FIXME
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
  sv_axpy(&a->r[ia], &b->r[ja], u);
}

void sm_addrow(sparsemat *a, sparsemat *b, uint16_t i, uint16_t j) {
  uint16_t ia = a->rl2a[i];
  uint16_t ja = b->rl2a[j];
  sv_add(&a->r[ia], &b->r[ja]);
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
  uint16_t ia = a->rl2a[i];
  sparsevec *v = &a->r[ia];

  *nnz = 0;
  *ones = 0;
  ones_at[0] = 0;
  ones_at[1] = 0;

  for (int idx = 0; idx < kv_size(v->idxs); idx++) {
    uint16_t j = kv_A(v->idxs, idx);
    uint16_t ja = a->cl2a[j];
    if (ja >= s && ja < e) {
      if (kv_A(v->vals, idx) != 0) {
        *nnz += 1;
        if (kv_A(v->vals, idx) == 1) {
          *ones += 1;
          if (*ones <= 2) {
            ones_at[*ones - 1] = kv_A(v->idxs, idx) - s;
          }
        }
      }
    }
  }
}

void sm_densify(sparsemat *a, uint8_t *d, uint16_t cols) {
  for (int i = 0; i < a->rows; i++) {
    uint8_t *dp = d + i * cols;
    uint16_t ia = a->rl2a[i];
    sparsevec *v = &a->r[ia];
    sv_densify(v, dp);
  }
}

void sm_print(sparsemat *a, FILE *stream) {
  fprintf(stream, "[%ux%u]\n", a->rows, a->cols);
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
