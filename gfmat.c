#include "gfmat.h"
#include "oblas.h"
#include <errno.h>

#include "gf2_2_tables.h"
#include "gf2_4_tables.h"
#include "gf2_8_tables.h"

#ifndef OCTMAT_ALIGN
#define OCTMAT_ALIGN 16
#endif

static const gf fields[] = {
    {
        .field = GF2_1,
        .exp = 1,
        .len = 1 << 1,
        .vpw = (sizeof(oblas_word) * 8),
    },
    {
        .field = GF2_2,
        .exp = 2,
        .len = 1 << 2,
        .vpw = (sizeof(oblas_word) * 4),
        .shuf_lo = GF2_2_SHUF_LO,
        .shuf_hi = GF2_2_SHUF_HI,
    },
    {
        .field = GF2_4,
        .exp = 4,
        .len = 1 << 4,
        .vpw = (sizeof(oblas_word) * 2),
        .shuf_lo = GF2_4_SHUF_LO,
        .shuf_hi = GF2_4_SHUF_HI,
    },
    {
        .field = GF2_8,
        .exp = 8,
        .len = 1 << 8,
        .vpw = (sizeof(oblas_word) * 1),
        .shuf_lo = GF2_8_SHUF_LO,
        .shuf_hi = GF2_8_SHUF_HI,
    },
};

#define ALIGN_TO(k, a) (((k) / (a)) + (((k) % (a)) ? 1 : 0)) * (a)

gfmat *gfmat_new(gf_field field, unsigned rows, unsigned cols) {
  gfmat *m = calloc(1, sizeof(gfmat));
  unsigned vpw = fields[field].vpw;
  m->field = field;
  m->rows = rows;
  m->cols = cols;
  m->stride = ((cols / vpw) + ((cols % vpw) ? 1 : 0));
  m->align = (field == GF2_1) ? sizeof(void *) : OCTMAT_ALIGN;
  m->stride = ALIGN_TO(m->stride, m->align);
  m->bits = oblas_alloc(rows, m->stride * sizeof(oblas_word), m->align);
  oblas_zero(m->bits, rows * m->stride * sizeof(oblas_word));
  return m;
}

gfmat *gfmat_copy(gfmat *_m) {
  if (!_m || !_m->bits)
    return NULL;
  gfmat *m = calloc(1, sizeof(gfmat));
  m->field = _m->field;
  m->rows = _m->rows;
  m->cols = _m->cols;
  m->stride = _m->stride;
  m->align = _m->align;
  m->bits = oblas_alloc(m->rows, m->stride * sizeof(oblas_word), m->align);
  memcpy(m->bits, _m->bits, sizeof(oblas_word) * m->stride * m->cols);
  return m;
}

void gfmat_free(gfmat *m) {
  if (m && m->bits)
    oblas_free(m->bits);
  if (m)
    free(m);
}

uint8_t gfmat_get(gfmat *m, unsigned i, unsigned j) {
  if (i >= m->rows || j >= m->cols)
    return 0;
  gf field = fields[m->field];
  oblas_word *a = m->bits + i * m->stride;
  return bfx_32(a[j / field.vpw], (j % field.vpw) * field.exp, field.exp);
}

void gfmat_set(gfmat *m, unsigned i, unsigned j, uint8_t b) {
  if (i >= m->rows || j >= m->cols)
    return;
  gf field = fields[m->field];
  oblas_word *a = m->bits + i * m->stride;
  unsigned q = j / field.vpw;
  a[q] = bfd_32(a[q], (j % field.vpw) * field.exp, field.exp, b % field.len);
}

void gfmat_print(gfmat *m, FILE *stream) {
  char *fn[] = {"gf2", "gf4", "gf16", "gf256"};
  fprintf(stream, "%s [%ux%u]\n", fn[m->field], (unsigned)m->rows,
          (unsigned)m->cols);
  fprintf(stream, "|     ");
  for (unsigned j = 0; j < m->cols; j++) {
    fprintf(stream, "| %03d ", j);
  }
  fprintf(stream, "|\n");
  for (unsigned i = 0; i < m->rows; i++) {
    fprintf(stream, "| %03d | %3d ", i, gfmat_get(m, i, 0));
    for (unsigned j = 1; j < m->cols; j++) {
      fprintf(stream, "| %3d ", gfmat_get(m, i, j));
    }
    fprintf(stream, "|\n");
  }
}

void gfmat_add(gfmat *a, gfmat *b, unsigned i, unsigned j) {
  oblas_word *ap = a->bits + i * a->stride;
  oblas_word *bp = b->bits + j * b->stride;
  oblas_xor((uint8_t *)ap, (uint8_t *)bp, a->stride * sizeof(oblas_word));
}

void gfmat_swaprow(gfmat *m, unsigned i, unsigned j) {
  if (i == j)
    return;

  oblas_word *a = m->bits + i * m->stride;
  oblas_word *b = m->bits + j * m->stride;
  oblas_swap((uint8_t *)a, (uint8_t *)b, m->stride * sizeof(oblas_word));
}

void gfmat_axpy(gfmat *a, gfmat *b, unsigned i, unsigned j, uint8_t u) {
  oblas_word *ap = a->bits + i * a->stride;
  oblas_word *bp = b->bits + j * b->stride;

  if (u == 0)
    return;
  if (u == 1)
    oblas_xor((uint8_t *)ap, (uint8_t *)bp, a->stride * sizeof(oblas_word));
  else
    oblas_axpy((uint8_t *)ap, (uint8_t *)bp, a->stride * sizeof(oblas_word),
               fields[a->field].shuf_lo + (u * 16),
               fields[a->field].shuf_hi + (u * 16));
}

void gfmat_scal(gfmat *a, unsigned i, uint8_t u) {
  oblas_word *ap = a->bits + i * a->stride;

  if (u < 2)
    return;

  oblas_scal((uint8_t *)ap, a->stride * sizeof(oblas_word),
             fields[a->field].shuf_lo + (u * 16),
             fields[a->field].shuf_hi + (u * 16));
}

void gfmat_zero(gfmat *a, unsigned i) {
  oblas_word *ap = a->bits + i * a->stride;
  oblas_zero((uint8_t *)ap, a->stride * sizeof(oblas_word));
}

void gfmat_fill(gfmat *m, unsigned i, uint8_t *dst) {
  gf field = fields[m->field];
  oblas_word *a = m->bits + i * m->stride;
  for (unsigned idx = 0; idx < m->stride; idx++) {
    oblas_word tmp = a[idx];
    while (tmp > 0) {
      unsigned tz = __builtin_ctz(tmp);
      unsigned q = tz / field.exp;
      tmp = tmp & (tmp - 1);
      dst[q + idx * field.vpw] |= 1 << (tz % field.vpw);
    }
  }
}

unsigned gfmat_nnz(gfmat *m, unsigned i, unsigned s, unsigned e) {
  if (i >= m->rows || s < 0 || s > e || e > (m->cols + 1))
    return 0;
  gf field = fields[m->field];
  oblas_word *a = m->bits + i * m->stride;
  unsigned nnz = 0;
  unsigned sq = s / field.vpw, eq = e / field.vpw;
  unsigned sr = s % field.vpw, er = e % field.vpw;
  oblas_word masks[2] = {~((1 << sr) - 1), ((1 << er) - 1)};

  if (1) {
    oblas_word tmp = a[sq], z = 0;
    while (tmp > 0) {
      z = z | 1 << (__builtin_ctz(tmp) / field.exp);
      tmp = tmp & (tmp - 1);
    }
    nnz += __builtin_popcount(z & masks[0]);
  }
  for (unsigned idx = sq + 1; idx < eq; idx++) {
    oblas_word tmp = a[idx], z = 0;
    while (tmp > 0) {
      z = z | 1 << (__builtin_ctz(tmp) / field.exp);
      tmp = tmp & (tmp - 1);
    }
    nnz += __builtin_popcount(z);
  }
  if (e > eq) {
    oblas_word tmp = a[eq], z = 0;
    while (tmp > 0) {
      z = z | 1 << (__builtin_ctz(tmp) / field.exp);
      tmp = tmp & (tmp - 1);
    }
    nnz += __builtin_popcount(z & masks[1]);
  }
  return nnz;
}

void gfmat_swapcol(gfmat *m, unsigned i, unsigned j) {
  if (i == j)
    return;

  gf field = fields[m->field];
  oblas_word *a = m->bits;
  unsigned p = i / field.vpw;
  unsigned q = j / field.vpw;
  unsigned p_at = (i % field.vpw) * field.exp;
  unsigned q_at = (j % field.vpw) * field.exp;
  for (unsigned r = 0; r < m->rows; r++, p += m->stride, q += m->stride) {
    uint8_t ival = bfx_32(a[p], p_at, field.exp);
    uint8_t jval = bfx_32(a[q], q_at, field.exp);
    a[p] = bfd_32(a[p], p_at, field.exp, jval);
    a[q] = bfd_32(a[q], q_at, field.exp, ival);
  }
}
