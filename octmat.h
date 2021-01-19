#ifndef OCTMAT_H
#define OCTMAT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gf2_8_tables.h"

#ifndef OCTMAT_ALIGN
#define OCTMAT_ALIGN 16
#endif

#define OCT_LOG GF2_8_LOG
#define OCT_EXP GF2_8_EXP
#define OCT_INV GF2_8_INV

typedef struct {
  uint8_t *data;
  size_t rows;
  size_t cols;
  size_t cols_al;
} octmat;

#define OM_INITIAL                                                             \
  { .rows = 0, .cols = 0, .cols_al = 0, .data = 0 }

#define om_R(v, x) ((v).data + ((x) * (v).cols_al))
#define om_P(v) om_R(v, 0)
#define om_A(v, x, y) (om_R(v, x)[(y)])

void om_resize(octmat *v, size_t rows, size_t cols);
void om_destroy(octmat *v);

void oswaprow(uint8_t *a, size_t i, size_t j, size_t k);
void oaxpy(uint8_t *a, uint8_t *b, size_t i, size_t j, size_t k, uint8_t u);
void oaddrow(uint8_t *a, uint8_t *b, size_t i, size_t j, size_t k);
void oscal(uint8_t *a, size_t i, size_t k, uint8_t u);
void oaxpy_b32(uint8_t *a, uint32_t *b, size_t i, size_t k, uint8_t u);

#endif
