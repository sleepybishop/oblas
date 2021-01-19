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
  unsigned rows;
  unsigned cols;
  unsigned stride;
  uint8_t *bits;
} octmat;

#define om_R(v, x) ((v)->bits + ((x) * (v)->stride))
#define om_P(v) om_R(v, 0)
#define om_A(v, x, y) (om_R(v, x)[(y)])

octmat *octmat_new(unsigned rows, unsigned cols);
void octmat_free(octmat *m);

void oswaprow(octmat *a, unsigned i, unsigned j);
void oaxpy(octmat *a, octmat *b, unsigned i, unsigned j, uint8_t u);
void oaddrow(octmat *a, octmat *b, unsigned i, unsigned j);
void oscal(octmat *a, unsigned i, uint8_t u);
void oaxpy_b32(octmat *a, uint32_t *b, unsigned i, uint8_t u);

#endif
