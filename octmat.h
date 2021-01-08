#ifndef OCTMAT_H
#define OCTMAT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef OCTMAT_ALIGN
#define OCTMAT_ALIGN 16
#endif

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

#endif
