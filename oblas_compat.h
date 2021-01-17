#ifndef OBLAS_COMPAT_H
#define OBLAS_COMPAT_H

#include <stdint.h>

#include "gf2_8_tables.h"
#include "octmat.h"

#define OCT_LOG GF2_8_LOG
#define OCT_EXP GF2_8_EXP
#define OCT_INV GF2_8_INV

#define ALIGNED_COLS(k)                                                        \
  (((k) / OCTMAT_ALIGN) + (((k) % OCTMAT_ALIGN) ? 1 : 0)) * OCTMAT_ALIGN

void ocopy(uint8_t *a, uint8_t *b, size_t i, size_t j, size_t k);
void oswaprow(uint8_t *a, size_t i, size_t j, size_t k);
void oaxpy(uint8_t *a, uint8_t *b, size_t i, size_t j, size_t k, uint8_t u);
void oaddrow(uint8_t *a, uint8_t *b, size_t i, size_t j, size_t k);
void oscal(uint8_t *a, size_t i, size_t k, uint8_t u);
void ozero(uint8_t *restrict a, size_t i, size_t k);
void ogemm(uint8_t *a, uint8_t *b, uint8_t *c, size_t n, size_t k, size_t m);
void oaxpy_b32(uint8_t *a, uint32_t *b, size_t i, size_t k, uint8_t u);

#endif
