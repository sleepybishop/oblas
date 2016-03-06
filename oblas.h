#ifndef OCTET_BLAS_H
#define OCTET_BLAS_H

#include <stdint.h>
#include "octmat.h"
#include "octtables.h"

#define OCTET_MUL(u, v) OCT_EXP[OCT_LOG[u] + OCT_LOG[v]]

#define OCTET_DIV(u, v) OCT_EXP[OCT_LOG[u] - OCT_LOG[v] + 255]

#define OCTET_SWAP(u, v)                                                       \
  do {                                                                         \
    uint8_t __tmp = (u);                                                       \
    (u) = (v);                                                                 \
    (v) = __tmp;                                                               \
  } while (0)

void ocopy(uint8_t *a, uint8_t *b, uint16_t i, uint16_t j, uint16_t k);
void oswaprow(uint8_t *a, uint16_t i, uint16_t j, uint16_t k);
void oswapcol(uint8_t *a, uint16_t i, uint16_t j, uint16_t k, uint16_t l);
void oaxpy(uint8_t *a, uint8_t *b, uint16_t i, uint16_t j, uint16_t k,
           uint8_t u);
void oaddrow(uint8_t *a, uint8_t *b, uint16_t i, uint16_t j, uint16_t k);
void odivrow(uint8_t *a, uint16_t i, uint16_t k, uint8_t u);
void ogemm(uint8_t *a, uint8_t *b, uint8_t *c, uint16_t n, uint16_t k,
           uint16_t m);

#endif
