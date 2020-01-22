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

#define ALIGNED_COLS(k)                                                        \
  (((k) / OCTMAT_ALIGN) + (((k) % OCTMAT_ALIGN) ? 1 : 0)) * OCTMAT_ALIGN

typedef uint8_t octet;

void *oalloc(size_t nmemb, size_t size, size_t align);

void ocopy(uint8_t *a, uint8_t *b, uint16_t i, uint16_t j, uint16_t k);
void oswaprow(uint8_t *a, uint16_t i, uint16_t j, uint16_t k);
void oswapcol(uint8_t *a, uint16_t i, uint16_t j, uint16_t k, uint16_t l);
void oaxpy(uint8_t *a, uint8_t *b, uint16_t i, uint16_t j, uint16_t k,
           uint8_t u);
void oaddrow(uint8_t *a, uint8_t *b, uint16_t i, uint16_t j, uint16_t k);
void oscal(uint8_t *a, uint16_t i, uint16_t k, uint8_t u);
void ozero(uint8_t *restrict a, uint16_t i, size_t k);
void ogemm(uint8_t *a, uint8_t *b, uint8_t *c, uint16_t n, uint16_t k,
           uint16_t m);
int onnz(uint8_t *a, uint16_t i, uint16_t s, uint16_t e, uint16_t k);

#endif
