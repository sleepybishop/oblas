#ifndef OBLAS_H
#define OBLAS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GF_ADD(a, b) ((a) ^ (b))
#define GF_SUB(a, b) ((a) ^ (b))
#define GF_MUL(t, a, b)                                                        \
  ((a == 0 || b == 0) ? 0 : (t##_EXP[t##_LOG[a] + t##_LOG[b]]))
#define GF_DIV(t, a, b, s) (t##_EXP[t##_LOG[a] - t##_LOG[b] + (s - 1)])

#define oblas_word uint32_t

uint32_t bfd_32(uint32_t word, uint8_t at, uint8_t len, uint8_t val);
uint32_t bfx_32(uint32_t word, uint8_t at, uint8_t len);

void *oblas_alloc(size_t nmemb, size_t size, size_t align);

#define oblas_free(p) free(p);
#define oblas_zero(p, k) memset(p, 0, k);
#define oblas_copy(p, q, k) memcpy(p, q, k);

void oblas_xor(uint8_t *dst, uint8_t *src, size_t k);
void oblas_axpy(uint8_t *dst, const uint8_t *src, size_t k, const uint8_t *u_lo,
                const uint8_t *u_hi);
void oblas_scal(uint8_t *dst, size_t k, const uint8_t *u_lo,
                const uint8_t *u_hi);
void oblas_swap(uint8_t *dst, uint8_t *src, size_t k);

void oblas_axpy_gf2_gf256_32(uint8_t *a, uint32_t *b, size_t k, uint8_t u);

#endif
