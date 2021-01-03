#ifndef OBLAS_UTIL_H
#define OBLAS_UTIL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void *oblas_alloc(size_t nmemb, size_t size, size_t align);

#define oblas_free(p) free(p);
#define oblas_zero(p, k) memset(p, 0, k);

void oblas_xor(uint8_t *dst, uint8_t *src, size_t k);

uint32_t bfd_32(uint32_t word, uint8_t at, uint8_t len, unsigned val);
uint32_t bfx_32(uint32_t word, uint8_t at, uint8_t len);

#endif
