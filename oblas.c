#include "oblas.h"
#include <errno.h>

uint32_t bfd_32(uint32_t word, uint8_t at, uint8_t len, uint8_t val) {
  uint32_t mask = ((1 << len) - 1) << at;
  return (word & ~mask) | (val << at);
}

uint32_t bfx_32(uint32_t word, uint8_t at, uint8_t len) {
  return (word >> at) & ((1 << len) - 1);
}

void *oblas_alloc(size_t nmemb, size_t size, size_t align) {
  void *aligned = NULL;
  size_t aligned_sz = ((size / align) + ((size % align) ? 1 : 0)) * align;
  if (posix_memalign((void *)&aligned, align, nmemb * aligned_sz) != 0)
    exit(ENOMEM);
  return aligned;
}

void oblas_xor(uint8_t *a, uint8_t *b, size_t k) {
  for (int idx = 0; idx < k; idx++) {
    a[idx] ^= b[idx];
  }
}

void oblas_swap(uint8_t *a, uint8_t *b, size_t k) {
  for (int idx = 0; idx < k; idx++) {
    uint8_t __tmp = a[idx];
    a[idx] = b[idx];
    b[idx] = __tmp;
  }
}

#ifdef OBLAS_SSE
#include "oblas_sse.c"
#else
#ifdef OBLAS_AVX
#include "oblas_avx.c"
#else
#ifdef OBLAS_AVX512
#include "oblas_avx512.c"
#else
#ifdef OBLAS_NEON
#include "oblas_neon.c"
#else
#include "oblas_ref.c"
#endif
#endif
#endif
#endif
