#include "util.h"
#include <errno.h>

void *oblas_alloc(size_t nmemb, size_t size, size_t align) {
  void *aligned = NULL;
  size_t aligned_sz = ((size / align) + ((size % align) ? 1 : 0)) * align;
  if (posix_memalign((void *)&aligned, align, nmemb * aligned_sz) != 0)
    exit(ENOMEM);
  return aligned;
}

uint32_t bfd_32(uint32_t word, uint8_t at, uint8_t len, unsigned val) {
  uint32_t mask = ((1 << len) - 1) << at;
  return (word & ~mask) | (val << at);
}

uint32_t bfx_32(uint32_t word, uint8_t at, uint8_t len) {
  return (word >> at) & ((1 << len) - 1);
}
