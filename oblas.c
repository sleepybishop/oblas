#include "oblas.h"
#include <errno.h>

void *oalloc(size_t nmemb, size_t size, size_t align) {
  void *aligned = NULL;
  size_t aligned_sz = ((size / align) + ((size % align) ? 1 : 0)) * align;

  if (posix_memalign((void *)&aligned, align, nmemb * aligned_sz) != 0) {
    exit(ENOMEM);
  }
  return aligned;
}

#ifdef OBLAS_SSE
#include "oblas_sse.c"
#else
#ifdef OBLAS_AVX
#include "oblas_avx.c"
#else
#ifdef OBLAS_NEON
#include "oblas_neon.c"
#else
#include "oblas_classic.c"
#endif
#endif
#endif
