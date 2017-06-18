#include "oblas.h"

#define ALIGNED_COLS(k)                                                        \
  (((k) / OCTMAT_ALIGN) + (((k) % OCTMAT_ALIGN) ? 1 : 0)) * OCTMAT_ALIGN
typedef uint8_t octet;

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
