#include "oblas.h"
#include "util.h"
#include <errno.h>

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
