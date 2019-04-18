#include <immintrin.h> /* AVX */

#include "oblas.h"
#include "octmul_hilo.h"

/* GCC doesn't include some intrinsics */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC)
static inline __m256i __attribute__((__always_inline__))
_mm256_loadu2_m128i(const __m128i *const hiaddr, const __m128i *const loaddr) {
  return _mm256_inserti128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128(loaddr)), _mm_loadu_si128(hiaddr),
      1);
}
#endif

void ocopy(uint8_t *restrict a, uint8_t *restrict b, uint16_t i, uint16_t j,
           uint16_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  __m256i *ap256 = (__m256i *)ap;
  __m256i *bp256 = (__m256i *)bp;
  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    _mm256_storeu_si256(ap256++, _mm256_loadu_si256(bp256++));
  }
}

void oswaprow(uint8_t *restrict a, uint16_t i, uint16_t j, uint16_t k) {
  if (i == j)
    return;
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = a + (j * ALIGNED_COLS(k));

  __m256i *ap256 = (__m256i *)ap;
  __m256i *bp256 = (__m256i *)bp;
  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    __m256i atmp = _mm256_loadu_si256((__m256i *)(ap256));
    __m256i btmp = _mm256_loadu_si256((__m256i *)(bp256));
    _mm256_storeu_si256(ap256++, btmp);
    _mm256_storeu_si256(bp256++, atmp);
  }
}

void oswapcol(octet *restrict a, uint16_t i, uint16_t j, uint16_t k,
              uint16_t l) {
  if (i == j)
    return;
  octet *ap = a;

  for (int idx = 0; idx < k; idx++, ap += ALIGNED_COLS(l)) {
    OCTET_SWAP(ap[i], ap[j]);
  }
}

void oaxpy(uint8_t *restrict a, uint8_t *restrict b, uint16_t i, uint16_t j,
           uint16_t k, uint8_t u) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  if (u == 0)
    return;

  if (u == 1)
    return oaddrow(a, b, i, j, k);

  const __m256i mask = _mm256_set1_epi8(0x0f);
  const __m256i urow_hi =
      _mm256_loadu2_m128i((__m128i *)OCT_MUL_HI[u], (__m128i *)OCT_MUL_HI[u]);
  const __m256i urow_lo =
      _mm256_loadu2_m128i((__m128i *)OCT_MUL_LO[u], (__m128i *)OCT_MUL_LO[u]);

  __m256i *ap256 = (__m256i *)ap;
  __m256i *bp256 = (__m256i *)bp;
  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    __m256i bx = _mm256_loadu_si256(bp256++);
    __m256i lo = _mm256_and_si256(bx, mask);
    bx = _mm256_srli_epi64(bx, 4);
    __m256i hi = _mm256_and_si256(bx, mask);
    lo = _mm256_shuffle_epi8(urow_lo, lo);
    hi = _mm256_shuffle_epi8(urow_hi, hi);

    _mm256_storeu_si256(ap256, _mm256_xor_si256(_mm256_loadu_si256(ap256),
                                                _mm256_xor_si256(lo, hi)));
    ap256++;
  }
}

void oaddrow(uint8_t *restrict a, uint8_t *restrict b, uint16_t i, uint16_t j,
             uint16_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  __m256i *ap256 = (__m256i *)ap;
  __m256i *bp256 = (__m256i *)bp;
  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    _mm256_storeu_si256(ap256, _mm256_xor_si256(_mm256_loadu_si256(ap256),
                                                _mm256_loadu_si256(bp256)));
    ap256++;
    bp256++;
  }
}

void oscal(uint8_t *restrict a, uint16_t i, uint16_t k, uint8_t u) {
  octet *ap = a + (i * ALIGNED_COLS(k));

  if (u < 2)
    return;

  const __m256i mask = _mm256_set1_epi8(0x0f);
  const __m256i urow_hi =
      _mm256_loadu2_m128i((__m128i *)OCT_MUL_HI[u], (__m128i *)OCT_MUL_HI[u]);
  const __m256i urow_lo =
      _mm256_loadu2_m128i((__m128i *)OCT_MUL_LO[u], (__m128i *)OCT_MUL_LO[u]);

  __m256i *ap256 = (__m256i *)ap;
  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    __m256i ax = _mm256_loadu_si256(ap256);
    __m256i lo = _mm256_and_si256(ax, mask);
    ax = _mm256_srli_epi64(ax, 4);
    __m256i hi = _mm256_and_si256(ax, mask);
    lo = _mm256_shuffle_epi8(urow_lo, lo);
    hi = _mm256_shuffle_epi8(urow_hi, hi);

    _mm256_storeu_si256(ap256++, _mm256_xor_si256(lo, hi));
  }
}

void ozero(uint8_t *restrict a, uint16_t i, size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  __m256i *ap256 = (__m256i *)ap;
  __m256i z256 = _mm256_setzero_si256();

  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    _mm256_storeu_si256(ap256++, z256);
  }
}

void ogemm(uint8_t *restrict a, uint8_t *restrict b, uint8_t *restrict c,
           uint16_t n, uint16_t k, uint16_t m) {
  octet *ap, *cp = c;

  for (int row = 0; row < n; row++, cp += ALIGNED_COLS(m)) {
    ap = a + (row * ALIGNED_COLS(k));

    ozero(cp, 0, m);
    for (int idx = 0; idx < k; idx++) {
      oaxpy(cp, b, 0, idx, m, ap[idx]);
    }
  }
}

void onnz(uint8_t *a, uint16_t i, uint16_t s, uint16_t e, uint16_t k, int *nnz,
          int *ones, int ones_idx[]) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  __m256i *ap256 = (__m256i *)ap;
  __m256i z256 = _mm256_setzero_si256();
  __m256i o256 = _mm256_set1_epi8(1);

  int s1 = s - (s % OCTMAT_ALIGN);
  int b1 = s - s1;              // bits to ignore from start
  int b2 = ALIGNED_COLS(e) - e; // bits to ignore from end
  int firstmask = (1 << b1) - 1;
  int lastmask = ((1 << b2) - 1) << (OCTMAT_ALIGN - b2);

  ap256 += s1 / OCTMAT_ALIGN;
  for (int idx = s1; idx < ALIGNED_COLS(e); idx += OCTMAT_ALIGN) {
    __m256i atmp = _mm256_loadu_si256((__m256i *)(ap256++));
    __m256i zeroes = _mm256_cmpeq_epi8(atmp, z256);
    __m256i oneses = _mm256_cmpeq_epi8(atmp, o256);

    int zeromask = _mm256_movemask_epi8(zeroes);
    int onesmask = _mm256_movemask_epi8(oneses);
    if (idx == s1) {
      zeromask |= firstmask;
      onesmask |= firstmask;
    }
    if (idx == ALIGNED_COLS(e) - OCTMAT_ALIGN) {
      zeromask |= lastmask;
      onesmask |= lastmask;
    }

    *nnz += __builtin_popcountll(~zeromask);
    *ones += __builtin_popcountll(~onesmask);
    if (*ones > 0) {
      int oneat = __builtin_ctz(~onesmask);
      ones_idx[0] = idx + oneat - s;
      if (*ones > 1) {
        onesmask |= 1 << oneat;
        ones_idx[1] = idx + __builtin_ctz(~onesmask) - s;
      }
    }
  }
}
