#include <emmintrin.h> /* sse2 */
#include <tmmintrin.h> /* sse3 */

#include "oblas.h"
#include "octmul_hilo.h"

void ocopy(uint8_t *restrict a, uint8_t *restrict b, uint16_t i, uint16_t j,
           uint16_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  __m128i *ap128 = (__m128i *)ap;
  __m128i *bp128 = (__m128i *)bp;
  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    _mm_storeu_si128(ap128++, _mm_loadu_si128(bp128++));
  }
}

void oswaprow(uint8_t *restrict a, uint16_t i, uint16_t j, uint16_t k) {
  if (i == j)
    return;
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = a + (j * ALIGNED_COLS(k));

  __m128i *ap128 = (__m128i *)ap;
  __m128i *bp128 = (__m128i *)bp;
  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    __m128i atmp = _mm_loadu_si128((__m128i *)(ap128));
    __m128i btmp = _mm_loadu_si128((__m128i *)(bp128));
    _mm_storeu_si128(ap128++, btmp);
    _mm_storeu_si128(bp128++, atmp);
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

  const __m128i mask = _mm_set1_epi8(0x0f);
  const __m128i urow_hi = _mm_loadu_si128((__m128i *)OCT_MUL_HI[u]);
  const __m128i urow_lo = _mm_loadu_si128((__m128i *)OCT_MUL_LO[u]);

  __m128i *ap128 = (__m128i *)ap;
  __m128i *bp128 = (__m128i *)bp;
  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    __m128i bx = _mm_loadu_si128(bp128++);
    __m128i lo = _mm_and_si128(bx, mask);
    bx = _mm_srli_epi64(bx, 4);
    __m128i hi = _mm_and_si128(bx, mask);
    lo = _mm_shuffle_epi8(urow_lo, lo);
    hi = _mm_shuffle_epi8(urow_hi, hi);

    _mm_storeu_si128(
        ap128, _mm_xor_si128(_mm_loadu_si128(ap128), _mm_xor_si128(lo, hi)));
    ap128++;
  }
}

void oaddrow(uint8_t *restrict a, uint8_t *restrict b, uint16_t i, uint16_t j,
             uint16_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  __m128i *ap128 = (__m128i *)ap;
  __m128i *bp128 = (__m128i *)bp;
  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    _mm_storeu_si128(
        ap128, _mm_xor_si128(_mm_loadu_si128(ap128), _mm_loadu_si128(bp128)));
    ap128++;
    bp128++;
  }
}

void oscal(uint8_t *restrict a, uint16_t i, uint16_t k, uint8_t u) {
  octet *ap = a + (i * ALIGNED_COLS(k));

  if (u < 2)
    return;

  const __m128i mask = _mm_set1_epi8(0x0f);
  const __m128i urow_hi = _mm_loadu_si128((__m128i *)OCT_MUL_HI[u]);
  const __m128i urow_lo = _mm_loadu_si128((__m128i *)OCT_MUL_LO[u]);

  __m128i *ap128 = (__m128i *)ap;
  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    __m128i ax = _mm_loadu_si128(ap128);
    __m128i lo = _mm_and_si128(ax, mask);
    ax = _mm_srli_epi64(ax, 4);
    __m128i hi = _mm_and_si128(ax, mask);
    lo = _mm_shuffle_epi8(urow_lo, lo);
    hi = _mm_shuffle_epi8(urow_hi, hi);

    _mm_storeu_si128(ap128++, _mm_xor_si128(lo, hi));
  }
}

void ozero(uint8_t *restrict a, uint16_t i, size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  __m128i *ap128 = (__m128i *)ap;
  __m128i z128 = _mm_setzero_si128();

  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    _mm_storeu_si128(ap128++, z128);
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
  __m128i *ap128 = (__m128i *)ap;
  __m128i z128 = _mm_setzero_si128();
  __m128i o128 = _mm_set1_epi8(1);

  int s1 = s - (s % OCTMAT_ALIGN);
  int b1 = s - s1;              // bits to ignore from start
  int b2 = ALIGNED_COLS(e) - e; // bits to ignore from end
  int firstmask = (1 << b1) - 1;
  int lastmask = ((1 << b2) - 1) << (OCTMAT_ALIGN - b2);

  ap128 += s1 / OCTMAT_ALIGN;
  for (int idx = s1; idx < ALIGNED_COLS(e); idx += OCTMAT_ALIGN) {
    __m128i atmp = _mm_loadu_si128((__m128i *)(ap128++));
    __m128i zeroes = _mm_cmpeq_epi8(atmp, z128);
    __m128i oneses = _mm_cmpeq_epi8(atmp, o128);

    int zeromask = _mm_movemask_epi8(zeroes) | 0xffff0000;
    int onesmask = _mm_movemask_epi8(oneses) | 0xffff0000;
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
