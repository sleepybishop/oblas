#include <emmintrin.h> /* sse2 */
#include <tmmintrin.h> /* sse3 */

#include "oblas.h"
#include "octmul_sse.h"

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

  const octet *mtab_hi = OCT_MUL_HI[u];
  const octet *mtab_lo = OCT_MUL_LO[u];

  const __m128i clr_mask = _mm_set1_epi8(0x0f);
  const __m128i urow_hi = _mm_loadu_si128((__m128i *)mtab_hi);
  const __m128i urow_lo = _mm_loadu_si128((__m128i *)mtab_lo);

  __m128i *ap128 = (__m128i *)ap;
  __m128i *bp128 = (__m128i *)bp;
  for (int idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    __m128i x0 = _mm_loadu_si128(bp128++);
    __m128i l0 = _mm_and_si128(x0, clr_mask);
    x0 = _mm_srli_epi64(x0, 4);
    __m128i h0 = _mm_and_si128(x0, clr_mask);
    l0 = _mm_shuffle_epi8(urow_lo, l0);
    h0 = _mm_shuffle_epi8(urow_hi, h0);

    _mm_storeu_si128(
        ap128, _mm_xor_si128(_mm_loadu_si128(ap128), _mm_xor_si128(l0, h0)));
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

void odivrow(uint8_t *restrict a, uint16_t i, uint16_t k, uint8_t u) {
  octet *ap = a + (i * ALIGNED_COLS(k));

  if (u == 0)
    return;

  octet u_log = OCT_LOG[u];
  for (int idx = 0; idx < k; idx++) {
    if (ap[idx] == 0)
      continue;
    ap[idx] = OCT_EXP[OCT_LOG[ap[idx]] - u_log + 255];
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
