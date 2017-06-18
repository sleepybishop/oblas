#include <emmintrin.h> /* sse2 */
#include <tmmintrin.h> /* sse3 */

#include "oblas.h"
#include "octmul_sse.h"

void ocopy(uint8_t *restrict a, uint8_t *restrict b, uint16_t i, uint16_t j,
           uint16_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  for (int idx = 0; idx < k; idx++) {
    ap[idx] = bp[idx];
  }
}

void oswaprow(uint8_t *restrict a, uint16_t i, uint16_t j, uint16_t k) {
  if (i == j)
    return;
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = a + (j * ALIGNED_COLS(k));

  for (int idx = 0; idx < k; idx++) {
    OCTET_SWAP(ap[idx], bp[idx]);
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

  for (int idx = 0; idx < (k / 16) * 16; idx += 16) {
    __m128i x0 = _mm_loadu_si128((__m128i *)(bp + idx));
    __m128i l0 = _mm_and_si128(x0, clr_mask);
    x0 = _mm_srli_epi64(x0, 4);
    __m128i h0 = _mm_and_si128(x0, clr_mask);
    l0 = _mm_shuffle_epi8(urow_lo, l0);
    h0 = _mm_shuffle_epi8(urow_hi, h0);

    __m128i *omg = (__m128i *)(ap + idx);
    _mm_storeu_si128(
        omg, _mm_xor_si128(_mm_loadu_si128(omg), _mm_xor_si128(l0, h0)));
  }

  for (int idx = (k / 16) * 16; idx < k; idx++) {
    ap[idx] ^= mtab_hi[bp[idx] >> 4] ^ mtab_lo[bp[idx] & 0xf];
  }
}

void oaddrow(uint8_t *restrict a, uint8_t *restrict b, uint16_t i, uint16_t j,
             uint16_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  for (int idx = 0; idx < (k / 16) * 16; idx += 16) {
    _mm_storeu_si128((__m128i *)(ap + idx),
                     _mm_xor_si128(_mm_loadu_si128((__m128i *)(ap + idx)),
                                   _mm_loadu_si128((__m128i *)(bp + idx))));
  }
  for (int idx = (k / 16) * 16; idx < k; idx++) {
    ap[idx] ^= bp[idx];
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

void ogemm(uint8_t *restrict a, uint8_t *restrict b, uint8_t *restrict c,
           uint16_t n, uint16_t k, uint16_t m) {
  octet *ap, *cp = c;

  for (int row = 0; row < n; row++, cp += ALIGNED_COLS(m)) {
    ap = a + (row * ALIGNED_COLS(k));

    for (int col = 0; col < m; col++)
      cp[col] = 0;

    for (int idx = 0; idx < k; idx++) {
      oaxpy(cp, b, 0, idx, m, ap[idx]);
    }
  }
}
