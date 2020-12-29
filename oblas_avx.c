#include <immintrin.h> /* AVX */

#include "oblas.h"

void ocopy(uint8_t *restrict a, uint8_t *restrict b, size_t i, size_t j,
           size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  __m256i *ap256 = (__m256i *)ap;
  __m256i *bp256 = (__m256i *)bp;
  for (size_t idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    _mm256_storeu_si256(ap256++, _mm256_loadu_si256(bp256++));
  }
}

void oswaprow(uint8_t *restrict a, size_t i, size_t j, size_t k) {
  if (i == j)
    return;
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = a + (j * ALIGNED_COLS(k));

  __m256i *ap256 = (__m256i *)ap;
  __m256i *bp256 = (__m256i *)bp;
  for (size_t idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    __m256i atmp = _mm256_loadu_si256((__m256i *)(ap256));
    __m256i btmp = _mm256_loadu_si256((__m256i *)(bp256));
    _mm256_storeu_si256(ap256++, btmp);
    _mm256_storeu_si256(bp256++, atmp);
  }
}

void oswapcol(octet *restrict a, size_t i, size_t j, size_t k, size_t l) {
  if (i == j)
    return;
  octet *ap = a;

  for (size_t idx = 0; idx < k; idx++, ap += ALIGNED_COLS(l)) {
    OCTET_SWAP(ap[i], ap[j]);
  }
}

void oaxpy(uint8_t *restrict a, uint8_t *restrict b, size_t i, size_t j,
           size_t k, uint8_t u) {
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
  for (size_t idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
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

void oaddrow(uint8_t *restrict a, uint8_t *restrict b, size_t i, size_t j,
             size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  __m256i *ap256 = (__m256i *)ap;
  __m256i *bp256 = (__m256i *)bp;
  for (size_t idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    _mm256_storeu_si256(ap256, _mm256_xor_si256(_mm256_loadu_si256(ap256),
                                                _mm256_loadu_si256(bp256)));
    ap256++;
    bp256++;
  }
}

void oscal(uint8_t *restrict a, size_t i, size_t k, uint8_t u) {
  octet *ap = a + (i * ALIGNED_COLS(k));

  if (u < 2)
    return;

  const __m256i mask = _mm256_set1_epi8(0x0f);
  const __m256i urow_hi =
      _mm256_loadu2_m128i((__m128i *)OCT_MUL_HI[u], (__m128i *)OCT_MUL_HI[u]);
  const __m256i urow_lo =
      _mm256_loadu2_m128i((__m128i *)OCT_MUL_LO[u], (__m128i *)OCT_MUL_LO[u]);

  __m256i *ap256 = (__m256i *)ap;
  for (size_t idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    __m256i ax = _mm256_loadu_si256(ap256);
    __m256i lo = _mm256_and_si256(ax, mask);
    ax = _mm256_srli_epi64(ax, 4);
    __m256i hi = _mm256_and_si256(ax, mask);
    lo = _mm256_shuffle_epi8(urow_lo, lo);
    hi = _mm256_shuffle_epi8(urow_hi, hi);

    _mm256_storeu_si256(ap256++, _mm256_xor_si256(lo, hi));
  }
}

void ozero(uint8_t *restrict a, size_t i, size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  __m256i *ap256 = (__m256i *)ap;
  __m256i z256 = _mm256_setzero_si256();

  for (size_t idx = 0; idx < ALIGNED_COLS(k); idx += OCTMAT_ALIGN) {
    _mm256_storeu_si256(ap256++, z256);
  }
}

void ogemm(uint8_t *restrict a, uint8_t *restrict b, uint8_t *restrict c,
           size_t n, size_t k, size_t m) {
  octet *ap, *cp = c;

  for (size_t row = 0; row < n; row++, cp += ALIGNED_COLS(m)) {
    ap = a + (row * ALIGNED_COLS(k));

    ozero(cp, 0, m);
    for (size_t idx = 0; idx < k; idx++) {
      oaxpy(cp, b, 0, idx, m, ap[idx]);
    }
  }
}

size_t onnz(uint8_t *a, size_t i, size_t s, size_t e, size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  size_t nz = 0;
  for (size_t idx = s; idx < e; idx++) {
    nz += (ap[idx] != 0);
  }
  return nz;
}

void oaxpy_b32(uint8_t *a, uint32_t *b, size_t i, size_t k, uint8_t u) {
  __m256i *ap256 = (__m256i *)(a + i * ALIGNED_COLS(k));
  __m256i scatter =
      _mm256_set_epi32(0x03030303, 0x03030303, 0x02020202, 0x02020202,
                       0x01010101, 0x01010101, 0x00000000, 0x00000000);
  __m256i cmpmask =
      _mm256_set_epi32(0x80402010, 0x08040201, 0x80402010, 0x08040201,
                       0x80402010, 0x08040201, 0x80402010, 0x08040201);
  __m256i u256 = _mm256_set1_epi8(u);

  for (size_t idx = 0, p = 0; idx < k;
       idx += 8 * sizeof(uint32_t), p++, ap256++) {
    __m256i bcast = _mm256_set1_epi32(b[p]);
    __m256i bytes = _mm256_shuffle_epi8(bcast, scatter);

    bytes = _mm256_andnot_si256(bytes, cmpmask);
    bytes = _mm256_and_si256(_mm256_cmpeq_epi8(bytes, _mm256_setzero_si256()),
                             u256);
    _mm256_storeu_si256(ap256,
                        _mm256_xor_si256(_mm256_loadu_si256(ap256), bytes));
  }
}
