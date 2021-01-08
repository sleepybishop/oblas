#include <immintrin.h> /* AVX */

void oblas_xor(uint8_t *a, uint8_t *b, size_t k) {
  __m256i *ap = (__m256i *)a, *ae = (__m256i *)(a + k);
  __m256i *bp = (__m256i *)b;
  for (; ap < ae; ap++, bp++) {
    _mm256_storeu_si256(
        ap, _mm256_xor_si256(_mm256_loadu_si256(ap), _mm256_loadu_si256(bp)));
  }
}

void oblas_axpy(uint8_t *a, const uint8_t *b, size_t k, const uint8_t *u_lo,
                const uint8_t *u_hi) {
  const __m256i mask = _mm256_set1_epi8(0x0f);
  const __m256i urow_lo = _mm256_loadu2_m128i((__m128i *)u_lo, (__m128i *)u_lo);
  const __m256i urow_hi = _mm256_loadu2_m128i((__m128i *)u_hi, (__m128i *)u_hi);
  __m256i *ap = (__m256i *)a, *ae = (__m256i *)(a + k);
  __m256i *bp = (__m256i *)b;
  for (; ap < ae; ap++, bp++) {
    __m256i bx = _mm256_loadu_si256(bp);
    __m256i lo = _mm256_and_si256(bx, mask);
    bx = _mm256_srli_epi64(bx, 4);
    __m256i hi = _mm256_and_si256(bx, mask);
    lo = _mm256_shuffle_epi8(urow_lo, lo);
    hi = _mm256_shuffle_epi8(urow_hi, hi);
    _mm256_storeu_si256(
        ap, _mm256_xor_si256(_mm256_loadu_si256(ap), _mm256_xor_si256(lo, hi)));
  }
}

void oblas_scal(uint8_t *a, size_t k, const uint8_t *u_lo,
                const uint8_t *u_hi) {
  const __m256i mask = _mm256_set1_epi8(0x0f);
  const __m256i urow_lo = _mm256_loadu2_m128i((__m128i *)u_lo, (__m128i *)u_lo);
  const __m256i urow_hi = _mm256_loadu2_m128i((__m128i *)u_hi, (__m128i *)u_hi);
  __m256i *ap = (__m256i *)a, *ae = (__m256i *)(a + k);
  for (; ap < ae; ap++) {
    __m256i ax = _mm256_loadu_si256(ap);
    __m256i lo = _mm256_and_si256(ax, mask);
    ax = _mm256_srli_epi64(ax, 4);
    __m256i hi = _mm256_and_si256(ax, mask);
    lo = _mm256_shuffle_epi8(urow_lo, lo);
    hi = _mm256_shuffle_epi8(urow_hi, hi);
    _mm256_storeu_si256(ap, _mm256_xor_si256(lo, hi));
  }
}

void oblas_swap(uint8_t *a, uint8_t *b, size_t k) {
  __m256i *ap = (__m256i *)a, *ae = (__m256i *)(a + k);
  __m256i *bp = (__m256i *)b;
  for (; ap < ae; ap++, bp++) {
    __m256i atmp = _mm256_loadu_si256((__m256i *)(ap));
    __m256i btmp = _mm256_loadu_si256((__m256i *)(bp));
    _mm256_storeu_si256(ap, btmp);
    _mm256_storeu_si256(bp, atmp);
  }
}

void oblas_axpy_gf2_gf256_32(uint8_t *a, uint32_t *b, size_t k, uint8_t u) {
  __m256i *ap = (__m256i *)a, *ae = (__m256i *)(a + k);
  __m256i scatter =
      _mm256_set_epi32(0x03030303, 0x03030303, 0x02020202, 0x02020202,
                       0x01010101, 0x01010101, 0x00000000, 0x00000000);
  __m256i cmpmask =
      _mm256_set_epi32(0x80402010, 0x08040201, 0x80402010, 0x08040201,
                       0x80402010, 0x08040201, 0x80402010, 0x08040201);
  __m256i up = _mm256_set1_epi8(u);
  for (unsigned p = 0; ap < ae; p++, ap++) {
    __m256i bcast = _mm256_set1_epi32(b[p]);
    __m256i ret = _mm256_shuffle_epi8(bcast, scatter);
    ret = _mm256_andnot_si256(ret, cmpmask);
    ret = _mm256_and_si256(_mm256_cmpeq_epi8(ret, _mm256_setzero_si256()), up);
    _mm256_storeu_si256(ap, _mm256_xor_si256(_mm256_loadu_si256(ap), ret));
  }
}
