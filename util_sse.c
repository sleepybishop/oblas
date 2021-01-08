#include <emmintrin.h> /* SSE2 */
#include <tmmintrin.h> /* SSE3 */

void oblas_xor(uint8_t *a, uint8_t *b, size_t k) {
  __m128i *ap = (__m128i *)a, *ae = (__m128i *)(a + k);
  __m128i *bp = (__m128i *)b;
  for (; ap < ae; ap++, bp++) {
    _mm_storeu_si128(ap,
                     _mm_xor_si128(_mm_loadu_si128(ap), _mm_loadu_si128(bp)));
  }
}

void oblas_axpy(uint8_t *a, const uint8_t *b, size_t k, const uint8_t *u_lo,
                const uint8_t *u_hi) {
  const __m128i mask = _mm_set1_epi8(0x0f);
  const __m128i urow_lo = _mm_loadu_si128((__m128i *)u_lo);
  const __m128i urow_hi = _mm_loadu_si128((__m128i *)u_hi);
  __m128i *ap = (__m128i *)a, *ae = (__m128i *)(a + k);
  __m128i *bp = (__m128i *)b;
  for (; ap < ae; ap++, bp++) {
    __m128i bx = _mm_loadu_si128(bp);
    __m128i lo = _mm_and_si128(bx, mask);
    bx = _mm_srli_epi64(bx, 4);
    __m128i hi = _mm_and_si128(bx, mask);
    lo = _mm_shuffle_epi8(urow_lo, lo);
    hi = _mm_shuffle_epi8(urow_hi, hi);
    _mm_storeu_si128(ap,
                     _mm_xor_si128(_mm_loadu_si128(ap), _mm_xor_si128(lo, hi)));
  }
}

void oblas_scal(uint8_t *a, size_t k, const uint8_t *u_lo,
                const uint8_t *u_hi) {
  const __m128i mask = _mm_set1_epi8(0x0f);
  const __m128i urow_lo = _mm_loadu_si128((__m128i *)u_lo);
  const __m128i urow_hi = _mm_loadu_si128((__m128i *)u_hi);
  __m128i *ap = (__m128i *)a, *ae = (__m128i *)(a + k);
  for (; ap < ae; ap++) {
    __m128i ax = _mm_loadu_si128(ap);
    __m128i lo = _mm_and_si128(ax, mask);
    ax = _mm_srli_epi64(ax, 4);
    __m128i hi = _mm_and_si128(ax, mask);
    lo = _mm_shuffle_epi8(urow_lo, lo);
    hi = _mm_shuffle_epi8(urow_hi, hi);
    _mm_storeu_si128(ap, _mm_xor_si128(lo, hi));
  }
}

void oblas_swap(uint8_t *a, uint8_t *b, size_t k) {
  __m128i *ap = (__m128i *)a, *ae = (__m128i *)(a + k);
  __m128i *bp = (__m128i *)b;
  for (; ap < ae; ap++, bp++) {
    __m128i atmp = _mm_loadu_si128((__m128i *)(ap));
    __m128i btmp = _mm_loadu_si128((__m128i *)(bp));
    _mm_storeu_si128(ap, btmp);
    _mm_storeu_si128(bp, atmp);
  }
}

void oblas_axpy_gf2_gf256_32(uint8_t *a, uint32_t *b, size_t k, uint8_t u) {
  __m128i *ap = (__m128i *)a, *ae = (__m128i *)(a + k);
  __m128i scatter_hi =
      _mm_set_epi32(0x03030303, 0x03030303, 0x02020202, 0x02020202);
  __m128i scatter_lo =
      _mm_set_epi32(0x01010101, 0x01010101, 0x00000000, 0x00000000);
  __m128i cmpmask =
      _mm_set_epi32(0x80402010, 0x08040201, 0x80402010, 0x08040201);
  __m128i up = _mm_set1_epi8(u);
  for (unsigned p = 0; ap < ae; p++, ap++) {
    __m128i bcast = _mm_set1_epi32(b[p]);
    __m128i ret_lo = _mm_shuffle_epi8(bcast, scatter_lo);
    __m128i ret_hi = _mm_shuffle_epi8(bcast, scatter_hi);
    ret_lo = _mm_andnot_si128(ret_lo, cmpmask);
    ret_hi = _mm_andnot_si128(ret_hi, cmpmask);
    ret_lo = _mm_and_si128(_mm_cmpeq_epi8(ret_lo, _mm_setzero_si128()), up);
    ret_hi = _mm_and_si128(_mm_cmpeq_epi8(ret_hi, _mm_setzero_si128()), up);
    _mm_storeu_si128(ap, _mm_xor_si128(_mm_loadu_si128(ap), ret_lo));
    ap++;
    _mm_storeu_si128(ap, _mm_xor_si128(_mm_loadu_si128(ap), ret_hi));
  }
}
