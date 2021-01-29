#include <immintrin.h> /* AVX512 */

void oblas_axpy(uint8_t *a, const uint8_t *b, size_t k, const uint8_t *u_lo,
                const uint8_t *u_hi) {
  const __m512i mask = _mm512_set1_epi8(0x0f);
  const __m128i ulo_128 = _mm_loadu_si128((__m128i *)u_lo);
  const __m128i uhi_128 = _mm_loadu_si128((__m128i *)u_hi);
  const __m512i urow_lo = _mm512_broadcast_i32x4(ulo_128);
  const __m512i urow_hi = _mm512_broadcast_i32x4(uhi_128);
  __m512i *ap = (__m512i *)a, *ae = (__m512i *)(a + k);
  __m512i *bp = (__m512i *)b;
  for (; ap < ae; ap++, bp++) {
    __m512i bx = _mm512_loadu_si512(bp);
    __m512i lo = _mm512_and_si512(bx, mask);
    bx = _mm512_srli_epi64(bx, 4);
    __m512i hi = _mm512_and_si512(bx, mask);
    lo = _mm512_shuffle_epi8(urow_lo, lo);
    hi = _mm512_shuffle_epi8(urow_hi, hi);
    _mm512_storeu_si512(
        ap, _mm512_xor_si512(_mm512_loadu_si512(ap), _mm512_xor_si512(lo, hi)));
  }
}

void oblas_scal(uint8_t *a, size_t k, const uint8_t *u_lo,
                const uint8_t *u_hi) {
  const __m512i mask = _mm512_set1_epi8(0x0f);
  const __m128i ulo_128 = _mm_loadu_si128((__m128i *)u_lo);
  const __m128i uhi_128 = _mm_loadu_si128((__m128i *)u_hi);
  const __m512i urow_lo = _mm512_broadcast_i32x4(ulo_128);
  const __m512i urow_hi = _mm512_broadcast_i32x4(uhi_128);
  __m512i *ap = (__m512i *)a, *ae = (__m512i *)(a + k);
  for (; ap < ae; ap++) {
    __m512i ax = _mm512_loadu_si512(ap);
    __m512i lo = _mm512_and_si512(ax, mask);
    ax = _mm512_srli_epi64(ax, 4);
    __m512i hi = _mm512_and_si512(ax, mask);
    lo = _mm512_shuffle_epi8(urow_lo, lo);
    hi = _mm512_shuffle_epi8(urow_hi, hi);
    _mm512_storeu_si512(ap, _mm512_xor_si512(lo, hi));
  }
}

void oblas_axpy_gf2_gf256_32(uint8_t *a, uint32_t *b, size_t k, uint8_t u) {
  __m512i *ap = (__m512i *)a, *ae = (__m512i *)(a + k);
  __m512i scatter = _mm512_set_epi32(
      0x03030303, 0x03030303, 0x02020202, 0x02020202, 0x01010101, 0x01010101,
      0x00000000, 0x00000000, 0x03030303, 0x03030303, 0x02020202, 0x02020202,
      0x01010101, 0x01010101, 0x00000000, 0x00000000);
  __m512i cmpmask = _mm512_set_epi32(
      0x80402010, 0x08040201, 0x80402010, 0x08040201, 0x80402010, 0x08040201,
      0x80402010, 0x08040201, 0x80402010, 0x08040201, 0x80402010, 0x08040201,
      0x80402010, 0x08040201, 0x80402010, 0x08040201);
  __m512i up = _mm512_set1_epi8(u);
  for (unsigned p = 0; ap < ae; p++, ap++) {
    __m512i bcast = _mm512_set1_epi32(b[p]);
    __m512i ret = _mm512_shuffle_epi8(bcast, scatter);
    ret = _mm512_andnot_si512(ret, cmpmask);
    __mmask64 tmp = _mm512_cmpeq_epi8_mask(ret, _mm512_setzero_si512());
    ret = _mm512_mask_blend_epi8(tmp, _mm512_setzero_si512(), up);
    _mm512_storeu_si512(ap, _mm512_xor_si512(_mm512_loadu_si512(ap), ret));
  }
}
