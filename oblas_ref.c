void oblas_axpy(uint8_t *a, const uint8_t *b, size_t k, const uint8_t *u_lo,
                const uint8_t *u_hi) {
  for (int idx = 0; idx < k; idx++) {
    uint8_t b_lo = b[idx] & 0x0f;
    uint8_t b_hi = b[idx] >> 4;
    a[idx] ^= u_hi[b_hi] ^ u_lo[b_lo];
  }
}

void oblas_scal(uint8_t *a, size_t k, const uint8_t *u_lo,
                const uint8_t *u_hi) {
  for (int idx = 0; idx < k; idx++) {
    uint8_t a_lo = a[idx] & 0x0f;
    uint8_t a_hi = a[idx] >> 4;
    a[idx] = u_hi[a_hi] ^ u_lo[a_lo];
  }
}

void oblas_axpy_gf2_gf256_32(uint8_t *a, uint32_t *b, size_t k, uint8_t u) {
  for (size_t idx = 0, p = 0; idx < k; idx += 32, p++) {
    uint32_t tmp = b[p];
    while (tmp > 0) {
      int tz = __builtin_ctz(tmp);
      tmp = tmp & (tmp - 1);
      a[tz + idx] ^= u;
    }
  }
}
