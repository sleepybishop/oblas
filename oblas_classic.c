#include "oblas.h"
#include "octmul_hilo.h"

void ocopy(uint8_t *restrict a, uint8_t *restrict b, size_t i, size_t j,
           size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  for (size_t idx = 0; idx < k; idx++) {
    ap[idx] = bp[idx];
  }
}

void oswaprow(uint8_t *restrict a, size_t i, size_t j, size_t k) {
  if (i == j)
    return;
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = a + (j * ALIGNED_COLS(k));

  for (size_t idx = 0; idx < k; idx++) {
    OCTET_SWAP(ap[idx], bp[idx]);
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

  const octet *urow_hi = OCT_MUL_HI[u];
  const octet *urow_lo = OCT_MUL_LO[u];
  for (size_t idx = 0; idx < k; idx++) {
    octet b_lo = bp[idx] & 0x0f;
    octet b_hi = (bp[idx] & 0xf0) >> 4;
    ap[idx] ^= urow_hi[b_hi] ^ urow_lo[b_lo];
  }
}

void oaddrow(uint8_t *restrict a, uint8_t *restrict b, size_t i, size_t j,
             size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  octet *bp = b + (j * ALIGNED_COLS(k));

  for (size_t idx = 0; idx < k; idx++) {
    ap[idx] ^= bp[idx];
  }
}

void oscal(uint8_t *restrict a, size_t i, size_t k, uint8_t u) {
  octet *ap = a + (i * ALIGNED_COLS(k));

  if (u < 2)
    return;

  const octet *urow_hi = OCT_MUL_HI[u];
  const octet *urow_lo = OCT_MUL_LO[u];
  for (size_t idx = 0; idx < k; idx++) {
    octet a_lo = ap[idx] & 0x0f;
    octet a_hi = (ap[idx] & 0xf0) >> 4;
    ap[idx] = urow_hi[a_hi] ^ urow_lo[a_lo];
  }
}

void ozero(uint8_t *restrict a, size_t i, size_t k) {
  octet *ap = a + (i * ALIGNED_COLS(k));
  for (size_t idx = 0; idx < k; idx++)
    ap[idx] = 0;
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
