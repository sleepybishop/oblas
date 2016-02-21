#include "oblas.h"
#include "stdio.h"
#include "stdlib.h"

typedef uint8_t octet;

void ocopy(uint8_t *restrict a, uint8_t *restrict b, uint16_t i, uint16_t j,
           uint16_t k) {
  octet *ap = a + (i * k);
  octet *bp = b + (j * k);

  for (int idx = 0; idx < k; idx++) {
    ap[idx] = bp[idx];
  }
}

void oswaprow(uint8_t *restrict a, uint16_t i, uint16_t j, uint16_t k) {
  if (i == j)
    return;
  octet *ap = a + (i * k);
  octet *bp = a + (j * k);

  for (int idx = 0; idx < k; idx++) {
    OCTET_SWAP(ap[idx], bp[idx]);
  }
}

void oswapcol(octet *restrict a, uint16_t i, uint16_t j, uint16_t k,
              uint16_t l) {
  if (i == j)
    return;
  octet *ap = a;

  for (int idx = 0; idx < k; idx++, ap += l) {
    OCTET_SWAP(ap[i], ap[j]);
  }
}

void oaxpy(uint8_t *restrict a, uint8_t *restrict b, uint16_t i, uint16_t j,
           uint16_t k, uint8_t u) {
  octet *ap = a + (i * k);
  octet *bp = b + (j * k);

  if (u == 0)
    return;

  octet u_log = OCT_LOG[u];
  for (int idx = 0; idx < k; idx++) {
    if (bp[idx] == 0)
      continue;
    ap[idx] ^= OCT_EXP[u_log + OCT_LOG[bp[idx]]];
  }
}

void oaddrow(uint8_t *restrict a, uint8_t *restrict b, uint16_t i, uint16_t j,
             uint16_t k) {
  octet *ap = a + (i * k);
  octet *bp = b + (j * k);

  for (int idx = 0; idx < k; idx++) {
    ap[idx] ^= bp[idx];
  }
}

void odivrow(uint8_t *restrict a, uint16_t i, uint16_t k, uint8_t u) {
  octet *ap = a + (i * k);

  octet u_log = OCT_LOG[u];

  for (int idx = 0; idx < k; idx++) {
    if (u == 0 || ap[idx] == 0)
      continue;
    ap[idx] = OCT_EXP[OCT_LOG[ap[idx]] - u_log + 255];
  }
}

void ogemm(uint8_t *restrict a, uint8_t *restrict b, uint8_t *restrict c,
           uint16_t n, uint16_t k, uint16_t m) {
  octet *bp = b, *cp = c;
  octet *a_log = malloc(k);
  octet *b_log = malloc(m * k);
  octet *blp = b_log;

  for (int col = 0; col < m; col++, bp = b, blp += k) {
    for (int idx = 0; idx < k; idx++, bp += m) {
      blp[idx] = OCT_LOG[bp[col]];
    }
  }

  for (int row = 0; row < n; row++, cp += m) {
    octet *ap = a + row * k;
    blp = b_log;

    for (int idx = 0; idx < k; idx++) {
      a_log[idx] = OCT_LOG[ap[idx]];
    }

    for (int col = 0; col < m; col++, blp += k) {
      octet acc = 0;

      for (int idx = 0; idx < k; idx++) {
        if (ap[idx] == 0 || blp[idx] == 255)
          continue;
        acc ^= OCT_EXP[a_log[idx] + blp[idx]];
      }
      cp[col] = acc;
    }
  }
  free(a_log);
  free(b_log);
}
