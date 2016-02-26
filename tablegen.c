#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint8_t prim[] = "101110001";
/* 1 + x^2 + x^3 + x^4 + x^8 */

static uint8_t OCT_EXP[UINT8_MAX];
static uint8_t OCT_LOG[UINT8_MAX + 1];

int main(int argc, char *argv[]) {

  OCT_LOG[8] = 0;

  uint8_t o = 1;
  for (int i = 0; i < 8; i++, o <<= 1) {
    OCT_EXP[i] = o;

    OCT_LOG[OCT_EXP[i]] = i;

    if (prim[i] == '1')
      OCT_EXP[8] |= o;
  }
  OCT_LOG[OCT_EXP[8]] = 8;

  o = 1 << 7;
  for (int i = 9; i < UINT8_MAX; i++) {
    if (OCT_EXP[i - 1] >= o)
      OCT_EXP[i] = OCT_EXP[8] ^ ((OCT_EXP[i - 1]) << 1);
    else
      OCT_EXP[i] = OCT_EXP[i - 1] << 1;

    OCT_LOG[OCT_EXP[i]] = i;
  }
  OCT_LOG[0] = UINT8_MAX;

  fprintf(stdout, "static const uint8_t OCT_LOG[] = {");
  for (int i = 0; i <= UINT8_MAX; i++) {
    fprintf(stdout, "%d,%c", OCT_LOG[i], (i % 16) ? ' ' : '\n');
  }
  fprintf(stdout, "};\n\n");

  fprintf(stdout, "static const uint8_t OCT_EXP[] = {\n");
  for (int i = 0; i < UINT8_MAX; i++) {
    fprintf(stdout, "%d,%c", OCT_EXP[i], (i % 16) ? ' ' : '\n');
  }
  /* exp table is repeated to avoid modulo ops on div/mul */
  for (int i = 0; i < UINT8_MAX; i++) {
    fprintf(stdout, "%d,%c", OCT_EXP[i], (i % 16) ? ' ' : '\n');
  }
  fprintf(stdout, "};\n\n");
  fprintf(stdout, "static const uint16_t OCT_EXP_SIZE = sizeof(OCT_EXP) / "
                  "sizeof(OCT_EXP[0]);\n\n");

  fprintf(stdout, "static const uint8_t OCT_MUL[256][256] = {");
  for (int i = 0; i <= UINT8_MAX; i++) {
    fprintf(stdout, "\n{");
    for (int j = 0; j <= UINT8_MAX; j++) {
      if (i == 0 || j == 0) {
        fprintf(stdout, "%d,", 0);
      } else {
        fprintf(stdout, "%d,", OCT_EXP[(OCT_LOG[i] + OCT_LOG[j]) % 255]);
      }
    }

    fprintf(stdout, "},\n");
  }
  fprintf(stdout, "};\n\n");
  return 0;
}
