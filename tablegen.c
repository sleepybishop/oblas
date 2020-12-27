#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { GF_2_2 = 0, GF_2_4 = 1, GF_2_8 = 2 };

enum {
  POLY_GF_2_2 = 7,   /*                      x^2 + x + 1 */
  POLY_GF_2_4 = 19,  /*          x^4             + x + 1 */
  POLY_GF_2_8 = 285, /* x^8    + x^4 + x^3 + x^2     + 1 */
};

typedef struct {
  uint8_t field;
  uint8_t exp;
  unsigned len;
  unsigned poly;
} gf;

typedef struct {
  uint8_t EXP[UINT8_MAX];
  uint8_t LOG[UINT8_MAX + 1];
} gftbl;

static gf fields[] = {
    {.field = GF_2_2, .exp = 2, .len = 4, .poly = POLY_GF_2_2},
    {.field = GF_2_4, .exp = 4, .len = 16, .poly = POLY_GF_2_4},
    {.field = GF_2_8, .exp = 8, .len = 256, .poly = POLY_GF_2_8},
};

void fill_tabs(gf field, gftbl *tabs) {
  uint8_t o = 1;
  for (int i = 0; i < field.exp; i++, o <<= 1) {
    tabs->EXP[i] = o;
    tabs->EXP[field.exp] |= field.poly & (1 << i);
  }

  o = 1 << (field.exp - 1);
  for (int i = (field.exp + 1); i < (field.len - 1); i++) {
    if (tabs->EXP[i - 1] >= o)
      tabs->EXP[i] = tabs->EXP[field.exp] ^ ((tabs->EXP[i - 1]) << 1);
    else
      tabs->EXP[i] = tabs->EXP[i - 1] << 1;
    tabs->EXP[i] = tabs->EXP[i] % field.len;
  }

  /* invert exp table */
  tabs->LOG[0] = field.len - 1;
  for (int i = 0; i < (field.len - 1); i++)
    tabs->LOG[tabs->EXP[i]] = i;
}

void print_tabs(FILE *stream, gf field, gftbl *tabs) {
  fprintf(stream, "#ifndef OCT_TABLES\n#define OCT_TABLES\n\n");
  fprintf(stream, "static const uint8_t OCT_LOG[] = {");
  for (int i = 0; i < field.len; i++)
    fprintf(stream, "%d,", tabs->LOG[i]);
  fprintf(stream, "};\n\n");

  fprintf(stream, "static const uint8_t OCT_EXP[] = {\n");
  for (int i = 0; i < 2 * (field.len - 1); i++)
    fprintf(stream, "%d,", tabs->EXP[i % (field.len - 1)]);

  fprintf(stream, "};\n\n");
  fprintf(stream, "static const uint16_t OCT_EXP_SIZE = sizeof(OCT_EXP) / "
                  "sizeof(OCT_EXP[0]);\n\n");

  fprintf(stream, "static const uint8_t OCT_INV[] = {");
  for (int i = 0; i < field.len; i++) {
    switch (i) {
    case 0:
      fprintf(stream, "%d,", 0);
      break;
    case 1:
      fprintf(stream, "%d,", 1);
      break;
    default:
      fprintf(stream, "%d,", tabs->EXP[tabs->LOG[0] - tabs->LOG[i]]);
    }
  }
  fprintf(stream, "};\n\n");

  fprintf(stream, "\n\n");
  fprintf(stream, "#endif\n");
}

int main(int argc, char *argv[]) {
  gftbl tabs;
  fill_tabs(fields[GF_2_8], &tabs);
  print_tabs(stdout, fields[GF_2_8], &tabs);
  return 0;
}
