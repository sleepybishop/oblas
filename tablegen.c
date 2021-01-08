#include "oblas.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum {
  POLY_GF2_1 = 3,   /*                            x + 1 */
  POLY_GF2_2 = 7,   /*                      x^2 + x + 1 */
  POLY_GF2_4 = 19,  /*          x^4             + x + 1 */
  POLY_GF2_8 = 285, /* x^8    + x^4 + x^3 + x^2     + 1 */
};

static const gf fields[] = {
    {.field = GF2_1, .exp = 1, .len = 1 << 1, .poly = POLY_GF2_1},
    {.field = GF2_2, .exp = 2, .len = 1 << 2, .poly = POLY_GF2_2},
    {.field = GF2_4, .exp = 4, .len = 1 << 4, .poly = POLY_GF2_4},
    {.field = GF2_8, .exp = 8, .len = 1 << 8, .poly = POLY_GF2_8},
};

typedef struct {
  uint8_t EXP[UINT8_MAX];
  uint8_t LOG[UINT8_MAX + 1];
  uint8_t SHUF_LO[UINT8_MAX + 1][16];
  uint8_t SHUF_HI[UINT8_MAX + 1][16];
} gftbl;

void fill_tabs(const gf_field f, gftbl *tabs) {
  uint8_t o = 1;
  gf field = fields[f];
  tabs->EXP[field.exp] = 0;
  for (int i = 0; i < field.exp; i++, o <<= 1) {
    tabs->EXP[i] = o;
    tabs->EXP[field.exp] |= field.poly & (1 << i);
  }

  assert(field.exp > 0);
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

void fill_shuffle_tabs(const gf_field f, gftbl *tabs) {
  gf field = fields[f];

  for (int i = 0; i < field.len; i++) {
    for (int j = 0; j < 16; j++) {
      tabs->SHUF_LO[i][j] = 0;
      tabs->SHUF_HI[i][j] = 0;
      if (i == 0 || j == 0)
        continue;
      switch (f) {
      case GF2_2:
        tabs->SHUF_LO[i][j] =
            tabs->EXP[(tabs->LOG[i] + tabs->LOG[j / field.len]) %
                      (field.len - 1)];
        tabs->SHUF_LO[i][j] <<= field.exp;
        tabs->SHUF_LO[i][j] |=
            tabs->EXP[(tabs->LOG[i] + tabs->LOG[j % field.len]) %
                      (field.len - 1)];
        tabs->SHUF_HI[i][j] = tabs->SHUF_LO[i][j] << field.len;
        break;
      case GF2_4:
        tabs->SHUF_LO[i][j] =
            tabs->EXP[(tabs->LOG[i] + tabs->LOG[j]) % (field.len - 1)];
        tabs->SHUF_HI[i][j] = tabs->SHUF_LO[i][j] << field.exp;
        break;
      case GF2_8:
        tabs->SHUF_LO[i][j] =
            tabs->EXP[(tabs->LOG[i] + tabs->LOG[j]) % (field.len - 1)];
        tabs->SHUF_HI[i][j] =
            tabs->EXP[(tabs->LOG[i] + tabs->LOG[j << 4]) % (field.len - 1)];
        break;
      default:
        break;
      }
    }
  }
}

void print_log_tab(FILE *stream, const gf field, const gftbl *tabs) {
  fprintf(stream, "{\n");
  for (int i = 0; i < field.len; i++) {
    fprintf(stream, "%3d,", tabs->LOG[i]);
    if (i && (i % 16 == 15))
      fprintf(stream, "\n");
  }
  fprintf(stream, "};\n\n");
}

void print_exp_tab(FILE *stream, const gf field, const gftbl *tabs) {
  fprintf(stream, "{\n");
  for (int i = 0; i < 2 * (field.len - 1); i++) {
    fprintf(stream, "%3d,", tabs->EXP[i % (field.len - 1)]);
    if (i && (i % 16 == 15))
      fprintf(stream, "\n");
  }
  fprintf(stream, "};\n\n");
}

void print_inv_tab(FILE *stream, const gf field, const gftbl *tabs) {
  fprintf(stream, "{\n");
  for (int i = 0; i < field.len; i++) {
    switch (i) {
    case 0:
      fprintf(stream, "%3d,", 0);
      break;
    case 1:
      fprintf(stream, "%3d,", 1);
      break;
    default:
      fprintf(stream, "%3d,", tabs->EXP[tabs->LOG[0] - tabs->LOG[i]]);
    }
    if (i && (i % 16 == 15))
      fprintf(stream, "\n");
  }
  fprintf(stream, "};\n\n");
}

void print_shuf_lo_tab(FILE *stream, const gf field, const gftbl *tabs) {
  fprintf(stream, "{\n");
  for (int i = 0; i < field.len; i++) {
    fprintf(stream, "{");
    for (int j = 0; j < 16; j++) {
      fprintf(stream, "%3d,", tabs->SHUF_LO[i][j]);
    }
    fprintf(stream, "},\n");
  }
  fprintf(stream, "};\n\n");
}

void print_shuf_hi_tab(FILE *stream, const gf field, const gftbl *tabs) {
  fprintf(stream, "{\n");
  for (int i = 0; i < field.len; i++) {
    fprintf(stream, "{");
    for (int j = 0; j < 16; j++) {
      fprintf(stream, "%3d,", tabs->SHUF_HI[i][j]);
    }
    fprintf(stream, "},\n");
  }
  fprintf(stream, "};\n\n");
}

void print_tabs(FILE *stream, const gf field, const gftbl *tabs) {
  char *pfx[] = {"GF2", "GF4", "GF16", "GF256"}, *prefix = pfx[field.field];

  fprintf(stream, "/* these tables were generated with polynomial: %u */\n\n",
          field.poly);
  fprintf(stream, "#ifndef %s_TABLES\n#define %s_TABLES\n\n", prefix, prefix);
  fprintf(stream, "/* clang-format off */\n");

  fprintf(stream, "static const uint8_t %s_LOG[] = \n", prefix);
  print_log_tab(stream, field, tabs);

  fprintf(stream, "static const uint8_t %s_EXP[] = \n", prefix);
  print_exp_tab(stream, field, tabs);

  fprintf(stream, "static const uint8_t %s_INV[] = \n", prefix);
  print_inv_tab(stream, field, tabs);

  fprintf(stream, "static const uint8_t %s_SHUF_LO[%d][16] = \n", prefix,
          field.len);
  print_shuf_lo_tab(stream, field, tabs);

  fprintf(stream, "static const uint8_t %s_SHUF_HI[%d][16] = \n", prefix,
          field.len);
  print_shuf_hi_tab(stream, field, tabs);

  fprintf(stream, "/* clang-format on */\n");
  fprintf(stream, "#endif\n");
}

int main(int argc, char *argv[]) {
  gftbl tabs;
  gf_field field;
  if (argc != 2)
    return 0;
  unsigned exp = strtol(argv[1], NULL, 10);
  switch (exp) {
  case 2:
    field = GF2_2;
    break;
  case 4:
    field = GF2_4;
    break;
  case 8:
    field = GF2_8;
    break;
  default:
    return -1;
  }

  fill_tabs(field, &tabs);
  fill_shuffle_tabs(field, &tabs);
  print_tabs(stdout, fields[field], &tabs);

  return 0;
}
