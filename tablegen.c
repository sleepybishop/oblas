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

void fill_tabs(const gf field, gftbl *tabs) {
  uint8_t o = 1;
  tabs->EXP[field.exp] = 0;
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

void print_log_tab(FILE *stream, const gf field, gftbl *tabs) {
  fprintf(stream, "{\n");
  for (int i = 0; i < field.len; i++) {
    fprintf(stream, "%3d,", tabs->LOG[i]);
    if (i && (i % 16 == 15))
      fprintf(stream, "\n");
  }
  fprintf(stream, "};\n\n");
}

void print_exp_tab(FILE *stream, const gf field, gftbl *tabs) {
  fprintf(stream, "{\n");
  for (int i = 0; i < 2 * (field.len - 1); i++) {
    fprintf(stream, "%3d,", tabs->EXP[i % (field.len - 1)]);
    if (i && (i % 16 == 15))
      fprintf(stream, "\n");
  }
  fprintf(stream, "};\n\n");
}

void print_inv_tab(FILE *stream, const gf field, gftbl *tabs) {
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

void print_shuf_lo_tab(FILE *stream, const gf field, gftbl *tabs) {
  fprintf(stream, "{\n");
  for (int i = 0; i < field.len; i++) {
    fprintf(stream, "{");
    for (int j = 0; j < 16; j++) {
      if (i == 0 || j == 0) {
        fprintf(stream, "%3d,", 0);
      } else {
        fprintf(stream, "%3d,",
                tabs->EXP[(tabs->LOG[i] + tabs->LOG[j]) % (field.len - 1)]);
      }
    }
    fprintf(stream, "},\n");
  }
  fprintf(stream, "};\n\n");
}

void print_shuf_hi_tab(FILE *stream, const gf field, gftbl *tabs) {
  fprintf(stream, "{\n");
  for (int i = 0; i < field.len; i++) {
    fprintf(stream, "{");
    for (int j = 0; j < 16; j++) {
      if (i == 0 || j == 0) {
        fprintf(stream, "%3d,", 0);
      } else {
        fprintf(
            stream, "%3d,",
            tabs->EXP[(tabs->LOG[i] + tabs->LOG[j << 4]) % (field.len - 1)]);
      }
    }
    fprintf(stream, "},\n");
  }
  fprintf(stream, "};\n\n");
}

void print_tabs(FILE *stream, const gf field, gftbl *tabs) {
  fprintf(stream, "#ifndef OCT_TABLES\n#define OCT_TABLES\n\n");

  fprintf(stream, "static const uint8_t OCT_LOG[] = \n");
  print_log_tab(stream, field, tabs);

  fprintf(stream, "static const uint8_t OCT_EXP[] = \n");
  print_exp_tab(stream, field, tabs);

  fprintf(stream, "static const uint8_t OCT_INV[] = \n");
  print_inv_tab(stream, field, tabs);

  fprintf(stream, "static const uint8_t OCT_MUL_LO[%d][16] = \n", field.len);
  print_shuf_lo_tab(stream, field, tabs);

  fprintf(stream, "static const uint8_t OCT_MUL_HI[%d][16] = \n", field.len);
  print_shuf_hi_tab(stream, field, tabs);

  fprintf(stream, "#endif\n");
}

int main(int argc, char *argv[]) {
  gftbl tabs;
  gf field = fields[GF_2_8];
  fill_tabs(field, &tabs);
  print_tabs(stdout, field, &tabs);

  return 0;
}
