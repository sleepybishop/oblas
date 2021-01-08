#include "octmat.h"
#include "oblas_compat.h"
#include "util.h"

void om_resize(octmat *v, size_t rows, size_t cols) {
  void *aligned = NULL;

  v->rows = rows;
  v->cols = cols;
  v->cols_al = ALIGNED_COLS(cols);
  aligned = (uint8_t *)oblas_alloc(v->rows, v->cols_al, OCTMAT_ALIGN);
  memset(aligned, 0, v->cols_al * rows);
  v->data = aligned;
}

void om_destroy(octmat *v) {
  v->rows = 0;
  v->cols = 0;
  v->cols_al = 0;
  oblas_free(v->data);
  v->data = NULL;
}
