# oblas

blas-like routines to solve systems in finite fields [gf2, gf4, gf16 and gf256]

#### Optimizing for different archs
 - NEON: `make CPPFLAGS+="-DOBLAS_NEON"`
 - SSE: `make CPPFLAGS+="-DOBLAS_SSE"`
 - AVX: `make CPPFLAGS+="-DOBLAS_AVX -DOCTMAT_ALIGN=32"`

#### Customizing
Edit `tablegen.c` to change polynomial/field size.

