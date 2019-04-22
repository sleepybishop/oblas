CPPFLAGS= -D_DEFAULT_SOURCE
#CPPFLAGS= -D_DEFAULT_SOURCE -DOBLAS_NEON -mfpu=neon
#CPPFLAGS= -D_DEFAULT_SOURCE -DOBLAS_SSE
#CPPFLAGS= -D_DEFAULT_SOURCE -DOBLAS_AVX -DOCTMAT_ALIGN=32

CFLAGS  = -O3 -std=c99 -Wall -march=native -funroll-loops

OBJ=oblas.o octmat.o sparsemat.o

all: liboblas.a

#tablegen: tablegen.o

#octtables.h: tablegen
#	./$< > $@
#	clang-format -style=LLVM -i $@

oblas.o: oblas.c oblas.h octtables.h

liboblas.a: $(OBJ)
	$(AR) rcs $@ $^

.PHONY: clean indent scan
clean:
	$(RM) *.o *.a

indent:
	clang-format -style=LLVM -i *.c *.h

scan:
	scan-build $(MAKE) clean all

