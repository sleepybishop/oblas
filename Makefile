CFLAGS  = -D_DEFAULT_SOURCE -O3 -g -std=c99 -Wall -march=native 
CFLAGS += -funroll-loops -ftree-vectorize -fno-inline 
#CFLAGS += -fopt-info-vec

OBJ=oblas.o octmat.o gf2.o

all: liboblas.a

tablegen: tablegen.c

octtables.h: tablegen
	./$< > $@
	clang-format -style=LLVM -i $@

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

